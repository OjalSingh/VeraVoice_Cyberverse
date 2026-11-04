"""Inference: load a trained bundle and score audio."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import torch

from .audio import chunks, load_audio, preprocess, to_mono_16k
from .embed import WavLMEmbedder
from .head import LayerWeightedHead

HEAD_FILE = "head.pt"
CONFIG_FILE = "config.json"


@dataclass
class Result:
    spoof_probability: float
    verdict: str  # "likely human", "likely synthetic" or "uncertain"
    chunk_probabilities: list[float]


def save_bundle(out_dir: str, head: LayerWeightedHead, backbone: str, threshold: float, metrics: dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    torch.save(head.state_dict(), os.path.join(out_dir, HEAD_FILE))
    with open(os.path.join(out_dir, CONFIG_FILE), "w") as f:
        json.dump({"backbone": backbone, "head": head.config, "threshold": threshold, "metrics": metrics}, f, indent=2)


class Detector:
    def __init__(self, bundle_dir: str, embedder: WavLMEmbedder | None = None, device: str | None = None):
        with open(os.path.join(bundle_dir, CONFIG_FILE)) as f:
            self.config = json.load(f)
        self.embedder = embedder or WavLMEmbedder(self.config["backbone"], device=device)
        self.head = LayerWeightedHead(**self.config["head"])
        state = torch.load(os.path.join(bundle_dir, HEAD_FILE), map_location="cpu", weights_only=True)
        self.head.load_state_dict(state)
        self.head.eval()
        self.threshold = float(self.config.get("threshold", 0.5))

    @torch.inference_mode()
    def predict_waveform(self, y: np.ndarray, sr: int, margin: float = 0.15) -> Result:
        y = to_mono_16k(y, sr)
        windows = [preprocess(c) for c in chunks(y)]
        feats = torch.from_numpy(self.embedder(windows))
        probs = torch.sigmoid(self.head(feats)).numpy().tolist()
        p = float(np.mean(probs))
        if abs(p - self.threshold) < margin:
            verdict = "uncertain"
        elif p > self.threshold:
            verdict = "likely synthetic"
        else:
            verdict = "likely human"
        return Result(p, verdict, [float(x) for x in probs])

    def predict_file(self, path: str) -> Result:
        return self.predict_waveform(load_audio(path), 16_000)
