"""Frozen WavLM feature extractor.

We keep WavLM frozen and store a mean-pooled vector for every hidden layer.
The classifier head then learns which layers matter. Lower and middle layers
carry most of the acoustic artefacts that vocoders leave behind, so this beats
using only the last layer, and it lets us extract features once and train the
head in seconds.
"""

from __future__ import annotations

import numpy as np
import torch

DEFAULT_BACKBONE = "microsoft/wavlm-base-plus"


class WavLMEmbedder:
    def __init__(self, backbone: str | None = DEFAULT_BACKBONE, device: str | None = None, model=None):
        from transformers import WavLMModel

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model if model is not None else WavLMModel.from_pretrained(backbone)
        self.model.eval().to(self.device)
        self.backbone = backbone

    @property
    def num_layers(self) -> int:
        return self.model.config.num_hidden_layers + 1  # + CNN/embedding output

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    @torch.inference_mode()
    def __call__(self, batch: list[np.ndarray]) -> np.ndarray:
        """Return per-layer mean-pooled features, shape (batch, layers, hidden).

        Clips should all have the same length (``audio.preprocess`` crops or pads to
        4 s). wavlm-base-plus uses group norm in its conv front end, which mixes
        padding into the statistics, so ragged batches are scored one clip at a time.
        """
        lengths = [len(x) for x in batch]
        if len(set(lengths)) > 1:
            return np.concatenate([self([x]) for x in batch])
        n = lengths[0]
        wav = np.zeros((len(batch), n), dtype=np.float32)
        for i, x in enumerate(batch):
            # WavLM expects zero-mean, unit-variance input.
            x = (x - x.mean()) / (x.std() + 1e-7)
            wav[i] = x
        out = self.model(torch.from_numpy(wav).to(self.device), output_hidden_states=True)
        hidden = torch.stack(out.hidden_states, dim=1)  # (B, layers, frames, hidden)
        return hidden.mean(2).float().cpu().numpy()
