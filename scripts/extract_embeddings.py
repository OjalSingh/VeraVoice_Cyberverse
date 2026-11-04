"""Run frozen WavLM over a manifest and save per-layer pooled features.

Usage: python scripts/extract_embeddings.py manifests/train.json feats/train.npz [--random-crop]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from veravoice.audio import load_audio, preprocess  # noqa: E402
from veravoice.embed import DEFAULT_BACKBONE, WavLMEmbedder  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("out")
    ap.add_argument("--backbone", default=DEFAULT_BACKBONE)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--random-crop", action="store_true", help="random 4 s window (use for train)")
    a = ap.parse_args()

    with open(a.manifest) as f:
        rows = json.load(f)
    emb = WavLMEmbedder(a.backbone)
    rng = np.random.default_rng(0)
    feats = np.zeros((len(rows), emb.num_layers, emb.hidden_size), dtype=np.float16)
    keep = np.ones(len(rows), dtype=bool)
    t0 = time.time()
    for start in range(0, len(rows), a.batch_size):
        batch, idx = [], []
        for i in range(start, min(start + a.batch_size, len(rows))):
            try:
                batch.append(preprocess(load_audio(rows[i]["path"]), random_crop=a.random_crop, rng=rng))
                idx.append(i)
            except Exception as e:  # corrupt or empty file
                print(f"[skip] {rows[i]['path']}: {e}")
                keep[i] = False
        if batch:
            feats[idx] = emb(batch).astype(np.float16)
        done = min(start + a.batch_size, len(rows))
        if (start // a.batch_size) % 20 == 0:
            rate = done / (time.time() - t0)
            print(f"{done}/{len(rows)} clips, {rate:.1f} clips/s")

    rows = [r for r, k in zip(rows, keep, strict=True) if k]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez_compressed(
        a.out,
        feats=feats[keep],
        labels=np.array([r["label"] for r in rows], dtype=np.int64),
        attacks=np.array([r["attack"] for r in rows]),
        sources=np.array([r["source"] for r in rows]),
        backbone=np.array(a.backbone),
    )
    print(f"saved {len(rows)} feature rows to {a.out}")


if __name__ == "__main__":
    main()
