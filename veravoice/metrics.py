"""Evaluation metrics. EER is the standard for spoof detection (ASVspoof)."""

from __future__ import annotations

import numpy as np


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Equal error rate.

    ``scores``: higher means more likely spoof. ``labels``: 1 = spoof, 0 = bona fide.
    Returns (eer, threshold) where the false accept rate equals the false reject rate.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    if labels.min() == labels.max():
        raise ValueError("EER needs both classes")
    order = np.argsort(scores, kind="mergesort")
    s, y = scores[order], labels[order]
    n_spoof, n_bona = y.sum(), (1 - y).sum()
    # Threshold just above each score: everything <= it is called bona fide.
    # miss = spoof called bona fide; fa = bona fide called spoof.
    miss = np.concatenate([[0], np.cumsum(y)]) / n_spoof
    fa = 1 - np.concatenate([[0], np.cumsum(1 - y)]) / n_bona
    thresholds = np.concatenate([[s[0] - 1e-9], s])
    idx = int(np.argmin(np.abs(miss - fa)))
    eer = float((miss[idx] + fa[idx]) / 2)
    return eer, float(thresholds[idx])
