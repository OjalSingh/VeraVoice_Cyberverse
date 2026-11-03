"""Audio loading and preprocessing shared by training and inference.

Every clip goes through the same steps, so the model can't learn shortcuts from
file format, sample rate, loudness or leading silence:
mono -> 16 kHz -> trim silence -> peak-normalize -> crop/pad to a fixed length.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16_000
MAX_SECONDS = 4.0


def load_audio(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load any file librosa can read as mono float32 at ``sr``."""
    import librosa

    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32)


def to_mono_16k(y: np.ndarray, sr: int) -> np.ndarray:
    """Convert an in-memory waveform (channels-last or 1-D) to mono 16 kHz."""
    import librosa

    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 2:
        # Accept (samples, channels) or (channels, samples).
        y = y.mean(axis=1) if y.shape[0] > y.shape[1] else y.mean(axis=0)
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
    return y.astype(np.float32)


def preprocess(
    y: np.ndarray,
    max_seconds: float = MAX_SECONDS,
    random_crop: bool = False,
    rng: np.random.Generator | None = None,
    top_db: float = 40.0,
) -> np.ndarray:
    """Trim silence, peak-normalize, and crop or repeat-pad to ``max_seconds``."""
    import librosa

    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        raise ValueError("empty audio")
    trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    if trimmed.size >= SAMPLE_RATE // 4:  # keep the original if trimming ate almost everything
        y = trimmed
    peak = float(np.max(np.abs(y)))
    if peak > 1e-6:
        y = y / peak * 0.95

    n = int(max_seconds * SAMPLE_RATE)
    if y.size > n:
        if random_crop:
            rng = rng or np.random.default_rng()
            start = int(rng.integers(0, y.size - n + 1))
        else:
            start = (y.size - n) // 2
        y = y[start : start + n]
    elif y.size < n:
        reps = int(np.ceil(n / y.size))
        y = np.tile(y, reps)[:n]
    return y.astype(np.float32)


def chunks(y: np.ndarray, seconds: float = MAX_SECONDS, hop_seconds: float = 2.0) -> list[np.ndarray]:
    """Split a long waveform into overlapping windows for inference."""
    n = int(seconds * SAMPLE_RATE)
    hop = int(hop_seconds * SAMPLE_RATE)
    if y.size <= n:
        return [y]
    starts = list(range(0, y.size - n + 1, hop))
    if starts[-1] + n < y.size:
        starts.append(y.size - n)
    return [y[s : s + n] for s in starts]
