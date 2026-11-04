import numpy as np

from veravoice.audio import MAX_SECONDS, SAMPLE_RATE, chunks, preprocess, to_mono_16k

N = int(MAX_SECONDS * SAMPLE_RATE)


def tone(seconds, sr=SAMPLE_RATE, amp=0.3):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_fixed_length_and_peak():
    for secs in (0.5, 4.0, 9.0):
        y = preprocess(tone(secs))
        assert y.shape == (N,)
        assert np.max(np.abs(y)) <= 0.95 + 1e-6


def test_silence_is_trimmed():
    y = np.concatenate([np.zeros(SAMPLE_RATE * 2, np.float32), tone(1.0)])
    out = preprocess(y)
    # After trimming, the tone is repeated to fill 4 s, so there is no 2 s of silence at the start.
    assert np.abs(out[:SAMPLE_RATE]).max() > 0.5


def test_loudness_does_not_matter():
    a = preprocess(tone(2.0, amp=0.05))
    b = preprocess(tone(2.0, amp=0.9))
    np.testing.assert_allclose(a, b, atol=1e-4)


def test_resample_and_stereo():
    stereo = np.stack([tone(1.0, sr=44100), tone(1.0, sr=44100)], axis=1)
    y = to_mono_16k(stereo, 44100)
    assert y.ndim == 1 and abs(len(y) - SAMPLE_RATE) <= 1


def test_chunks_cover_whole_clip():
    y = np.arange(int(9.3 * SAMPLE_RATE), dtype=np.float32)
    cs = chunks(y)
    assert all(len(c) == N for c in cs)
    assert cs[0][0] == 0 and cs[-1][-1] == y[-1]
    assert len(chunks(tone(2.0))) == 1
