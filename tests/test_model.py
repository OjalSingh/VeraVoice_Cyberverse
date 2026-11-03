import json

import numpy as np
import soundfile as sf
import torch

from veravoice.audio import SAMPLE_RATE
from veravoice.detector import Detector, save_bundle
from veravoice.head import LayerWeightedHead


def test_embedder_batching_is_consistent(tiny_embedder):
    rng = np.random.default_rng(0)
    a, b = (rng.standard_normal(SAMPLE_RATE).astype(np.float32) for _ in range(2))
    batched = tiny_embedder([a, b])
    assert batched.shape == (2, tiny_embedder.num_layers, tiny_embedder.hidden_size)
    np.testing.assert_allclose(batched[0], tiny_embedder([a])[0], atol=1e-4)
    # Ragged batches fall back to one clip at a time instead of zero-padding.
    ragged = tiny_embedder([a, rng.standard_normal(SAMPLE_RATE * 2).astype(np.float32)])
    np.testing.assert_allclose(ragged[0], batched[0], atol=1e-4)


def test_head_learns_a_separable_problem():
    from scripts.train_head import train

    torch.manual_seed(0)
    x = torch.randn(3000, 3, 16)
    y = (x[:, 1, 0] > 0).float()  # the signal lives in layer 1 only
    head, eer, thr = train(x[:2500], y[:2500], x[2500:], y[2500:], epochs=40, batch=64, log=lambda *_: None)
    assert eer < 0.1
    assert int(head.layer_weights().argmax()) == 1


def test_bundle_round_trip(tmp_path, tiny_embedder):
    head = LayerWeightedHead(tiny_embedder.num_layers, tiny_embedder.hidden_size)
    save_bundle(str(tmp_path), head, "tiny-test", threshold=0.4, metrics={"dev_eer": 0.1})
    assert json.loads((tmp_path / "config.json").read_text())["threshold"] == 0.4

    det = Detector(str(tmp_path), embedder=tiny_embedder)
    wav = tmp_path / "clip.wav"
    sf.write(wav, 0.3 * np.sin(np.linspace(0, 2000, 44100 * 7)).astype(np.float32), 44100)
    r = det.predict_file(str(wav))
    assert 0.0 <= r.spoof_probability <= 1.0
    assert len(r.chunk_probabilities) >= 2  # 7 s clip is scored in overlapping windows
    assert r.verdict in {"likely human", "likely synthetic", "uncertain"}
