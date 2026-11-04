import numpy as np
import pytest

from veravoice.metrics import compute_eer


def test_perfect_separation_is_zero():
    eer, thr = compute_eer(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1]))
    assert eer == 0.0
    assert 0.2 <= thr < 0.8


def test_inverted_scores_is_one():
    eer, _ = compute_eer(np.array([0.9, 0.8, 0.2, 0.1]), np.array([0, 0, 1, 1]))
    assert eer == 1.0


def test_random_scores_near_half():
    rng = np.random.default_rng(0)
    eer, _ = compute_eer(rng.random(20000), rng.integers(0, 2, 20000))
    assert abs(eer - 0.5) < 0.02


def test_known_overlap():
    # One bona fide scores above one spoof out of 4 each: EER = 25%.
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.6, 0.8, 0.9, 0.95])
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    eer, _ = compute_eer(scores, labels)
    assert eer == pytest.approx(0.25)


def test_needs_both_classes():
    with pytest.raises(ValueError):
        compute_eer(np.array([0.1, 0.2]), np.array([1, 1]))
