"""Unit tests for inference components."""

import numpy as np
import pytest

from src.inference.post_processor import PostProcessor
from src.utils import normalize_landmarks, smooth_predictions, compute_hand_angles


def test_normalize_landmarks_shape():
    lm = np.random.rand(21, 3).astype(np.float32)
    normed = normalize_landmarks(lm)
    assert normed.shape == (63,)


def test_normalize_landmarks_wrist_at_origin():
    lm = np.random.rand(21, 3).astype(np.float32)
    normed = normalize_landmarks(lm).reshape(21, 3)
    assert normed.shape == (21, 3)


def test_smooth_predictions_empty():
    assert smooth_predictions([]) == -1


def test_smooth_predictions_single():
    assert smooth_predictions([3]) == 3


def test_smooth_predictions_majority():
    result = smooth_predictions([1, 1, 1, 2, 3])
    assert result == 1


def test_compute_hand_angles():
    lm = np.random.rand(21, 3).astype(np.float32)
    angles = compute_hand_angles(lm)
    assert angles.shape == (15,)
    assert np.all(angles >= 0)
    assert np.all(angles <= np.pi)


def test_post_processor_update():
    pp = PostProcessor()
    # Fill buffer with consistent predictions above threshold
    for _ in range(pp.buffer.maxlen):
        pp.update("A", 0.9)
    assert pp.get_stable_prediction() == "A"


def test_post_processor_reset():
    pp = PostProcessor()
    pp.update("A", 0.9)
    pp.reset()
    assert pp.get_stable_prediction() is None
    assert pp.confirmed_gesture is None
