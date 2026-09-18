from __future__ import annotations

import numpy as np
import pytest

from open_hand_model import HandDetection, HandPoseModel


def make_detection_row(confidence: float = 0.91, handedness: float = 0.2) -> np.ndarray:
    row = np.zeros(132, dtype=np.float32)
    row[:4] = [10, 20, 100, 120]
    row[4:67] = np.arange(63, dtype=np.float32)
    row[67:130] = np.arange(63, dtype=np.float32) / 10
    row[-2] = handedness
    row[-1] = confidence
    return row


def fake_model() -> HandPoseModel:
    model = object.__new__(HandPoseModel)

    class Palm:
        def infer(self, image):
            return [object()]

    class Hand:
        def infer(self, image, palm):
            return np.array([make_detection_row()])

    model.palm_detector = Palm()
    model.hand_detector = Hand()
    return model


def test_detect_returns_structured_landmarks():
    result = fake_model().detect(np.zeros((128, 128, 3), dtype=np.uint8))
    assert len(result) == 1
    assert isinstance(result[0], HandDetection)
    assert result[0].bbox.shape == (4,)
    assert result[0].landmarks.shape == (21, 3)
    assert result[0].world_landmarks.shape == (21, 3)
    assert result[0].handedness == "Left"
    assert result[0].confidence == pytest.approx(0.91)


def test_detect_maps_right_handedness():
    model = fake_model()
    model.hand_detector.infer = lambda image, palm: np.array([make_detection_row(handedness=0.8)])
    assert model.detect(np.zeros((10, 10, 3), dtype=np.uint8))[0].handedness == "Right"


def test_detect_rejects_bad_image_shape():
    with pytest.raises(ValueError, match="shape"):
        fake_model().detect(np.zeros((10, 10), dtype=np.uint8))


def test_detect_rejects_bad_image_dtype():
    with pytest.raises(TypeError, match="dtype"):
        fake_model().detect(np.zeros((10, 10, 3), dtype=np.float32))


def test_detect_rejects_empty_image():
    with pytest.raises(ValueError, match="empty"):
        fake_model().detect(np.zeros((0, 10, 3), dtype=np.uint8))


def test_detection_serializes_without_numpy_types():
    value = HandDetection(
        bbox=[1, 2, 3, 4],
        landmarks=np.zeros((21, 3)),
        world_landmarks=np.zeros((21, 3)),
        confidence=0.5,
        handedness="Unknown",
    ).as_dict()
    assert value["bbox"] == [1.0, 2.0, 3.0, 4.0]
    assert value["landmarks"][0] == [0.0, 0.0, 0.0]


def test_detection_rejects_invalid_shapes_and_values():
    zeros = np.zeros((21, 3))
    with pytest.raises(ValueError, match="bbox"):
        HandDetection([1, 2], zeros, zeros, 0.5, "Left")
    with pytest.raises(ValueError, match="confidence"):
        HandDetection([1, 2, 3, 4], zeros, zeros, 1.1, "Left")
    with pytest.raises(ValueError, match="handedness"):
        HandDetection([1, 2, 3, 4], zeros, zeros, 0.5, "Maybe")
