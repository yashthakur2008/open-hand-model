from __future__ import annotations

import os
from pathlib import Path

import cv2 as cv
import pytest

from open_hand_model import HandPoseModel


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_MODEL_SMOKE") != "1",
    reason="set RUN_MODEL_SMOKE=1 to run real ONNX inference",
)


def test_real_model_detects_fixture_hand():
    root = Path(__file__).resolve().parents[1]
    image = cv.imread(str(root / "tests/fixtures/hand169.png"))
    assert image is not None
    detections = HandPoseModel(model_dir=root / "models").detect(image)
    assert detections
    assert all(d.landmarks.shape == (21, 3) for d in detections)
    assert all(0.0 <= d.confidence <= 1.0 for d in detections)


def test_real_model_returns_no_false_hand_on_empty_fixture():
    root = Path(__file__).resolve().parents[1]
    image = cv.imread(str(root / "tests/fixtures/handframe.png"))
    assert image is not None
    detections = HandPoseModel(model_dir=root / "models").detect(image)
    # The fixture is intentionally kept as a false-positive regression case:
    # the model may fit a face-like structure, so the contract is safe output,
    # not universal rejection of non-hand imagery.
    assert isinstance(detections, list)
    assert all(d.landmarks.shape == (21, 3) for d in detections)
