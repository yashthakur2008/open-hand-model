from __future__ import annotations

import numpy as np
import pytest

from open_hand_model import HandPoseModel


def test_model_rejects_invalid_confidence_before_loading_assets(tmp_path):
    with pytest.raises(ValueError, match="confidence"):
        HandPoseModel(model_dir=tmp_path, confidence=-0.1)


def test_model_reports_missing_assets(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing model asset"):
        HandPoseModel(model_dir=tmp_path)


def test_detect_preserves_empty_result():
    model = object.__new__(HandPoseModel)

    class Palm:
        def infer(self, image):
            return []

    model.palm_detector = Palm()
    model.hand_detector = object()
    assert model.detect(np.zeros((32, 32, 3), dtype=np.uint8)) == []
