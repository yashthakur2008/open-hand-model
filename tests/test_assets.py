from __future__ import annotations

import hashlib
from pathlib import Path

from open_hand_model import HandPoseModel


ROOT = Path(__file__).resolve().parents[1]
ASSETS = {
    "models/palm_detection_mediapipe_2023feb.onnx": "78ff51c38496b7fc8b8ebdb6cc8c1abb02fa6c38427c6848254cdaba57fcce7c",
    "models/handpose_estimation_mediapipe_2023feb.onnx": "db0898ae717b76b075d9bf563af315b29562e11f8df5027a1ef07b02bef6d81c",
    "src/open_hand_model/assets/palm_detection_mediapipe_2023feb.onnx": "78ff51c38496b7fc8b8ebdb6cc8c1abb02fa6c38427c6848254cdaba57fcce7c",
    "src/open_hand_model/assets/handpose_estimation_mediapipe_2023feb.onnx": "db0898ae717b76b075d9bf563af315b29562e11f8df5027a1ef07b02bef6d81c",
}


def test_model_assets_exist_and_are_nonempty():
    for relative in ASSETS:
        path = ROOT / relative
        assert path.is_file(), relative
        assert path.stat().st_size > 100_000, relative


def test_model_asset_hashes_are_pinned():
    for relative, expected in ASSETS.items():
        digest = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert digest == expected, relative


def test_default_model_loads_bundled_package_assets():
    model = HandPoseModel()
    assert model.palm_detector.model_path.endswith("palm_detection_mediapipe_2023feb.onnx")
    assert model.hand_detector.model_path.endswith("handpose_estimation_mediapipe_2023feb.onnx")
