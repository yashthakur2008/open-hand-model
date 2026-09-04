"""Public hand-pose model API built on OpenCV DNN."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np

from .hand import MPHandPose
from .palm import MPPalmDet


_DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models"


@dataclass(frozen=True)
class HandDetection:
    """One detected hand in image and world coordinates."""

    bbox: np.ndarray
    landmarks: np.ndarray
    world_landmarks: np.ndarray
    confidence: float
    handedness: str

    def __post_init__(self) -> None:
        bbox = np.asarray(self.bbox, dtype=np.float32)
        landmarks = np.asarray(self.landmarks, dtype=np.float32)
        world = np.asarray(self.world_landmarks, dtype=np.float32)
        if bbox.shape != (4,):
            raise ValueError(f"bbox must have shape (4,), got {bbox.shape}")
        if landmarks.shape != (21, 3):
            raise ValueError(f"landmarks must have shape (21, 3), got {landmarks.shape}")
        if world.shape != (21, 3):
            raise ValueError(f"world_landmarks must have shape (21, 3), got {world.shape}")
        confidence = float(self.confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {confidence}")
        if self.handedness not in {"Left", "Right", "Unknown"}:
            raise ValueError(f"unsupported handedness: {self.handedness!r}")
        object.__setattr__(self, "bbox", bbox)
        object.__setattr__(self, "landmarks", landmarks)
        object.__setattr__(self, "world_landmarks", world)
        object.__setattr__(self, "confidence", confidence)

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-friendly output without exposing mutable arrays."""
        return {
            "bbox": self.bbox.tolist(),
            "landmarks": self.landmarks.tolist(),
            "world_landmarks": self.world_landmarks.tolist(),
            "confidence": self.confidence,
            "handedness": self.handedness,
        }


class HandPoseModel:
    """Detect palms and estimate 21 hand landmarks with OpenCV DNN."""

    def __init__(
        self,
        model_dir: str | Path = _DEFAULT_MODEL_DIR,
        confidence: float = 0.8,
        backend: int = cv.dnn.DNN_BACKEND_OPENCV,
        target: int = cv.dnn.DNN_TARGET_CPU,
    ) -> None:
        if not 0.0 <= float(confidence) <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        model_dir = Path(model_dir)
        palm_path = model_dir / "palm_detection_mediapipe_2023feb.onnx"
        hand_path = model_dir / "handpose_estimation_mediapipe_2023feb.onnx"
        missing = [str(path) for path in (palm_path, hand_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing model asset(s): " + ", ".join(missing))
        self.palm_detector = MPPalmDet(
            modelPath=str(palm_path),
            nmsThreshold=0.3,
            scoreThreshold=0.6,
            backendId=backend,
            targetId=target,
        )
        self.hand_detector = MPHandPose(
            modelPath=str(hand_path),
            confThreshold=float(confidence),
            backendId=backend,
            targetId=target,
        )

    def detect(self, image: np.ndarray) -> list[HandDetection]:
        """Return all detected hands for a BGR uint8 image."""
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must have shape (height, width, 3)")
        if image.dtype != np.uint8:
            raise TypeError(f"image must have dtype uint8, got {image.dtype}")
        if image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError("image must not be empty")

        detections: list[HandDetection] = []
        palms = self.palm_detector.infer(image)
        for palm in palms:
            result = self.hand_detector.infer(image, palm)
            if result is None:
                continue
            for row in np.asarray(result).reshape(-1, 132):
                handedness = "Left" if float(row[-2]) <= 0.5 else "Right"
                detections.append(
                    HandDetection(
                        bbox=row[:4],
                        landmarks=row[4:67].reshape(21, 3),
                        world_landmarks=row[67:130].reshape(21, 3),
                        confidence=float(row[-1]),
                        handedness=handedness,
                    )
                )
        return detections
