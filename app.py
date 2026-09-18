"""Small Flask API entrypoint for Vercel and local demos."""

from __future__ import annotations

import base64
import binascii
from functools import lru_cache
from pathlib import Path
from typing import Any
import sys

# Vercel loads this root entrypoint directly. Make the src-layout package
# importable even when the project itself has not been installed as a wheel.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from flask import Flask, jsonify, request

app = Flask(__name__)


@lru_cache(maxsize=1)
def _model() -> Any:
    """Load the bundled model once per serverless instance."""
    from open_hand_model import HandPoseModel

    return HandPoseModel()


def _decode_image() -> np.ndarray:
    """Decode an uploaded image or a base64 JSON payload into BGR pixels."""
    import cv2 as cv
    import numpy as np

    payload = request.files.get("image")
    if payload is not None:
        encoded = payload.read()
    else:
        body = request.get_json(silent=True) or {}
        value = body.get("image")
        if not isinstance(value, str):
            raise ValueError("send an image file or a JSON image base64 string")
        if "," in value and value.startswith("data:"):
            value = value.split(",", 1)[1]
        try:
            encoded = base64.b64decode(value, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("image must be valid base64") from exc

    image = cv.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv.IMREAD_COLOR)
    if image is None:
        raise ValueError("image could not be decoded")
    return image


@app.get("/")
def index() -> Any:
    return jsonify(
        {
            "name": "open-hand-model",
            "status": "ok",
            "endpoints": {"health": "/health", "metadata": "/metadata", "predict": "/predict"},
        }
    )


@app.get("/health")
def health() -> Any:
    return jsonify({"status": "ok"})


@app.get("/metadata")
def metadata() -> Any:
    return jsonify(
        {
            "name": "open-hand-model",
            "version": "0.1.0",
            "landmarks": 21,
            "license": "Apache-2.0",
        }
    )


@app.post("/predict")
def predict() -> Any:
    try:
        detections = _model().detect(_decode_image())
    except (TypeError, ValueError, FileNotFoundError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"detections": [detection.as_dict() for detection in detections]})


if __name__ == "__main__":
    app.run(debug=True)
