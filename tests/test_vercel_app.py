from __future__ import annotations

import io

from app import MAX_UPLOAD_BYTES, app


def test_health_endpoint() -> None:
    response = app.test_client().get("/health")
    assert response.status_code == 200
    assert response.json == {"status": "ok"}


def test_metadata_endpoint() -> None:
    response = app.test_client().get("/metadata")
    assert response.status_code == 200
    assert response.json["landmarks"] == 21
    assert response.json["license"] == "Apache-2.0"


def test_predict_requires_an_image() -> None:
    response = app.test_client().post("/predict", json={})
    assert response.status_code == 400
    assert "image" in response.json["error"]


def test_predict_rejects_oversized_upload() -> None:
    response = app.test_client().post(
        "/predict",
        data={"image": (io.BytesIO(b"x" * (MAX_UPLOAD_BYTES + 1024)), "large.bin")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 413
    assert "limit" in response.json["error"]
