from __future__ import annotations

from app import app


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
