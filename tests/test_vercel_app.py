from __future__ import annotations

from app import MODEL_ASSET_NAMES, app


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


def test_ready_endpoint_reports_bundled_assets():
    response = app.test_client().get("/ready")
    assert response.status_code == 200
    assert response.json["status"] == "ready"
    assert response.json["assets"] == list(MODEL_ASSET_NAMES)


def test_ready_endpoint_reports_missing_assets(monkeypatch, tmp_path):
    monkeypatch.setattr("app.BUNDLED_ASSET_DIR", tmp_path)
    response = app.test_client().get("/ready")
    assert response.status_code == 503
    assert response.json["status"] == "not_ready"
    assert set(response.json["missing"]) == set(MODEL_ASSET_NAMES)
