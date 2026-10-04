"""Tests des routes de l'API."""

import pytest


def test_health_returns_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_info_exposes_metadata(client):
    r = client.get("/info")
    assert r.status_code == 200
    body = r.json()
    assert body["model_name"] == "bank_marketing"
    assert body["model_version"] == "v1.0.0"
    assert body["scenario"] == "S4"
    assert body["decision_threshold"] == 0.30


def test_predict_valid_payload(client, valid_payload):
    r = client.post("/predict", json=valid_payload)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["subscription_probability"] <= 1.0
    assert body["decision"] in {"contact", "do_not_contact", "abstain"}
    assert body["model_version"] == "v1.0.0"
    assert "request_id" in body


def test_predict_returns_request_id_header(client, valid_payload):
    r = client.post("/predict", json=valid_payload, headers={"X-Request-ID": "test-123"})
    assert r.headers.get("X-Request-ID") == "test-123"


@pytest.mark.parametrize(
    "field,value",
    [
        ("contact", "pigeon"),
        ("month", "janvier"),
        ("default", "peut-être"),
        ("emp.var.rate", 99.0),
    ],
)
def test_predict_invalid_payload_returns_422(client, valid_payload, field, value):
    payload = {**valid_payload, field: value}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_missing_field_returns_422(client, valid_payload):
    payload = {k: v for k, v in valid_payload.items() if k != "euribor3m"}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422
