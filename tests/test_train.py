"""Tests de persistance et contract test du modèle."""

import json

import numpy as np
import pytest

from src.config import FINAL_SCENARIO, FINAL_THRESHOLD, METADATA_PATH, MODEL_PATH
from src.data import load_prepared
from src.preprocessing import build_scenarios


@pytest.fixture(scope="module")
def artifacts():
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        pytest.skip("Artefacts absents : lancer `python -m src.train`")
    import joblib

    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return model, metadata


def test_metadata_contract_keys(artifacts):
    _, metadata = artifacts
    required = {
        "model_name", "model_version", "created_at", "sklearn_version",
        "dataset_sha256", "feature_columns", "metrics_test",
    }
    assert required <= set(metadata)


def test_metadata_scenario_and_threshold(artifacts):
    _, metadata = artifacts
    assert metadata["scenario"] == FINAL_SCENARIO
    assert metadata["decision_threshold"] == FINAL_THRESHOLD


def test_model_contract_predict(artifacts):
    """Contract test : formes et bornes des sorties du modèle rechargé."""
    model, metadata = artifacts
    X, _ = load_prepared()
    columns = build_scenarios(X.columns)[FINAL_SCENARIO]
    assert list(columns) == metadata["feature_columns"]

    sample = X[columns].head(3)
    prediction = model.predict(sample)
    proba = model.predict_proba(sample)

    assert prediction.shape == (3,)
    assert proba.shape == (3, 2)
    assert (proba >= 0).all() and (proba <= 1).all()
    assert set(prediction.tolist()) <= {0, 1}


def test_model_stability(artifacts):
    """Deux prédictions successives donnent le même résultat."""
    model, metadata = artifacts
    X, _ = load_prepared()
    columns = metadata["feature_columns"]
    sample = X[columns].head(5)
    first = model.predict_proba(sample)
    second = model.predict_proba(sample)
    assert np.allclose(first, second)
