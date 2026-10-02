"""Tests des modules de données et de prétraitement."""

import numpy as np
import pandas as pd
import pytest

from src.config import (
    CAMPAIGN_HISTORY_FEATURES,
    NUMERIC_FEATURES,
    SENSITIVE_FEATURES,
    TARGET_COLUMN,
)
from src.data import (
    build_features,
    clean_dataset,
    get_X_y,
    load_dataset,
    load_prepared,
)
from src.preprocessing import build_scenarios, make_preprocessor


@pytest.fixture(scope="module")
def raw_df() -> pd.DataFrame:
    return load_dataset()


def test_load_dataset_shape(raw_df):
    assert raw_df.shape == (41188, 21)
    assert TARGET_COLUMN in raw_df.columns


def test_clean_dataset_removes_duplicates(raw_df):
    cleaned = clean_dataset(raw_df)
    assert cleaned.duplicated().sum() == 0
    assert len(cleaned) == len(raw_df) - 12


def test_build_features_replaces_pdays(raw_df):
    features = build_features(raw_df)
    assert "pdays" not in features.columns
    assert "previously_contacted" in features.columns
    assert set(features["previously_contacted"].unique()) <= {0, 1}


def test_get_X_y_encoding(raw_df):
    df = build_features(clean_dataset(raw_df))
    X, y = get_X_y(df)
    assert TARGET_COLUMN not in X.columns
    assert set(y.unique()) <= {0, 1}
    assert len(X) == len(y)


def test_load_prepared_is_consistent():
    X, y = load_prepared()
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == 20  # 20 explicatives après remplacement de pdays


def test_scenarios_are_nested():
    X, _ = load_prepared()
    scenarios = build_scenarios(X.columns)

    assert len(scenarios["S1"]) == 20
    assert "duration" not in scenarios["S2"]
    assert all(f not in scenarios["S3"] for f in SENSITIVE_FEATURES)
    assert all(f not in scenarios["S4"] for f in CAMPAIGN_HISTORY_FEATURES)
    # Emboîtement : S4 ⊂ S3 ⊂ S2 ⊂ S1
    assert set(scenarios["S4"]) <= set(scenarios["S3"]) <= set(scenarios["S2"]) <= set(scenarios["S1"])


def test_preprocessor_output_has_no_nan():
    X, y = load_prepared()
    scenarios = build_scenarios(X.columns)
    preprocessor = make_preprocessor(scenarios["S1"])

    X_prep = preprocessor.fit_transform(X)
    assert not np.isnan(X_prep).any()


def test_preprocessor_is_deterministic():
    X, _ = load_prepared()
    scenarios = build_scenarios(X.columns)
    preprocessor = make_preprocessor(scenarios["S1"])

    first = preprocessor.fit_transform(X)
    second = preprocessor.transform(X)
    assert np.allclose(first, second)


def test_preprocessor_handles_unknown_category():
    """Une modalité inconnue ne doit pas casser la transformation."""
    X, _ = load_prepared()
    scenarios = build_scenarios(X.columns)
    preprocessor = make_preprocessor(scenarios["S1"])
    preprocessor.fit(X)

    sample = X.iloc[[0]].copy()
    sample["job"] = "metier_inconnu"
    out = preprocessor.transform(sample)
    assert out.shape[0] == 1
    assert not np.isnan(out).any()
