"""Tests du module de modélisation."""

import numpy as np
import pandas as pd
import pytest

from src.config import N_SPLITS, RANDOM_STATE
from src.modeling import (
    benchmark,
    compute_metrics,
    cross_validate_model,
    make_models,
    make_pipeline,
    measure_cost,
    threshold_analysis,
)
from src.preprocessing import build_scenarios


@pytest.fixture(scope="module")
def small_data():
    rng = np.random.default_rng(RANDOM_STATE)
    n = 600
    X = pd.DataFrame({
        "age": rng.integers(18, 80, n),
        "duration": rng.integers(0, 1000, n),
        "job": rng.choice(["admin.", "blue-collar", "technician"], n),
        "education": rng.choice(["basic.9y", "university.degree", "unknown"], n),
        "campaign": rng.integers(1, 10, n),
        "previously_contacted": rng.integers(0, 2, n),
    })
    y = pd.Series(rng.integers(0, 2, n), name="y")
    return X, y


def test_make_models_has_three_families():
    models = make_models()
    assert set(models) == {"logreg", "random_forest", "gradient_boosting"}


def test_compute_metrics_keys():
    y_true = [0, 1, 1, 0, 1, 0]
    y_proba = [0.1, 0.8, 0.6, 0.4, 0.9, 0.2]
    metrics = compute_metrics(y_true, y_proba)
    assert set(metrics) >= {"f1_macro", "f1_positive", "precision_positive", "recall_positive", "roc_auc"}
    assert 0.0 <= metrics["f1_macro"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_threshold_changes_predictions():
    y_true = [0, 1, 1, 0, 1, 0]
    y_proba = [0.4, 0.6, 0.55, 0.45, 0.7, 0.3]
    low = compute_metrics(y_true, y_proba, threshold=0.3)
    high = compute_metrics(y_true, y_proba, threshold=0.7)
    assert low["recall_positive"] >= high["recall_positive"]


def test_pipeline_no_leakage_shape(small_data):
    X, y = small_data
    pipeline = make_pipeline(make_models()["logreg"], list(X.columns))
    pipeline.fit(X, y)
    proba = pipeline.predict_proba(X.iloc[:5])
    assert proba.shape == (5, 2)


def test_cross_validate_is_deterministic(small_data):
    X, y = small_data
    est = make_models()["logreg"]
    first = cross_validate_model(est, list(X.columns), X, y, n_splits=3)
    second = cross_validate_model(est, list(X.columns), X, y, n_splits=3)
    assert first == second


def test_benchmark_shape(small_data):
    X, y = small_data
    scenarios = {"S1": list(X.columns), "S2": [c for c in X.columns if c != "duration"]}
    models = {"logreg": make_models()["logreg"]}
    table = benchmark(models, scenarios, X, y)
    assert len(table) == 2
    assert {"scenario", "model", "f1_macro_mean", "roc_auc_mean"} <= set(table.columns)


def test_measure_cost_returns_timings(small_data):
    X, y = small_data
    cost = measure_cost(make_models()["logreg"], list(X.columns), X, y, n_samples=50)
    assert cost["fit_time_s"] >= 0
    assert cost["inference_ms_per_1k"] >= 0


def test_threshold_analysis_monotonic_recall(small_data):
    X, y = small_data
    table = threshold_analysis(
        make_models()["logreg"], list(X.columns), X, y,
        thresholds=[0.2, 0.5, 0.8], n_splits=3,
    )
    assert list(table["threshold"]) == [0.2, 0.5, 0.8]
    # Baisser le seuil augmente (ou maintient) le rappel
    assert table["recall_positive"].iloc[0] >= table["recall_positive"].iloc[-1]
    # Et augmente le nombre de contacts
    assert table["n_contacted"].iloc[0] >= table["n_contacted"].iloc[-1]

