"""Tests de l'évaluation continue et du garde-fou."""

import json

import pytest

from src.config import REPO_ROOT
from src.evaluate import (
    REFERENCE_BASELINE,
    REFERENCE_SET,
    check_thresholds,
    evaluate,
)


@pytest.mark.skipif(not REFERENCE_SET.exists(), reason="Jeu de référence absent")
def test_evaluate_returns_metrics():
    metrics = evaluate()
    assert set(metrics) == {"f1_macro", "recall_positive", "roc_auc"}
    assert 0.0 <= metrics["f1_macro"] <= 1.0


@pytest.mark.skipif(not REFERENCE_BASELINE.exists(), reason="Golden run absent")
def test_no_violation_when_stable():
    baseline = json.loads(REFERENCE_BASELINE.read_text(encoding="utf-8"))
    assert check_thresholds(baseline, baseline) == []


def test_degraded_produces_violations():
    baseline = {"f1_macro": 0.70, "recall_positive": 0.85, "roc_auc": 0.80}
    degraded = {"f1_macro": 0.30, "recall_positive": 0.20, "roc_auc": 0.50}
    violations = check_thresholds(degraded, baseline)
    assert len(violations) >= 3
