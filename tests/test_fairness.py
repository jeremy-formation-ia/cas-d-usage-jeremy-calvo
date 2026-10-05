"""Tests du module d'équité (performance par sous-groupe)."""

import numpy as np
import pandas as pd
import pytest

from src.fairness import add_age_group, group_performance, subgroup_performance


def test_add_age_group_labels():
    df = pd.DataFrame({"age": [20, 30, 40, 50, 60, 70]})
    out = add_age_group(df)
    assert out["age_group"].tolist() == ["17-25", "26-35", "36-45", "46-55", "56-65", "66+"]


def test_group_performance_values():
    y_true = [1, 1, 0, 0]
    y_pred = [1, 0, 1, 0]
    groups = ["A", "A", "B", "B"]
    table = group_performance(y_true, y_pred, groups)
    row_a = table[table["group"] == "A"].iloc[0]
    assert row_a["n"] == 2
    assert row_a["recall"] == 0.5  # 1 vrai positif sur 2
    assert row_a["selection_rate"] == 0.5


def test_subgroup_performance_runs():
    from sklearn.dummy import DummyClassifier

    X = pd.DataFrame({"age": [20, 30, 40, 50], "job": ["a", "a", "b", "b"]})
    y = pd.Series([0, 1, 0, 1])
    model = DummyClassifier(strategy="prior").fit(X, y)
    table = subgroup_performance(model, X, y, sensitive="job")
    assert set(table.columns) >= {"group", "n", "selection_rate", "recall", "fpr", "fnr"}
