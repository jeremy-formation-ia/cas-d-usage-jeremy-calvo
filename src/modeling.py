"""Modèles candidats, validation croisée et métriques de classification."""

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from src.config import (
    DECISION_THRESHOLD,
    N_SPLITS,
    RANDOM_STATE,
)
from src.preprocessing import make_preprocessor


def make_models() -> dict:
    """Renvoie les modèles candidats, tous avec ``class_weight='balanced'``.

    Trois familles sont représentées : linéaire (référence), ensemble d'arbres
    (forêt aléatoire) et boosting.
    """
    return {
        "logreg": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "gradient_boosting": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=RANDOM_STATE,
        ),
    }


def make_pipeline(estimator, columns) -> Pipeline:
    """Assemble le prétraitement du scénario et le modèle dans un Pipeline.

    Le prétraitement est ainsi re-fitté à chaque fold de validation croisée,
    ce qui évite toute fuite de données.
    """
    return Pipeline([
        ("preprocessor", make_preprocessor(columns)),
        ("model", estimator),
    ])


def compute_metrics(y_true, y_proba, threshold=DECISION_THRESHOLD) -> dict:
    """Calcule les métriques de classification pour un seuil donné."""
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    return {
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_positive": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "precision_positive": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_positive": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def cross_validate_model(estimator, columns, X, y, n_splits=N_SPLITS, threshold=DECISION_THRESHOLD) -> dict:
    """Validation croisée stratifiée : moyenne et écart-type de chaque métrique."""
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics = []

    for train_idx, val_idx in splitter.split(X, y):
        pipeline = make_pipeline(clone(estimator), columns)
        pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = pipeline.predict_proba(X.iloc[val_idx])[:, 1]
        fold_metrics.append(compute_metrics(y.iloc[val_idx], proba, threshold=threshold))

    metric_names = ["f1_macro", "f1_positive", "precision_positive", "recall_positive", "roc_auc"]
    summary = {}
    for name in metric_names:
        values = [m[name] for m in fold_metrics]
        summary[f"{name}_mean"] = float(np.mean(values))
        summary[f"{name}_std"] = float(np.std(values))
    return summary


def benchmark(models, scenarios, X, y, threshold=DECISION_THRESHOLD) -> pd.DataFrame:
    """Tableau modèle × scénario en validation croisée."""
    rows = []
    for scenario_name, columns in scenarios.items():
        for model_name, estimator in models.items():
            scores = cross_validate_model(estimator, columns, X, y, threshold=threshold)
            rows.append({"scenario": scenario_name, "model": model_name, **scores})
    return pd.DataFrame(rows)


def measure_cost(estimator, columns, X, y, n_samples=1000) -> dict:
    """Mesure le temps d'entraînement et la latence d'inférence (sur n_samples)."""
    pipeline = make_pipeline(clone(estimator), columns)

    t0 = time.perf_counter()
    pipeline.fit(X, y)
    fit_time = time.perf_counter() - t0

    sample = X.iloc[:n_samples]
    t0 = time.perf_counter()
    pipeline.predict_proba(sample)
    inference_ms = (time.perf_counter() - t0) * 1000
    return {"fit_time_s": round(fit_time, 3), "inference_ms_per_1k": round(inference_ms, 3)}


def threshold_analysis(estimator, columns, X, y, thresholds=None, n_splits=N_SPLITS) -> pd.DataFrame:
    """Impact du seuil de décision, estimé en validation croisée sur le train.

    Les probabilités out-of-fold évitent de faire le choix du seuil sur le
    jeu de test. On rapporte précision, rappel, F1 et F2 pour chaque seuil.
    """
    from sklearn.metrics import fbeta_score

    if thresholds is None:
        thresholds = [0.2, 0.25, 0.3, 0.35, 0.4, 0.5]

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    pipeline = make_pipeline(clone(estimator), columns)
    proba = cross_val_predict(pipeline, X, y, cv=splitter, method="predict_proba")[:, 1]

    rows = []
    for th in thresholds:
        pred = (proba >= th).astype(int)
        rows.append({
            "threshold": th,
            "precision_positive": precision_score(y, pred, pos_label=1, zero_division=0),
            "recall_positive": recall_score(y, pred, pos_label=1),
            "f1_positive": f1_score(y, pred, pos_label=1, zero_division=0),
            "f1_macro": f1_score(y, pred, average="macro", zero_division=0),
            "f2_positive": fbeta_score(y, pred, beta=2, pos_label=1, zero_division=0),
            "n_contacted": int(pred.sum()),
            "n_missed": int(((y == 1) & (pred == 0)).sum()),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from src.data import load_prepared
    from src.preprocessing import build_scenarios

    X, y = load_prepared()
    scenarios = build_scenarios(X.columns)
    results = benchmark(make_models(), scenarios, X, y)
    print(results[["scenario", "model", "f1_macro_mean", "roc_auc_mean", "recall_positive_mean"]].round(3).to_string(index=False))
