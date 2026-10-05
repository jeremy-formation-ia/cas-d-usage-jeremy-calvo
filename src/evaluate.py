"""Évaluation continue du modèle sur un jeu de référence figé.

Recalcule les métriques du modèle retenu sur un sous-échantillon figé et les
compare à un « golden run » gelé. Le script renvoie un code de sortie non nul
si une métrique passe sous un seuil, ce qui bloque la CI (et donc la release).

Deux baselines à ne pas confondre :
- la baseline communiquée (métriques du test complet, dans model.json) ;
- le golden run (mesure sur le jeu de référence au moment du gel), seul
  utilisé par le garde-fou.
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    FINAL_SCENARIO,
    FINAL_THRESHOLD,
    RANDOM_STATE,
    REPO_ROOT,
)
from src.data import load_prepared
from src.modeling import compute_metrics, make_models, make_pipeline
from src.preprocessing import build_scenarios

REFERENCE_SET = REPO_ROOT / "data" / "reference_set.csv"
REFERENCE_BASELINE = REPO_ROOT / "data" / "reference_baseline.json"

# Seuils : plancher absolu + baisse maximale tolérée vs golden run
THRESHOLDS = {
    "f1_macro": {"absolute_min": 0.40, "max_drop": 0.05},
    "recall_positive": {"absolute_min": 0.70, "max_drop": 0.05},
    "roc_auc": {"absolute_min": 0.70, "max_drop": 0.05},
}

METRICS = ["f1_macro", "recall_positive", "roc_auc"]


def build_reference_set(n_rows: int = 1000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Extrait un sous-échantillon stratifié du test et le fige sur disque."""
    X, y = load_prepared()
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    sample = X_test.copy()
    sample["y"] = y_test.values
    parts = [
        group.sample(min(len(group), n_rows // 2), random_state=seed)
        for _, group in sample.groupby("y")
    ]
    sample = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    sample.to_csv(REFERENCE_SET, index=False)
    return sample


def load_reference_set() -> pd.DataFrame:
    if not REFERENCE_SET.exists():
        raise FileNotFoundError(
            f"Jeu de référence introuvable ({REFERENCE_SET}). Lancer `--freeze-reference`."
        )
    return pd.read_csv(REFERENCE_SET)


def evaluate(degrade: bool = False) -> dict:
    """Recalcule les métriques du modèle retenu sur le jeu de référence."""
    reference = load_reference_set()
    columns = build_scenarios([c for c in reference.columns if c != "y"])[FINAL_SCENARIO]

    X_ref = reference[columns]
    y_ref = reference["y"]

    X, y = load_prepared()
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    pipeline = make_pipeline(make_models()["gradient_boosting"], columns)
    pipeline.fit(X_train, y_train)

    proba = pipeline.predict_proba(X_ref)[:, 1]
    if degrade:
        proba = 1.0 - proba

    metrics = compute_metrics(y_ref, proba, threshold=FINAL_THRESHOLD)
    return {k: metrics[k] for k in METRICS}


def check_thresholds(metrics: dict, baseline: dict) -> list:
    """Liste des violations (plancher absolu ou baisse trop forte)."""
    violations = []
    for name, rule in THRESHOLDS.items():
        value = metrics[name]
        if value < rule["absolute_min"]:
            violations.append(f"{name}={value:.3f} < plancher {rule['absolute_min']}")
        if baseline.get(name) is not None and baseline[name] - value > rule["max_drop"]:
            violations.append(
                f"{name} a chuté de {baseline[name] - value:.3f} (> {rule['max_drop']})"
            )
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description="Évaluation continue du modèle")
    parser.add_argument("--freeze-reference", action="store_true",
                        help="construit le jeu de référence")
    parser.add_argument("--freeze-baseline", action="store_true",
                        help="gèle le golden run à partir du jeu de référence")
    parser.add_argument("--release-tag", default="dev")
    parser.add_argument("--degrade", action="store_true",
                        help="simule une dégradation (test du garde-fou)")
    args = parser.parse_args()

    if args.freeze_reference:
        sample = build_reference_set()
        print(f"Jeu de référence figé : {REFERENCE_SET} ({len(sample)} lignes)")
        return 0

    if args.freeze_baseline:
        metrics = evaluate()
        REFERENCE_BASELINE.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Golden run gelé : {REFERENCE_BASELINE}")
        print(json.dumps(metrics, indent=2))
        return 0

    if not REFERENCE_BASELINE.exists():
        raise FileNotFoundError(
            f"Golden run introuvable ({REFERENCE_BASELINE}). Lancer `--freeze-baseline`."
        )
    baseline = json.loads(REFERENCE_BASELINE.read_text(encoding="utf-8"))

    metrics = evaluate(degrade=args.degrade)
    violations = check_thresholds(metrics, baseline)

    print(f"Release : {args.release_tag}")
    print("Métriques :", json.dumps({k: round(v, 4) for k, v in metrics.items()}))
    if violations:
        print("VIOLATIONS :")
        for v in violations:
            print(f"  - {v}")
        return 1
    print("Toutes les métriques sont au-dessus des seuils.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
