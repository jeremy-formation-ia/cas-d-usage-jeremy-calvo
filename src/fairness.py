"""Mesure de la performance du modèle par sous-groupe (équité).

Le sujet demande de vérifier si le taux de sélection ou les taux de faux
positifs / faux négatifs varient fortement d'un groupe à l'autre. Ce module
applique le modèle retenu à des sous-groupes (tranches d'âge, profession) et
rapporte, par groupe, la taille, le taux de sélection (part prédite « oui »),
le rappel et le taux de faux positifs / faux négatifs.
"""

import numpy as np
import pandas as pd

from src.config import FINAL_THRESHOLD

AGE_BINS = [16, 25, 35, 45, 55, 65, 100]
AGE_LABELS = ["17-25", "26-35", "36-45", "46-55", "56-65", "66+"]


def add_age_group(df: pd.DataFrame, age_col: str = "age") -> pd.DataFrame:
    """Ajoute une colonne de tranche d'âge."""
    df = df.copy()
    df["age_group"] = pd.cut(df[age_col], bins=AGE_BINS, labels=AGE_LABELS)
    return df


def group_performance(y_true, y_pred, groups) -> pd.DataFrame:
    """Performance par groupe : taille, taux de sélection, rappel, taux FP/FN.

    - taux de sélection : part de « oui » prédits dans le groupe ;
    - rappel : part des vrais « oui » correctement détectés ;
    - taux FP : part des vrais « non » prédits « oui » ;
    - taux FN : part des vrais « oui » prédits « non ».
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    frame = pd.DataFrame({"group": np.asarray(groups), "y_true": y_true, "y_pred": y_pred})

    rows = []
    for name, sub in frame.groupby("group", observed=True):
        pos = sub[sub["y_true"] == 1]
        neg = sub[sub["y_true"] == 0]
        rows.append({
            "group": str(name),
            "n": len(sub),
            "selection_rate": float(sub["y_pred"].mean()),
            "recall": float((pos["y_pred"] == 1).mean()) if len(pos) else np.nan,
            "fpr": float((neg["y_pred"] == 1).mean()) if len(neg) else np.nan,
            "fnr": float((pos["y_pred"] == 0).mean()) if len(pos) else np.nan,
        })
    return pd.DataFrame(rows)


def subgroup_performance(estimator, X, y, sensitive, threshold=FINAL_THRESHOLD) -> pd.DataFrame:
    """Performance par sous-groupe pour une variable sensible (âge, profession)."""
    proba = estimator.predict_proba(X)[:, 1]
    y_pred = (proba >= threshold).astype(int)
    return group_performance(y, y_pred, X[sensitive])
