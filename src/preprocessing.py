"""Recette de prétraitement et définition des scénarios de variables."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.config import (
    CAMPAIGN_HISTORY_FEATURES,
    CATEGORICAL_FEATURES,
    EDUCATION_ORDER,
    NUMERIC_FEATURES,
    ORDINAL_FEATURES,
    SENSITIVE_FEATURES,
)


def make_preprocessor(columns) -> ColumnTransformer:
    """Construit le ColumnTransformer pour un jeu de colonnes donné.

    Mêmes transformations pour tous les scénarios : seul le périmètre change.
    Les imputers sont conservés pour la robustesse en production (aucun
    manquant aujourd'hui, mais la recette doit tenir si un champ vide apparaît).
    """
    columns = list(columns)

    numeric = [c for c in NUMERIC_FEATURES if c in columns]
    categorical = [c for c in CATEGORICAL_FEATURES if c in columns]
    ordinal = [c for c in ORDINAL_FEATURES if c in columns]

    transformers = []

    if numeric:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric))

    if categorical:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical))

    if ordinal:
        transformers.append(("ord", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(
                categories=[EDUCATION_ORDER],
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )),
        ]), ordinal))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_scenarios(all_features) -> dict[str, list[str]]:
    """Construit les scénarios emboîtés de jeux de variables.

    S1 : toutes les variables (référence).
    S2 : sans ``duration`` (fuite d'information).
    S3 : sans variables sensibles (éthique).
    S4 : sans historique de campagne (modèle minimal).
    """
    all_features = list(all_features)

    s2 = [c for c in all_features if c != "duration"]
    s3 = [c for c in s2 if c not in SENSITIVE_FEATURES]
    s4 = [c for c in s3 if c not in CAMPAIGN_HISTORY_FEATURES]

    return {"S1": all_features, "S2": s2, "S3": s3, "S4": s4}


if __name__ == "__main__":
    from src.data import load_prepared

    X, _ = load_prepared()
    scenarios = build_scenarios(X.columns)
    for name, cols in scenarios.items():
        print(f"{name} : {len(cols)} variables")
