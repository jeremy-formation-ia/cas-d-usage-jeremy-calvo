"""Chargement, nettoyage et construction des variables."""

import pandas as pd

from src.config import (
    DATA_PATH,
    PDAYS_SENTINEL,
    PREVIOUSLY_CONTACTED_COLUMN,
    TARGET_COLUMN,
    TARGET_POSITIVE,
)


def load_dataset(path=DATA_PATH) -> pd.DataFrame:
    """Charge le jeu de données brut (séparateur point-virgule)."""
    return pd.read_csv(path, sep=";")


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les doublons exacts et réinitialise l'index.

    Les doublons représentent une part négligeable du jeu et, faute
    d'identifiant client, on ne peut pas déterminer s'il s'agit d'un même
    client ou de deux clients au profil identique.
    """
    return df.drop_duplicates().reset_index(drop=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construit les variables dérivées.

    ``pdays`` est une sentinelle (999 = jamais recontacté), pas une durée :
    on la remplace par un indicateur binaire ``previously_contacted``.
    """
    df = df.copy()
    df[PREVIOUSLY_CONTACTED_COLUMN] = (df["pdays"] != PDAYS_SENTINEL).astype(int)
    return df.drop(columns="pdays")


def get_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Sépare les variables explicatives de la cible encodée (oui -> 1)."""
    y = (df[TARGET_COLUMN] == TARGET_POSITIVE).astype(int)
    X = df.drop(columns=TARGET_COLUMN)
    return X, y


def load_prepared() -> tuple[pd.DataFrame, pd.Series]:
    """Chaîne complète : chargement, nettoyage, features, séparation X/y."""
    df = build_features(clean_dataset(load_dataset()))
    return get_X_y(df)


if __name__ == "__main__":
    X, y = load_prepared()
    print("Dimensions de X :", X.shape)
    print("Répartition de la cible :")
    print(y.value_counts(normalize=True).round(4))
