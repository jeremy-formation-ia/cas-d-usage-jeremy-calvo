"""Entraînement et persistance du modèle retenu.

Modèle : HistGradientBoosting sur le scénario S4, seuil de décision 0,30
(arbitrage de la section 6). Le pipeline complet (prétraitement + modèle)
est persisté, ainsi que ses métadonnées.
"""

import hashlib
import json
import platform
from datetime import datetime, timezone

import joblib
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_PATH,
    FINAL_SCENARIO,
    FINAL_THRESHOLD,
    METADATA_PATH,
    MODEL_NAME,
    MODEL_PATH,
    MODEL_VERSION,
    MODELS_DIR,
    RANDOM_STATE,
)
from src.data import load_prepared
from src.modeling import compute_metrics, make_models, make_pipeline
from src.preprocessing import build_scenarios


def dataset_sha256(path=DATA_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train_and_evaluate() -> tuple:
    """Entraîne le modèle retenu et renvoie (pipeline, métriques test, colonnes)."""
    X, y = load_prepared()
    scenarios = build_scenarios(X.columns)
    columns = scenarios[FINAL_SCENARIO]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    pipeline = make_pipeline(make_models()["gradient_boosting"], columns)
    pipeline.fit(X_train, y_train)

    proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, proba, threshold=FINAL_THRESHOLD)
    return pipeline, metrics, columns


def save_model(pipeline, metrics, columns) -> tuple:
    """Persiste le pipeline et ses métadonnées."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH, compress=3)

    metadata = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scenario": FINAL_SCENARIO,
        "decision_threshold": FINAL_THRESHOLD,
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "dataset_sha256": dataset_sha256(),
        "feature_columns": list(columns),
        "metrics_test": {
            k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in metrics.items()
        },
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return MODEL_PATH, METADATA_PATH


def load_model():
    """Recharge le pipeline persisté."""
    return joblib.load(MODEL_PATH)


def predict_proba(records) -> list:
    """Probabilités de souscription pour une liste d'enregistrements."""
    pipeline = load_model()
    frame = pd.DataFrame(records)
    return pipeline.predict_proba(frame)[:, 1].tolist()


if __name__ == "__main__":
    pipeline, metrics, columns = train_and_evaluate()
    model_path, meta_path = save_model(pipeline, metrics, columns)
    print(f"Modèle persisté : {model_path}")
    print(f"Métadonnées     : {meta_path}")
    print(f"Scénario : {FINAL_SCENARIO} | seuil : {FINAL_THRESHOLD}")
    print("Métriques test :")
    for k in ["f1_macro", "precision_positive", "recall_positive", "roc_auc"]:
        print(f"  {k:20s}: {metrics[k]:.3f}")
