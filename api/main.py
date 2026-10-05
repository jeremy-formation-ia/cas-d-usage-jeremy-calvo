"""API de prédiction de souscription — Marketing bancaire (Atlas IA).

Expose le modèle retenu (HistGradientBoosting, scénario S4, seuil 0,30) :
- ``GET /health`` : état du service ;
- ``GET /info`` : métadonnées du modèle ;
- ``POST /predict`` : probabilité de souscription pour un profil client.
"""

import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from loguru import logger
from prometheus_fastapi_instrumentator import Instrumentator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.metrics import observe_prediction
from api.middleware import LoggingMiddleware
from api.schemas import ClientProfile, HealthResponse, InfoResponse, PredictionResponse

# --- Journalisation ----------------------------------------------------------

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)
logger.remove()
logger.add(sys.stderr, level="INFO", colorize=True)
logger.add(
    LOGS_DIR / "api.log",
    rotation="10 MB",
    retention="7 days",
    compression="zip",
    serialize=True,
    enqueue=True,
    level="INFO",
)

# --- Artefacts --------------------------------------------------------------

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"
METADATA_PATH = MODELS_DIR / "model.json"

# Zone d'incertitude (abstention) autour du seuil de décision
REJECTION_LOW = 0.25
REJECTION_HIGH = 0.35


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle et ses métadonnées au démarrage (une seule fois)."""
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise RuntimeError(f"Artefacts manquants dans {MODELS_DIR} (lancer `python -m src.train`)")
    app.state.model = joblib.load(MODEL_PATH)
    app.state.metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    logger.info(
        "Modèle chargé : {name} {version} (scénario {scenario})",
        name=app.state.metadata["model_name"],
        version=app.state.metadata["model_version"],
        scenario=app.state.metadata["scenario"],
    )
    yield
    app.state.model = None
    logger.info("Modèle libéré")


app = FastAPI(
    title="Bank Marketing API",
    version="1.0.0",
    description="Prédiction de souscription à un dépôt à terme (modèle bank_marketing v1.0.0).",
    lifespan=lifespan,
)
app.add_middleware(LoggingMiddleware)

# Métriques HTTP + endpoint /metrics (lu par Prometheus)
Instrumentator(should_group_status_codes=False).instrument(app).expose(
    app, endpoint="/metrics", include_in_schema=False
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """État du service et du modèle."""
    loaded = getattr(app.state, "model", None) is not None
    return HealthResponse(status="ok" if loaded else "degraded", model_loaded=loaded)


@app.get("/info", response_model=InfoResponse)
async def info() -> InfoResponse:
    """Métadonnées du modèle servi."""
    if getattr(app.state, "model", None) is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Modèle non chargé")
    meta = app.state.metadata
    return InfoResponse(
        api_version=app.version,
        model_name=meta["model_name"],
        model_version=meta["model_version"],
        scenario=meta["scenario"],
        decision_threshold=meta["decision_threshold"],
        sklearn_version=meta.get("sklearn_version"),
        dataset_sha256=meta.get("dataset_sha256"),
        metrics_test=meta.get("metrics_test"),
    )


@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict(profile: ClientProfile, request: Request) -> PredictionResponse:
    """Estime la probabilité de souscription d'un client."""
    if getattr(app.state, "model", None) is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Modèle non chargé")

    request_id = getattr(request.state, "request_id", "n/a")
    threshold = app.state.metadata["decision_threshold"]

    try:
        X = pd.DataFrame([profile.model_dump(by_alias=True)])
        proba = float(app.state.model.predict_proba(X)[0, 1])
    except Exception as exc:  # noqa: BLE001 — garde large en production
        logger.bind(request_id=request_id).exception("Échec de la prédiction")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Échec de la prédiction : {exc.__class__.__name__}",
        ) from exc

    if REJECTION_LOW <= proba <= REJECTION_HIGH:
        decision = "abstain"
    elif proba >= threshold:
        decision = "contact"
    else:
        decision = "do_not_contact"

    logger.bind(request_id=request_id).info(
        "Prédiction : proba={proba:.3f} décision={decision}", proba=proba, decision=decision
    )

    observe_prediction(decision=decision, probability=proba)

    return PredictionResponse(
        subscription_probability=round(proba, 4),
        decision=decision,
        threshold=threshold,
        model_version=app.state.metadata["model_version"],
        request_id=request_id,
    )
