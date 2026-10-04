"""Schémas Pydantic de l'API de prédiction de souscription.

Le schéma d'entrée est aligné sur les variables du scénario retenu (S4).
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

YesNoUnknown = Literal["yes", "no", "unknown"]
Month = Literal["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
DayOfWeek = Literal["mon", "tue", "wed", "thu", "fri"]

_EXAMPLE = {
    "default": "no",
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "month": "may",
    "day_of_week": "mon",
    "emp.var.rate": 1.1,
    "cons.price.idx": 93.994,
    "cons.conf.idx": -36.4,
    "euribor3m": 4.857,
    "nr.employed": 5191.0,
}


class ClientProfile(BaseModel):
    """Profil client en entrée de /predict (variables du scénario S4)."""

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={"example": _EXAMPLE},
    )

    default: YesNoUnknown = Field(..., description="Crédit en défaut de paiement")
    housing: YesNoUnknown = Field(..., description="Prêt immobilier en cours")
    loan: YesNoUnknown = Field(..., description="Prêt personnel en cours")
    contact: Literal["cellular", "telephone"] = Field(..., description="Moyen de contact")
    month: Month = Field(..., description="Mois du dernier contact")
    day_of_week: DayOfWeek = Field(..., description="Jour de la semaine du dernier contact")
    emp_var_rate: float = Field(
        ..., ge=-3.4, le=1.4, alias="emp.var.rate", description="Taux de variation de l'emploi"
    )
    cons_price_idx: float = Field(
        ..., ge=92.0, le=95.0, alias="cons.price.idx", description="Indice des prix"
    )
    cons_conf_idx: float = Field(
        ..., ge=-51.0, le=-26.0, alias="cons.conf.idx", description="Indice de confiance"
    )
    euribor3m: float = Field(..., ge=0.6, le=5.1, description="Taux Euribor 3 mois")
    nr_employed: float = Field(
        ..., ge=4960.0, le=5230.0, alias="nr.employed", description="Nombre de salariés"
    )


class PredictionResponse(BaseModel):
    """Réponse de /predict."""

    subscription_probability: float = Field(..., ge=0.0, le=1.0)
    decision: Literal["contact", "do_not_contact", "abstain"]
    threshold: float
    model_version: str
    request_id: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool


class InfoResponse(BaseModel):
    api_version: str
    model_name: str
    model_version: str
    scenario: str
    decision_threshold: float
    sklearn_version: str | None = None
    dataset_sha256: str | None = None
    metrics_test: dict | None = None
