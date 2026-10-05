"""Métriques Prometheus de l'API.

Expose des métriques métier en complément des métriques HTTP automatiques :
distribution des décisions, distribution des probabilités et compteur
d'abstentions. Ce sont ces séries que Grafana affiche.
"""

from prometheus_client import Counter, Histogram

PREDICTIONS_TOTAL = Counter(
    "bank_marketing_predictions_total",
    "Nombre de prédictions servies, par décision.",
    labelnames=("decision",),
)

PREDICTION_PROBABILITY = Histogram(
    "bank_marketing_prediction_probability",
    "Distribution des probabilités de souscription prédites.",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

ABSTENTIONS_TOTAL = Counter(
    "bank_marketing_abstentions_total",
    "Nombre de prédictions en zone d'incertitude (renvoi au conseiller).",
)


def observe_prediction(decision: str, probability: float) -> None:
    """Enregistre une prédiction (décision + probabilité)."""
    PREDICTIONS_TOTAL.labels(decision=decision).inc()
    PREDICTION_PROBABILITY.observe(probability)
    if decision == "abstain":
        ABSTENTIONS_TOTAL.inc()
