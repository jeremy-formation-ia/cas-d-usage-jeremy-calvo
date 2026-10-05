"""Génère du trafic artificiel vers l'API pour alimenter Prometheus / Grafana.

Envoie des profils clients aléatoires (valides selon le schéma de l'API) à
``POST /predict`` à intervalle régulier. Les métriques s'accumulent alors dans
Prometheus et les panneaux Grafana deviennent visibles.

Exemple :
    python scripts/generate_traffic.py --url http://localhost:8000 --requests 200 --delay 0.05
"""

import argparse
import random
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CHOICES = {
    "default": ["no", "unknown", "yes"],
    "housing": ["yes", "no", "unknown"],
    "loan": ["no", "yes", "unknown"],
    "contact": ["cellular", "telephone"],
    "month": ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
    "day_of_week": ["mon", "tue", "wed", "thu", "fri"],
}
RANGES = {
    "emp.var.rate": (-3.4, 1.4),
    "cons.price.idx": (92.201, 94.767),
    "cons.conf.idx": (-50.8, -26.9),
    "euribor3m": (0.634, 5.045),
    "nr.employed": (4963.6, 5228.1),
}


def random_payload() -> dict:
    payload = {k: random.choice(v) for k, v in CHOICES.items()}
    for key, (low, high) in RANGES.items():
        payload[key] = round(random.uniform(low, high), 3)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Générateur de trafic pour l'API")
    parser.add_argument("--url", default="http://localhost:8000", help="URL de base de l'API")
    parser.add_argument("--requests", type=int, default=200, help="nombre de requêtes à envoyer")
    parser.add_argument("--delay", type=float, default=0.05, help="délai entre requêtes (s)")
    parser.add_argument("--seed", type=int, default=None, help="graine aléatoire (reproductibilité)")
    parser.add_argument("--loop", action="store_true",
                        help="envoyer du trafic en continu (Ctrl+C pour arrêter)")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    endpoint = f"{args.url.rstrip('/')}/predict"
    decisions: dict[str, int] = {}
    errors = 0
    sent = 0

    print(f"Cible : {endpoint} | {'mode continu' if args.loop else str(args.requests) + ' requêtes'}")
    try:
        while True:
            sent += 1
            try:
                response = requests.post(endpoint, json=random_payload(), timeout=5)
                response.raise_for_status()
                decision = response.json().get("decision", "?")
                decisions[decision] = decisions.get(decision, 0) + 1
            except requests.RequestException as exc:
                errors += 1
                if errors == 1:
                    print(f"Première erreur : {exc}")
            if sent % 20 == 0:
                print(f"{sent} requêtes envoyées — {decisions} (erreurs: {errors})")
            if not args.loop and sent >= args.requests:
                break
            time.sleep(args.delay)
    except KeyboardInterrupt:
        print("\nInterruption manuelle.")

    print("\nRépartition des décisions :", decisions)
    print(f"Erreurs : {errors}")
    print("Ouvrir Grafana : http://localhost:3001 (admin/admin) → dashboard « Bank Marketing »")
    return 0 if errors < sent else 1


if __name__ == "__main__":
    sys.exit(main())
