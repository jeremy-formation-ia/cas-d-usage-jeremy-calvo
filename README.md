# Cas d'usage IA — Marketing bancaire

Prédire la souscription à un dépôt à terme à partir du jeu de données [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing), pour prioriser les campagnes de démarchage téléphonique d'une banque de détail.

**Dépôt GitHub** : https://github.com/jeremy-formation-ia/cas-d-usage-jeremy-calvo

## Contenu

| Dossier / fichier | Rôle |
|---|---|
| `notebooks/cas-usage-marketing-bancaire.ipynb` | Livrable principal : cadrage, EDA, préparation, modélisation, arbitrage, interprétation, industrialisation, suivi, éthique |
| `notebooks/journal-de-bord.ipynb` | Journal de bord (livrable de certification) |
| `src/` | Logique métier : `config.py`, `data.py`, `preprocessing.py`, `modeling.py`, `train.py`, `evaluate.py`, `fairness.py` |
| `api/` | API FastAPI (`/health`, `/info`, `/predict`, `/metrics`) + tests |
| `ui/` | Interface Streamlit pour les conseillers |
| `models/` | Modèle persisté (`model.joblib`) + métadonnées (`model.json`) |
| `data/` | Dataset UCI + jeu de référence pour l'évaluation continue |
| `prometheus/`, `grafana/` | Monitoring (Prometheus :9090, Grafana :3001) |
| `scripts/generate_traffic.py` | Trafic artificiel pour le dashboard |
| `.github/workflows/ci.yml` | CI : tests, évaluation continue (garde-fou), build Docker |
| `experiments.md` | Suivi des expériences (modèles × scénarios) |
| `Sujet_Examen_Atlas_IA_Marketing_Bancaire.pdf` | Sujet de l'examen |

## Modèle retenu

HistGradientBoosting + scénario S4 (sans `duration`, sans variables sensibles, sans historique de campagne) + seuil 0,30. Test scellé : rappel 0,851 · précision 0,178 · F1 macro 0,477 · ROC-AUC 0,807.

## Utilisation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/train.py                 # entraîne et persiste le modèle
uvicorn api.main:app --reload       # API sur http://localhost:8000
streamlit run ui/app.py             # UI sur http://localhost:8501
docker compose up                   # API + UI + Prometheus + Grafana
pytest tests/ api/tests/            # 38 tests
```

Le notebook s'exécute avec le kernel du venv (Python 3.11, scikit-learn 1.5.1 — le modèle sérialisé est incompatible avec une version différente).
