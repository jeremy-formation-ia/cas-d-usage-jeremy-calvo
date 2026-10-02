"""Configuration centrale : chemins, constantes et paramètres reproductibles."""

from pathlib import Path

# Racine du dépôt (le dossier qui contient data/)
REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = REPO_ROOT / "data" / "bank-additional-full.csv"

RANDOM_STATE = 42

TARGET_COLUMN = "y"
TARGET_POSITIVE = "yes"
TARGET_NEGATIVE = "no"

# Sentinelle de pdays signifiant « jamais recontacté »
PDAYS_SENTINEL = 999
PREVIOUSLY_CONTACTED_COLUMN = "previously_contacted"

# Jeu de variables sensibles retirées au scénario 3
SENSITIVE_FEATURES = ["age", "job", "marital", "education"]

# Historique de campagne retiré au scénario 4
CAMPAIGN_HISTORY_FEATURES = ["campaign", "previously_contacted", "previous", "poutcome"]

# Colonnes par nature (l'ensemble effectif dépend du scénario)
NUMERIC_FEATURES = [
    "age", "duration", "campaign", "previous", "previously_contacted",
    "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed",
]
CATEGORICAL_FEATURES = [
    "job", "marital", "default", "housing", "loan",
    "contact", "month", "day_of_week", "poutcome",
]
ORDINAL_FEATURES = ["education"]

# Ordre sémantique des niveaux d'éducation (les « unknown » sont codés à part, -1)
EDUCATION_ORDER = [
    "illiterate",
    "basic.4y",
    "basic.6y",
    "basic.9y",
    "high.school",
    "professional.course",
    "university.degree",
]

# Modélisation
N_SPLITS = 5
DECISION_THRESHOLD = 0.5
PRIMARY_METRIC = "f1_macro"
