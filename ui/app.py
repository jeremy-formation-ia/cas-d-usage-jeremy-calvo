"""Interface Streamlit — estimation de la probabilité de souscription.

Formulaire pour saisir un profil client et interroger l'API de prédiction.
"""

import os
import random

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

# Valeurs possibles et plages (issues de l'exploration des données)
CHOICES = {
    "default": ["no", "unknown", "yes"],
    "housing": ["yes", "no", "unknown"],
    "loan": ["no", "yes", "unknown"],
    "contact": ["cellular", "telephone"],
    "month": ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
    "day_of_week": ["mon", "tue", "wed", "thu", "fri"],
}
RANGES = {
    "emp_var_rate": (-3.4, 1.4),
    "cons_price_idx": (92.201, 94.767),
    "cons_conf_idx": (-50.8, -26.9),
    "euribor3m": (0.634, 5.045),
    "nr_employed": (4963.6, 5228.1),
}

DEFAULTS = {
    "default": "no",
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "month": "may",
    "day_of_week": "mon",
    "emp_var_rate": 1.1,
    "cons_price_idx": 93.994,
    "cons_conf_idx": -36.4,
    "euribor3m": 4.857,
    "nr_employed": 5191.0,
}

st.set_page_config(page_title="Marketing bancaire — Ciblage", page_icon="📞")

st.title("Ciblage des campagnes de démarchage")
st.caption(
    "Estime la probabilité qu'un client souscrive à un dépôt à terme. "
    "Outil d'aide à la décision : le conseiller garde la main."
)


def randomize_fields() -> None:
    """Remplit tous les champs avec des valeurs aléatoires."""
    for key, options in CHOICES.items():
        st.session_state[key] = random.choice(options)
    for key, (low, high) in RANGES.items():
        st.session_state[key] = round(random.uniform(low, high), 3)


for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.button("🎲 Valeurs aléatoires", on_click=randomize_fields)

with st.form("client_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.selectbox("Crédit en défaut", CHOICES["default"], key="default")
        st.selectbox("Prêt immobilier", CHOICES["housing"], key="housing")
        st.selectbox("Prêt personnel", CHOICES["loan"], key="loan")
        st.selectbox("Moyen de contact", CHOICES["contact"], key="contact")
        st.selectbox("Mois du dernier contact", CHOICES["month"], key="month")
        st.selectbox("Jour de la semaine", CHOICES["day_of_week"], key="day_of_week")

    with col2:
        st.number_input("Taux de variation de l'emploi", step=0.1, key="emp_var_rate")
        st.number_input("Indice des prix", step=0.01, format="%.3f", key="cons_price_idx")
        st.number_input("Indice de confiance", step=0.1, key="cons_conf_idx")
        st.number_input("Euribor 3 mois", step=0.01, format="%.3f", key="euribor3m")
        st.number_input("Nombre de salariés", step=1.0, key="nr_employed")

    submitted = st.form_submit_button("Estimer")

if submitted:
    payload = {
        "default": st.session_state["default"],
        "housing": st.session_state["housing"],
        "loan": st.session_state["loan"],
        "contact": st.session_state["contact"],
        "month": st.session_state["month"],
        "day_of_week": st.session_state["day_of_week"],
        "emp.var.rate": st.session_state["emp_var_rate"],
        "cons.price.idx": st.session_state["cons_price_idx"],
        "cons.conf.idx": st.session_state["cons_conf_idx"],
        "euribor3m": st.session_state["euribor3m"],
        "nr.employed": st.session_state["nr_employed"],
    }
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
    except requests.RequestException as exc:
        st.error(f"Erreur d'appel à l'API : {exc}")
    else:
        proba = result["subscription_probability"]
        decision = result["decision"]
        st.metric("Probabilité de souscription", f"{proba:.1%}")

        if decision == "contact":
            st.success("À contacter — probabilité supérieure au seuil.")
        elif decision == "abstain":
            st.warning("Confiance insuffisante — à transmettre au conseiller.")
        else:
            st.info("Ne pas contacter — probabilité faible.")

        st.caption(
            f"Seuil de décision : {result['threshold']:.2f} · "
            f"Modèle : {result['model_version']} · requête {result['request_id'][:8]}"
        )

st.divider()
st.caption("Le modèle ne s'appuie sur aucune variable personnelle sensible (âge, profession, situation familiale, éducation).")
