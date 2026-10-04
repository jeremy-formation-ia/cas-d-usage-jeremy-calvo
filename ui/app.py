"""Interface Streamlit — estimation de la probabilité de souscription.

Formulaire pour saisir un profil client et interroger l'API de prédiction.
"""

import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Marketing bancaire — Ciblage", page_icon="📞")

st.title("Ciblage des campagnes de démarchage")
st.caption(
    "Estime la probabilité qu'un client souscrive à un dépôt à terme. "
    "Outil d'aide à la décision : le conseiller garde la main."
)

with st.form("client_form"):
    col1, col2 = st.columns(2)

    with col1:
        default = st.selectbox("Crédit en défaut", ["no", "unknown", "yes"])
        housing = st.selectbox("Prêt immobilier", ["yes", "no", "unknown"])
        loan = st.selectbox("Prêt personnel", ["no", "yes", "unknown"])
        contact = st.selectbox("Moyen de contact", ["cellular", "telephone"])
        month = st.selectbox(
            "Mois du dernier contact",
            ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
            index=2,
        )
        day_of_week = st.selectbox("Jour de la semaine", ["mon", "tue", "wed", "thu", "fri"])

    with col2:
        emp_var_rate = st.number_input("Taux de variation de l'emploi", value=1.1, step=0.1)
        cons_price_idx = st.number_input("Indice des prix", value=93.994, step=0.01, format="%.3f")
        cons_conf_idx = st.number_input("Indice de confiance", value=-36.4, step=0.1)
        euribor3m = st.number_input("Euribor 3 mois", value=4.857, step=0.01, format="%.3f")
        nr_employed = st.number_input("Nombre de salariés", value=5191.0, step=1.0)

    submitted = st.form_submit_button("Estimer")

if submitted:
    payload = {
        "default": default,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "month": month,
        "day_of_week": day_of_week,
        "emp.var.rate": emp_var_rate,
        "cons.price.idx": cons_price_idx,
        "cons.conf.idx": cons_conf_idx,
        "euribor3m": euribor3m,
        "nr.employed": nr_employed,
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
