# Module 4 - Application Streamlit
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import TARGET_COL, decode_value, get_column_types, load_mappings
from src.models import get_feature_columns, load_model, time_based_split
from src.explainer import (
    compute_shap_values,
    describe_feature,
    get_explainer,
    humanize_contributions,
    plot_transaction_explanation,
    top_contributing_features,
)

DATA_PATH = os.path.join("data", "processed", "train_clean.parquet")
MODEL_PATH = os.path.join("models", "lightgbm_baseline.pkl")
MAPPINGS_PATH = os.path.join("data", "processed", "category_mappings.pkl")
SAMPLE_SIZE = 500
RANDOM_STATE = 42

st.set_page_config(page_title="FraudShield AI", page_icon=":shield:", layout="wide")


@st.cache_resource
def load_model_cached():
    return load_model(MODEL_PATH)


@st.cache_resource
def load_mappings_cached():
    return load_mappings(MAPPINGS_PATH)


@st.cache_data
def load_sample():
    df = pd.read_parquet(DATA_PATH)
    feature_cols = get_feature_columns(df)
    _, test = time_based_split(df, test_size=0.2)
    sample = test.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)
    return sample, feature_cols


@st.cache_resource
def get_explainer_cached(_model):
    return get_explainer(_model)


st.title(":shield: FraudShield AI")
st.caption("Détection de fraude bancaire par IA explicable — LUMERIA BANK")

with st.expander(":question: Comment utiliser cette application ?", expanded=False):
    st.markdown(
        """
        1. **Choisissez une transaction** dans le menu à gauche (parmi un échantillon de test).
        2. Le bandeau du haut donne le **verdict** : le risque estimé et la décision du modèle.
        3. La section **"En résumé"** explique en une phrase les principales raisons de cette décision.
        4. Le graphique détaillé plus bas (réservé aux profils techniques) montre l'impact
           précis de chaque variable, si vous voulez aller plus loin.

        Le **seuil de décision** (menu de gauche) fixe à partir de quelle probabilité une
        transaction est bloquée. Le baisser détecte plus de fraudes mais bloque aussi plus
        de transactions légitimes — c'est un arbitrage métier, pas une valeur figée.
        """
    )

model = load_model_cached()
mappings = load_mappings_cached()
sample, feature_cols = load_sample()
explainer = get_explainer_cached(model)

X_sample = sample[feature_cols].astype(np.float32)
sample["proba_fraude"] = model.predict_proba(X_sample)[:, 1]

st.sidebar.header("Sélection de la transaction")
sort_option = st.sidebar.radio(
    "Trier l'échantillon par",
    ["Probabilité de fraude (décroissant)", "TransactionID"],
)
if sort_option.startswith("Probabilité"):
    display_order = sample["proba_fraude"].sort_values(ascending=False).index
else:
    display_order = sample["TransactionID"].sort_values().index

idx = st.sidebar.selectbox(
    "Transaction",
    options=display_order,
    format_func=lambda i: f"#{int(sample.loc[i, 'TransactionID'])} — {sample.loc[i, 'proba_fraude']:.1%} de risque",
)

threshold = st.sidebar.slider(
    "Seuil de décision",
    0.0, 1.0, 0.5, 0.01,
    help="Une transaction est bloquée si sa probabilité de fraude dépasse ce seuil.",
)

row = sample.loc[idx]
proba = row["proba_fraude"]
is_flagged = proba >= threshold
is_actual_fraud = row[TARGET_COL] == 1

if is_flagged:
    st.error(f"### :rotating_light: Transaction jugée À RISQUE ({proba:.1%} de probabilité de fraude)")
else:
    st.success(f"### :white_check_mark: Transaction jugée SÛRE ({proba:.1%} de probabilité de fraude)")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Probabilité de fraude", f"{proba:.1%}")
col2.metric("Décision du modèle", "Bloquée" if is_flagged else "Autorisée")
col3.metric("Montant", f"${row['TransactionAmt']:.2f}")
col4.metric("Vérité terrain (historique)", "Fraude" if is_actual_fraud else "Légitime")

st.divider()

st.subheader("Détails de la transaction")
detail_cols = ["ProductCD", "card4", "card6", "DeviceType"]
details = {col: decode_value(mappings, col, row[col]) for col in detail_cols}
details_df = pd.DataFrame([details])
st.table(details_df)

st.divider()

explanation = compute_shap_values(explainer, X_sample.loc[[idx]])
contributions = top_contributing_features(explanation, index=0, top_n=10)

st.subheader(":speech_balloon: En résumé")
for sentence in humanize_contributions(contributions, top_n=5):
    st.markdown(f"- {sentence}")
st.caption(
    "Certaines variables (préfixe V) sont des indicateurs comportementaux propriétaires du "
    "fournisseur de données, dont le détail exact n'est pas public — leur nom seul n'est donc "
    "pas parlant, mais leur effet mesuré (augmente/diminue le risque) l'est."
)

with st.expander(":gear: Détails techniques (pour data scientists)", expanded=False):
    st.markdown(
        "Graphique **SHAP** : chaque barre est une variable. **Rouge = augmente** le risque de "
        "fraude, **bleu = diminue** le risque. La longueur de la barre indique l'ampleur de l'effet."
    )
    fig = plot_transaction_explanation(explanation, index=0)
    st.pyplot(fig)

    st.markdown("**Top 10 variables contributives**")
    contributions_display = contributions.copy()
    contributions_display.insert(1, "description", contributions_display["feature"].apply(describe_feature))
    st.dataframe(contributions_display, use_container_width=True)
