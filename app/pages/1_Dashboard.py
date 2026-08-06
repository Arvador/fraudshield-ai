# Page Dashboard - vue d'ensemble volumes, tendances, performance du modele
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from common import load_mappings_cached, load_model_cached, load_test_with_predictions
from src.pipeline import TARGET_COL, decode_value

st.set_page_config(page_title="FraudShield AI - Dashboard", page_icon=":bar_chart:", layout="wide")

st.title(":bar_chart: Dashboard FraudShield")
st.caption(
    "Vue d'ensemble sur l'échantillon de test (transactions les plus récentes, jamais vues "
    "par le modèle à l'entraînement) — équivalent d'un reporting Power BI, intégré à l'application."
)

model = load_model_cached()
mappings = load_mappings_cached()
test, feature_cols = load_test_with_predictions()

threshold = st.sidebar.slider(
    "Seuil de décision",
    0.0, 1.0, 0.5, 0.01,
    help="Une transaction est bloquée si sa probabilité de fraude dépasse ce seuil.",
)
predicted_flag = (test["proba_fraude"] >= threshold).astype(int)

# --- KPIs -----------------------------------------------------------------
roc_auc = roc_auc_score(test[TARGET_COL], test["proba_fraude"])
pr_auc = average_precision_score(test[TARGET_COL], test["proba_fraude"])
precision = precision_score(test[TARGET_COL], predicted_flag, zero_division=0)
recall = recall_score(test[TARGET_COL], predicted_flag, zero_division=0)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Transactions (test)", f"{len(test):,}")
col2.metric("Taux de fraude réel", f"{test[TARGET_COL].mean():.2%}")
col3.metric("ROC-AUC", f"{roc_auc:.3f}")
col4.metric("PR-AUC", f"{pr_auc:.3f}")
col5.metric("Transactions bloquées", f"{predicted_flag.mean():.1%}")

st.caption(
    f"Au seuil actuel ({threshold:.2f}) : précision {precision:.1%} (parmi les transactions "
    f"bloquées, proportion de vraies fraudes) — rappel {recall:.1%} (proportion des fraudes "
    "réellement détectées)."
)

st.divider()

# --- Tendance dans le temps -------------------------------------------------
st.subheader("Évolution du taux de fraude")
st.caption(
    "`TransactionDT` est un temps relatif (secondes depuis un point de référence arbitraire, "
    "pas une date calendaire réelle) — on l'agrège ici en \"jours relatifs\" depuis le début du jeu de test."
)
daily = test.copy()
daily["jour_relatif"] = daily["TransactionDT"] // 86400
daily_stats = daily.groupby("jour_relatif").agg(
    volume=(TARGET_COL, "count"),
    taux_fraude=(TARGET_COL, "mean"),
).reset_index()

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**Volume de transactions par jour**")
    st.bar_chart(daily_stats.set_index("jour_relatif")["volume"])
with col_b:
    st.markdown("**Taux de fraude par jour**")
    st.line_chart(daily_stats.set_index("jour_relatif")["taux_fraude"])

st.divider()

# --- Segments a risque -------------------------------------------------------
st.subheader("Taux de fraude par segment")


def fraud_rate_by_decoded(df: pd.DataFrame, col: str) -> pd.Series:
    grouped = df.groupby(col)[TARGET_COL].mean().sort_values(ascending=False)
    grouped.index = [decode_value(mappings, col, code) for code in grouped.index]
    return grouped


col_c, col_d = st.columns(2)
with col_c:
    st.markdown("**Par catégorie de produit (`ProductCD`)**")
    st.bar_chart(fraud_rate_by_decoded(test, "ProductCD"))
with col_d:
    st.markdown("**Par type d'appareil (`DeviceType`)**")
    st.bar_chart(fraud_rate_by_decoded(test, "DeviceType"))

st.divider()

# --- Performance du modele ----------------------------------------------
st.subheader("Performance du modèle")

col_e, col_f = st.columns(2)

with col_e:
    st.markdown("**Matrice de confusion**")
    cm = confusion_matrix(test[TARGET_COL], predicted_flag)
    cm_df = pd.DataFrame(
        cm,
        index=["Vrai : non-fraude", "Vrai : fraude"],
        columns=["Prédit : non-fraude", "Prédit : fraude"],
    )
    st.table(cm_df)

with col_f:
    st.markdown("**Distribution des probabilités prédites**")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(test.loc[test[TARGET_COL] == 0, "proba_fraude"], bins=30, alpha=0.6, label="Non-fraude", color="#4C72B0")
    ax.hist(test.loc[test[TARGET_COL] == 1, "proba_fraude"], bins=30, alpha=0.6, label="Fraude", color="#C44E52")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1, label=f"Seuil ({threshold:.2f})")
    ax.set_xlabel("Probabilité de fraude prédite")
    ax.set_ylabel("Nombre de transactions")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)

st.markdown("**Top 15 variables — importance globale du modèle**")
importances = pd.Series(model.feature_importances_, index=feature_cols)
top15 = importances.sort_values(ascending=False).head(15)
fig2, ax2 = plt.subplots(figsize=(8, 5))
top15.sort_values().plot.barh(ax=ax2, color="#4C72B0")
ax2.set_xlabel("Importance (nombre de splits)")
plt.tight_layout()
st.pyplot(fig2)
