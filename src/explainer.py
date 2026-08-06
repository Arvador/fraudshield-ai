# Module 3 - IA Explicable (SHAP)
import matplotlib.pyplot as plt
import pandas as pd
import shap


def get_explainer(model) -> shap.TreeExplainer:
    return shap.TreeExplainer(model)


def compute_shap_values(explainer: shap.TreeExplainer, X) -> shap.Explanation:
    explanation = explainer(X)
    if explanation.values.ndim == 3:
        # (n_samples, n_features, n_classes) -> on garde la classe positive (fraude)
        explanation = explanation[:, :, 1]
    return explanation


def plot_global_importance(explanation: shap.Explanation, max_display: int = 20) -> plt.Figure:
    shap.summary_plot(explanation, max_display=max_display, show=False)
    fig = plt.gcf()
    plt.tight_layout()
    return fig


def plot_transaction_explanation(explanation: shap.Explanation, index: int = 0) -> plt.Figure:
    shap.plots.waterfall(explanation[index], show=False)
    fig = plt.gcf()
    plt.tight_layout()
    return fig


def top_contributing_features(explanation: shap.Explanation, index: int, top_n: int = 10) -> pd.DataFrame:
    row = explanation[index]
    contributions = pd.DataFrame({
        "feature": row.feature_names,
        "valeur": row.data,
        "impact_shap": row.values,
    })
    contributions["impact_abs"] = contributions["impact_shap"].abs()
    return contributions.sort_values("impact_abs", ascending=False).head(top_n).drop(columns="impact_abs")


# Glossaire des familles de variables IEEE-CIS, pour un public non technique.
# Les colonnes V1-V339 sont des variables comportementales proprietaires (Vesta,
# l'organisateur du jeu de donnees) dont la signification exacte n'est pas
# publiee : on l'indique honnetement plutot que d'inventer une explication.
FEATURE_GLOSSARY = {
    "TransactionAmt": "Montant de la transaction",
    "ProductCD": "Categorie de produit achete",
    "addr1": "Zone geographique de facturation",
    "addr2": "Pays de facturation",
    "dist1": "Distance entre l'adresse de facturation et de livraison",
    "dist2": "Distance entre l'adresse de facturation et de livraison (variante)",
    "P_emaildomain": "Domaine email de l'acheteur",
    "R_emaildomain": "Domaine email du destinataire",
    "DeviceType": "Type d'appareil utilise (mobile, ordinateur...)",
    "DeviceInfo": "Modele/systeme de l'appareil utilise",
    "has_identity_data": "Presence de donnees d'identification pour cette transaction",
}
FEATURE_GLOSSARY_PREFIXES = {
    "card": "Information de carte bancaire (reseau, banque, type)",
    "C": "Compteur (ex. nombre d'adresses ou d'appareils associes a la carte)",
    "D": "Delai temporel (ex. jours depuis la transaction precedente)",
    "M": "Indicateur de correspondance (ex. nom = adresse de facturation)",
    "V": "Variable comportementale proprietaire (Vesta), non documentee publiquement",
    "id_": "Information technique sur l'appareil ou la session",
}


def describe_feature(feature_name: str) -> str:
    if feature_name in FEATURE_GLOSSARY:
        return FEATURE_GLOSSARY[feature_name]
    for prefix, description in FEATURE_GLOSSARY_PREFIXES.items():
        if feature_name.startswith(prefix):
            return description
    return "Variable non documentee"


def humanize_contributions(contributions: pd.DataFrame, top_n: int = 5) -> list[str]:
    top = contributions.reindex(contributions["impact_shap"].abs().sort_values(ascending=False).index).head(top_n)
    if len(top) == 0:
        return []

    max_impact = top["impact_shap"].abs().max()
    sentences = []
    for _, r in top.iterrows():
        direction = "augmente" if r["impact_shap"] > 0 else "diminue"
        ratio = abs(r["impact_shap"]) / max_impact if max_impact else 0
        intensity = "fortement" if ratio > 0.66 else "modérément" if ratio > 0.33 else "légèrement"
        description = describe_feature(r["feature"])
        sentences.append(f"{description} {direction} {intensity} le risque de fraude.")
    return sentences
