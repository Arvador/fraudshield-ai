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
