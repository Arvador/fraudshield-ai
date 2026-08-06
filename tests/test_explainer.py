import pandas as pd

from src.explainer import describe_feature, humanize_contributions


def test_describe_feature_known_column():
    assert describe_feature("TransactionAmt") == "Montant de la transaction"


def test_describe_feature_prefix_match():
    assert "carte" in describe_feature("card1")
    assert "Vesta" in describe_feature("V258")


def test_describe_feature_unknown_falls_back():
    assert describe_feature("colonne_bizarre") == "Variable non documentee"


def test_humanize_contributions_orders_by_absolute_impact():
    contributions = pd.DataFrame({
        "feature": ["TransactionAmt", "card1", "D2"],
        "valeur": [500, 111, 30],
        "impact_shap": [0.1, -2.0, 0.5],
    })
    sentences = humanize_contributions(contributions, top_n=2)

    assert len(sentences) == 2
    assert "carte" in sentences[0]
    assert "diminue" in sentences[0]
