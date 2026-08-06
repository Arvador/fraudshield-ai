import numpy as np
import pandas as pd
import pytest

from src.pipeline import get_column_types
from src.models import (
    evaluate_model,
    get_feature_columns,
    time_based_split,
    train_lightgbm,
)


def test_time_based_split_respects_chronological_order():
    df = pd.DataFrame({
        "TransactionDT": [300, 100, 500, 200, 400],
        "value": ["e", "a", "z", "b", "y"],
    })
    train, test = time_based_split(df, test_size=0.4)

    assert train["TransactionDT"].max() <= test["TransactionDT"].min()
    assert len(train) + len(test) == len(df)


def test_get_feature_columns_excludes_ids_and_target():
    df = pd.DataFrame({
        "TransactionID": [1, 2],
        "TransactionDT": [10, 20],
        "isFraud": [0, 1],
        "TransactionAmt": [50.0, 75.0],
    })
    features = get_feature_columns(df)

    assert features == ["TransactionAmt"]


@pytest.fixture
def synthetic_dataset():
    rng = np.random.default_rng(42)
    n = 300
    df = pd.DataFrame({
        "TransactionID": np.arange(n),
        "TransactionDT": np.arange(n) * 100,
        "isFraud": (rng.random(n) < 0.2).astype(int),
        "TransactionAmt": rng.uniform(1, 500, n).astype("float32"),
        "ProductCD": rng.integers(0, 4, n).astype("int32"),
        "card1": rng.integers(0, 10, n).astype("int32"),
    })
    return df


def test_train_and_evaluate_lightgbm_runs_end_to_end(synthetic_dataset):
    train, test = time_based_split(synthetic_dataset, test_size=0.3)
    feature_cols = get_feature_columns(synthetic_dataset)
    categorical_cols, _ = get_column_types(synthetic_dataset)

    model = train_lightgbm(
        train,
        feature_cols,
        categorical_cols,
        params={"n_estimators": 10, "num_leaves": 7, "min_child_samples": 5, "verbosity": -1},
    )
    metrics = evaluate_model(model, test, feature_cols)

    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert len(metrics["confusion_matrix"]) == 2
