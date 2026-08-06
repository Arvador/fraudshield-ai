# Module 2 - Entrainement des modeles ML
import gc
import os

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from src.pipeline import ID_COLS, TARGET_COL, get_column_types

DEFAULT_PARAMS = {
    "objective": "binary",
    "metric": "average_precision",
    "is_unbalance": True,
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_bin": 63,
    "random_state": 42,
    "n_jobs": -1,
}


def time_based_split(df: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_sorted = df.sort_values("TransactionDT")
    split_idx = int(len(df_sorted) * (1 - test_size))
    train = df_sorted.iloc[:split_idx]
    test = df_sorted.iloc[split_idx:]
    return train, test


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ID_COLS and c != TARGET_COL]


def train_lightgbm(
    train: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
    params: dict | None = None,
) -> lgb.LGBMClassifier:
    model_params = {**DEFAULT_PARAMS, **(params or {})}
    model = lgb.LGBMClassifier(**model_params)

    # Cast en float32 homogene : un mix int32/float32 force numpy a promouvoir
    # en float64 lors de la conversion interne LightGBM, doublant la RAM requise.
    X_train = train[feature_cols].astype(np.float32)
    gc.collect()

    cat_features = [c for c in categorical_cols if c in feature_cols]
    model.fit(
        X_train,
        train[TARGET_COL],
        categorical_feature=cat_features,
    )
    return model


def evaluate_model(model: lgb.LGBMClassifier, test: pd.DataFrame, feature_cols: list[str]) -> dict:
    y_true = test[TARGET_COL]
    X_test = test[feature_cols].astype(np.float32)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, digits=3),
    }
    return metrics


def save_model(model: lgb.LGBMClassifier, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str) -> lgb.LGBMClassifier:
    return joblib.load(path)
