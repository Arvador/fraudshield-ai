# Chargements partages entre les pages de l'application Streamlit
import os

import numpy as np
import pandas as pd
import streamlit as st

from src.explainer import get_explainer
from src.models import get_feature_columns, load_model, time_based_split
from src.pipeline import load_mappings

DATA_PATH = os.path.join("data", "processed", "train_clean.parquet")
MODEL_PATH = os.path.join("models", "lightgbm_baseline.pkl")
MAPPINGS_PATH = os.path.join("data", "processed", "category_mappings.pkl")
SAMPLE_SIZE = 500
RANDOM_STATE = 42


@st.cache_resource
def load_model_cached():
    return load_model(MODEL_PATH)


@st.cache_resource
def load_mappings_cached():
    return load_mappings(MAPPINGS_PATH)


@st.cache_resource
def get_explainer_cached(_model):
    return get_explainer(_model)


@st.cache_data
def load_test_with_predictions():
    df = pd.read_parquet(DATA_PATH)
    feature_cols = get_feature_columns(df)
    _, test = time_based_split(df, test_size=0.2)
    test = test.copy()
    del df

    model = load_model_cached()
    X_test = test[feature_cols].astype(np.float32)
    test["proba_fraude"] = model.predict_proba(X_test)[:, 1]
    return test, feature_cols
