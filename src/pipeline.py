# Module 1 - Pipeline ETL & Preprocessing
import gc
import os

import joblib
import numpy as np
import pandas as pd

CHUNK_SIZE = 50_000

ID_COLS = ["TransactionID", "TransactionDT"]
TARGET_COL = "isFraud"
CATEGORICAL_PREFIXES = ("card", "M", "id_")
CATEGORICAL_EXPLICIT = [
    "ProductCD", "addr1", "addr2", "P_emaildomain", "R_emaildomain",
    "DeviceType", "DeviceInfo",
]


def downcast_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    float_cols = chunk.select_dtypes(include=["float64"]).columns
    chunk[float_cols] = chunk[float_cols].astype(np.float32)

    int_cols = chunk.select_dtypes(include=["int64"]).columns
    for col in int_cols:
        chunk[col] = pd.to_numeric(chunk[col], downcast="integer")

    return chunk


def load_csv_chunked(path: str, chunksize: int = CHUNK_SIZE) -> pd.DataFrame:
    chunks = []
    for chunk in pd.read_csv(path, chunksize=chunksize):
        chunks.append(downcast_chunk(chunk))

    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()
    return df


def get_column_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical_cols = [
        c for c in df.columns
        if c in CATEGORICAL_EXPLICIT or c.startswith(CATEGORICAL_PREFIXES)
    ]
    numerical_cols = [
        c for c in df.columns
        if c not in categorical_cols and c not in ID_COLS and c != TARGET_COL
    ]
    return categorical_cols, numerical_cols


def encode_categoricals(df: pd.DataFrame, categorical_cols: list[str]) -> tuple[pd.DataFrame, dict]:
    mappings = {}
    for col in categorical_cols:
        cat = df[col].astype("category")
        mappings[col] = list(cat.cat.categories)
        df[col] = cat.cat.codes.astype("int32")
    return df, mappings


def decode_value(mappings: dict, col: str, code) -> str:
    categories = mappings.get(col)
    if categories is None:
        return str(code)
    code = int(code)
    if code < 0 or code >= len(categories):
        return "Inconnu"
    return str(categories[code])


def add_identity_flag(df: pd.DataFrame, identity_marker_col: str = "DeviceType") -> pd.DataFrame:
    df["has_identity_data"] = df[identity_marker_col].notna().astype("int8")
    return df


def load_and_prepare_data(
    transaction_path: str,
    identity_path: str,
    encode: bool = True,
) -> tuple[pd.DataFrame, dict]:
    df_transaction = load_csv_chunked(transaction_path)
    df_identity = load_csv_chunked(identity_path)

    df = df_transaction.merge(df_identity, on="TransactionID", how="left")
    del df_transaction, df_identity
    gc.collect()

    df = add_identity_flag(df)

    mappings = {}
    if encode:
        categorical_cols, _ = get_column_types(df)
        df, mappings = encode_categoricals(df, categorical_cols)

    return df, mappings


def save_processed(df: pd.DataFrame, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)


def save_mappings(mappings: dict, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(mappings, output_path)


def load_mappings(path: str) -> dict:
    return joblib.load(path)
