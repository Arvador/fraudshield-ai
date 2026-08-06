import numpy as np
import pandas as pd

from src.pipeline import (
    add_identity_flag,
    decode_value,
    downcast_chunk,
    encode_categoricals,
    get_column_types,
)


def test_downcast_chunk_converts_float64_to_float32():
    df = pd.DataFrame({"a": np.array([1.0, 2.0], dtype="float64")})
    result = downcast_chunk(df)
    assert result["a"].dtype == np.float32


def test_downcast_chunk_downcasts_int64():
    df = pd.DataFrame({"a": np.array([1, 2, 3], dtype="int64")})
    result = downcast_chunk(df)
    assert result["a"].dtype != np.int64
    assert np.issubdtype(result["a"].dtype, np.integer)


def test_get_column_types_splits_known_prefixes():
    df = pd.DataFrame({
        "TransactionID": [1, 2],
        "TransactionDT": [100, 200],
        "isFraud": [0, 1],
        "ProductCD": ["W", "C"],
        "card1": [111, 222],
        "M1": ["T", "F"],
        "id_01": [1.0, 2.0],
        "TransactionAmt": [50.0, 75.0],
        "V1": [0.1, 0.2],
    })
    categorical_cols, numerical_cols = get_column_types(df)

    assert set(categorical_cols) == {"ProductCD", "card1", "M1", "id_01"}
    assert set(numerical_cols) == {"TransactionAmt", "V1"}


def test_encode_categoricals_returns_codes_and_mapping():
    df = pd.DataFrame({"ProductCD": ["W", "C", "W", None]})
    result, mappings = encode_categoricals(df, ["ProductCD"])

    assert "ProductCD" in mappings
    assert result["ProductCD"].dtype == np.int32
    # La valeur manquante recoit toujours le code -1 (convention pandas cat.codes)
    assert result.loc[3, "ProductCD"] == -1


def test_decode_value_roundtrip():
    df = pd.DataFrame({"ProductCD": ["W", "C", "W"]})
    result, mappings = encode_categoricals(df, ["ProductCD"])
    code_for_w = result.loc[0, "ProductCD"]

    assert decode_value(mappings, "ProductCD", code_for_w) == "W"


def test_decode_value_handles_missing_code():
    mappings = {"ProductCD": ["C", "W"]}
    assert decode_value(mappings, "ProductCD", -1) == "Inconnu"


def test_decode_value_handles_unknown_column():
    mappings = {"ProductCD": ["C", "W"]}
    assert decode_value(mappings, "colonne_inexistante", 5) == "5"


def test_add_identity_flag_marks_non_null_rows():
    df = pd.DataFrame({"DeviceType": ["mobile", None, "desktop"]})
    result = add_identity_flag(df)

    assert result["has_identity_data"].tolist() == [1, 0, 1]
