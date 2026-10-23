"""Unit tests for dataset loading, inspection, and schema verification."""

import pytest
import pandas as pd
from src.data_loader import load_raw_data, inspect_dataset, prepare_combined_text_series
from src.config import TARGET_COLUMN


def test_load_raw_data():
    df = load_raw_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert TARGET_COLUMN in df.columns
    # Target column must be binary (0 or 1)
    unique_vals = set(df[TARGET_COLUMN].unique())
    assert unique_vals.issubset({0, 1})


def test_inspect_dataset():
    df = load_raw_data()
    summary = inspect_dataset(df)
    assert "total_rows" in summary
    assert summary["total_rows"] > 10000
    assert "class_distribution" in summary
    assert "legitimate_0" in summary["class_distribution"]
    assert "fraudulent_1" in summary["class_distribution"]
    assert summary["class_distribution"]["fraudulent_1"] > 0
    assert summary["class_distribution"]["legitimate_0"] > summary["class_distribution"]["fraudulent_1"]


def test_prepare_combined_text_series():
    df = load_raw_data().head(50)
    X_text, y = prepare_combined_text_series(df)
    assert len(X_text) == 50
    assert len(y) == 50
    assert isinstance(X_text.iloc[0], str)
