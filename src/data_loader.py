"""Data loading, schema verification, and dataset statistics for PHANTOM // JOBINTEL."""

import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd

from src.config import (
    get_raw_data_path,
    TARGET_COLUMN,
    TEXT_COLUMNS,
    METADATA_COLUMNS,
    DATASET_SUMMARY_PATH,
)


def load_raw_data(file_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load raw job posting dataset from CSV.
    
    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If the required target column is missing.
    """
    path = file_path if file_path is not None else get_raw_data_path()
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at expected path: {path}")

    # Load with low_memory=False to ensure consistent dtype parsing
    df = pd.read_csv(path, low_memory=False)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Expected target column '{TARGET_COLUMN}' not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )

    # Ensure target is integer
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").fillna(0).astype(int)

    return df


def inspect_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute comprehensive, strictly verified dataset metrics and summaries.
    """
    total_rows, total_cols = df.shape
    class_counts = df[TARGET_COLUMN].value_counts().to_dict()
    fraud_count = int(class_counts.get(1, 0))
    legit_count = int(class_counts.get(0, 0))
    fraud_pct = (fraud_count / total_rows * 100) if total_rows > 0 else 0.0
    legit_pct = (legit_count / total_rows * 100) if total_rows > 0 else 0.0

    # Missing values
    missing_counts = df.isnull().sum().to_dict()
    missing_pcts = {
        col: round(float(count / total_rows * 100), 2)
        for col, count in missing_counts.items()
    }

    # Duplicates
    exact_duplicates = int(df.duplicated().sum())
    # Duplicates ignoring job_id if present
    content_cols = [c for c in df.columns if c != "job_id"]
    content_duplicates = int(df.duplicated(subset=content_cols).sum())

    summary = {
        "total_rows": int(total_rows),
        "total_columns": int(total_cols),
        "columns": list(df.columns),
        "class_distribution": {
            "legitimate_0": legit_count,
            "fraudulent_1": fraud_count,
            "legitimate_pct": round(legit_pct, 2),
            "fraudulent_pct": round(fraud_pct, 2),
            "imbalance_ratio": round(legit_count / fraud_count, 2) if fraud_count > 0 else None,
        },
        "exact_duplicates": exact_duplicates,
        "content_duplicates": content_duplicates,
        "missing_counts": missing_counts,
        "missing_percentages": missing_pcts,
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }

    return summary


def save_dataset_summary(summary: Dict[str, Any], output_path: Optional[Path] = None) -> Path:
    """Save dataset statistics summary to JSON."""
    out = output_path if output_path is not None else DATASET_SUMMARY_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return out


def prepare_combined_text_series(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Combine text and contextual features into a clean single text series for TF-IDF vectorization.
    Keeps training and inference transformations strictly compatible.
    """
    def build_job_text(row: pd.Series) -> str:
        parts = []
        # Title
        t = str(row.get("title", "")).strip()
        if t and t.lower() != "nan":
            parts.append(f"Title: {t}")

        # Company profile
        cp = str(row.get("company_profile", "")).strip()
        if cp and cp.lower() != "nan":
            parts.append(f"Company Profile: {cp}")

        # Description
        desc = str(row.get("description", "")).strip()
        if desc and desc.lower() != "nan":
            parts.append(f"Description: {desc}")

        # Requirements
        req = str(row.get("requirements", "")).strip()
        if req and req.lower() != "nan":
            parts.append(f"Requirements: {req}")

        # Benefits
        ben = str(row.get("benefits", "")).strip()
        if ben and ben.lower() != "nan":
            parts.append(f"Benefits: {ben}")

        # Contextual metadata
        loc = str(row.get("location", "")).strip()
        if loc and loc.lower() != "nan":
            parts.append(f"Location: {loc}")

        emp = str(row.get("employment_type", "")).strip()
        if emp and emp.lower() != "nan":
            parts.append(f"Employment: {emp}")

        exp = str(row.get("required_experience", "")).strip()
        if exp and exp.lower() != "nan":
            parts.append(f"Experience: {exp}")

        edu = str(row.get("required_education", "")).strip()
        if edu and edu.lower() != "nan":
            parts.append(f"Education: {edu}")

        ind = str(row.get("industry", "")).strip()
        if ind and ind.lower() != "nan":
            parts.append(f"Industry: {ind}")

        func = str(row.get("function", "")).strip()
        if func and func.lower() != "nan":
            parts.append(f"Function: {func}")

        return " \n ".join(parts)

    X_text = df.apply(build_job_text, axis=1)
    y = df[TARGET_COLUMN]
    return X_text, y
