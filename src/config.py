"""Configuration settings for PHANTOM // JOBINTEL ML Backend & Frontend."""

import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
ASSETS_DIR = PROJECT_ROOT / "assets"
SRC_DIR = PROJECT_ROOT / "src"

# Matplotlib cache configuration
os.environ["MPLCONFIGDIR"] = str(REPORTS_DIR / ".mpl_cache")

# Data file candidates
RAW_DATA_FILE_CANDIDATES = [
    DATA_DIR / "fake_job_postings.csv",
    PROJECT_ROOT / "dataset" / "fake_job_postings.csv",
]

def get_raw_data_path() -> Path:
    """Return the first existing raw data file path."""
    for path in RAW_DATA_FILE_CANDIDATES:
        if path.exists():
            return path
    # Default to data/fake_job_postings.csv if neither exists yet
    return DATA_DIR / "fake_job_postings.csv"

# Saved Model & Artifact Paths
BEST_MODEL_PATH = MODELS_DIR / "best_model_pipeline.joblib"
MODEL_METRICS_PATH = REPORTS_DIR / "model_metrics.json"
MODEL_COMPARISON_CSV = REPORTS_DIR / "model_comparison.csv"
DATASET_SUMMARY_PATH = REPORTS_DIR / "dataset_summary.json"

# Column Definitions
TARGET_COLUMN = "fraudulent"

TEXT_COLUMNS = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits",
]

METADATA_COLUMNS = [
    "location",
    "department",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
    "telecommuting",
    "has_company_logo",
    "has_questions",
]

# All available feature columns
ALL_FEATURE_COLUMNS = TEXT_COLUMNS + METADATA_COLUMNS

# Modeling Parameters
RANDOM_SEED = 42
TEST_SIZE = 0.20
MAX_FEATURES_TFIDF = 10000
NGRAM_RANGE = (1, 2)
MIN_DF = 2

# Cybersecurity Theme Palette
THEME = {
    "obsidian_black": "#05080D",
    "deep_navy": "#091321",
    "elevated_panel": "#101B2A",
    "neon_cyan": "#00E5FF",
    "electric_blue": "#328BFF",
    "emerald": "#00F5A0",
    "muted_mint": "#9CF8D2",
    "warning_amber": "#FFBF47",
    "threat_red": "#FF4D6D",
    "text_primary": "#EAF2FF",
    "text_secondary": "#8DA1B8",
    "card_border": "#00E5FF",
    "danger_border": "#FF4D6D",
    "success_border": "#00F5A0",
}
