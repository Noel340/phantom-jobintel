# PHANTOM // JOBINTEL — Comprehensive File-by-File Guide

This document explains the role, dependencies, and design choices of every directory and file in the **PHANTOM // JOBINTEL** repository.

---

### Root Directory

- **`app.py`**: The primary user interface entry point built with Streamlit. It loads the pre-trained model pipeline once, injects custom dark-mode cybersecurity CSS styles, presents the 5 telemetry views (Command Center, Job Analyzer, Model Intelligence, Dataset Intelligence, and Methodology/Viva), and handles both manual text and URL extraction flows.
- **`train.py`**: The root training script executed via `python train.py`. It calls `train_and_evaluate_all_models()` from `src/train_model.py`, logs progress to stdout, saves the champion pipeline, and prints final evaluation metrics.
- **`requirements.txt`**: Complete list of pinned and verified Python packages necessary to run training, testing, and the Streamlit dashboard (`streamlit`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `joblib`, `beautifulsoup4`, `requests`, `pytest`).
- **`README.md`**: The master repository documentation containing the project abstract, architecture diagram, empirical benchmark table, installation instructions, and Windows PowerShell usage commands.
- **`.gitignore`**: Version control rules excluding Python virtual environments (`.venv`), bytecode caches (`__pycache__`), test caches (`.pytest_cache`), and temporary cache directories.

---

### `src/` Directory (Core Logic & Modules)

- **`src/__init__.py`**: Marks `src/` as a Python package and exports package metadata (`__version__ = "1.0.0"`).
- **`src/config.py`**: Centralized configuration module. Defines project file paths (`PROJECT_ROOT`, `DATA_DIR`, `MODELS_DIR`, `REPORTS_DIR`), raw data search candidates, feature column lists, ML hyperparameters (random seed 42, test split 0.20, max features 10,000), and cybersecurity theme hex color definitions. Also redirects Matplotlib's cache directory to avoid Windows permission issues.
- **`src/security.py`**: Security module implementing defensive networking and Server-Side Request Forgery (SSRF) defenses. Functions include `validate_url()` (validates schemes, inspects hostnames, resolves DNS pre-flight to block RFC 1918 private subnets and cloud metadata endpoints), `is_ip_private_or_reserved()`, and `sanitize_text()`.
- **`src/url_analyzer.py`**: Web scraper module designed specifically for recruitment pages. Implements `fetch_and_extract_job()`, which fetches public pages using streaming with a 2 MB size cap, validates redirects against SSRF, parses structured schema.org `JobPosting` JSON-LD scripts, falls back to semantic HTML containers, and cleans the text.
- **`src/data_loader.py`**: Dataset ingestion and schema validation. Implements `load_raw_data()` to safely load `fake_job_postings.csv`, `inspect_dataset()` to compute verified class distributions and missingness statistics, and `prepare_combined_text_series()` to combine text and metadata columns into a single unified feature.
- **`src/preprocessing.py`**: Leakage-free text cleaning and normalization routines. Implements `clean_text()` (strips HTML, replaces URLs, emails, phone numbers, and currency symbols with standardized tokens, removes punctuation noise, and collapses whitespace) and `format_job_input_for_inference()` to format single job records during inference.
- **`src/train_model.py`**: Full model training and evaluation coordinator. Trains Logistic Regression, Multinomial Naive Bayes, and Calibrated Linear SVM; computes test metrics; generates EDA distribution charts and confusion matrices; selects the champion model based on Fraud F1; and saves artifacts.
- **`src/evaluate.py`**: Performance metric calculation and visualization plotting. Calculates accuracy, macro/weighted F1, fraud-specific precision/recall/F1, confusion matrices, and ROC/PR AUC. Contains `plot_confusion_matrix()` and `plot_model_comparison()`.
- **`src/predict.py`**: Production inference engine. Implements `JobIntelPredictor`, which loads the serialized pipeline, formats input, calls `predict()` and `predict_proba()`, extracts token explainability, triggers `check_suspicious_indicators()` (rule-based scanner for Telegram, upfront checks, free emails), and compiles actionable verification guidance.
- **`src/explainability.py`**: Interpretability module for linear models. Implements `extract_top_global_features()` to extract global fraud vs legitimate vocabulary weights, and `explain_instance_prediction()` to compute token-level contributions for individual job postings.

---

### `tests/` Directory (Automated Test Suite)

- **`tests/test_data_loader.py`**: Tests dataset loading, binary target validation, row count assertions, and text series preparation.
- **`tests/test_preprocessing.py`**: Validates HTML stripping, email/URL/currency tokenization, none/empty string handling, and inference input formatting.
- **`tests/test_prediction.py`**: Validates rule-based heuristic indicator detection (Telegram interviews, check advance scams, free email recruiters, crypto tasks) and verifies clean ads have zero false alarms.
- **`tests/test_security.py`**: Tests SSRF defenses: verifies allowed public URLs, ensures localhost/127.0.0.1/private RFC 1918 IPs are blocked, verifies non-HTTP schemes are rejected, and tests text sanitization.
- **`tests/test_url_analyzer.py`**: Tests JSON-LD parsing, semantic HTML container extraction, and SSRF rejection on cloud metadata endpoints.
- **`tests/test_app.py`**: End-to-end integration tests verifying `JobIntelPredictor` handling of empty inputs, valid job inputs, probability ranges, and handoff of extracted URL data to the predictor.

---

### `data/`, `models/`, and `reports/` Directories

- **`data/fake_job_postings.csv`**: Real dataset of 17,880 recruitment postings downloaded from Kaggle.
- **`models/best_model_pipeline.joblib`**: Serialized champion pipeline containing fitted `TfidfVectorizer` and calibrated Linear SVM.
- **`reports/`**: Complete suite of visual and tabular artifacts generated during training, including `dataset_summary.json`, `model_metrics.json`, `model_comparison.csv`, comparison charts, confusion matrix heatmaps, EDA distribution charts, academic reports, and viva preparation guides.
