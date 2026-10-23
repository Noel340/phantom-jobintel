"""End-to-end model training, comparison, and serialization pipeline for PHANTOM // JOBINTEL."""

import json
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from src.config import (
    BEST_MODEL_PATH,
    MODEL_METRICS_PATH,
    MODEL_COMPARISON_CSV,
    RANDOM_SEED,
    TEST_SIZE,
    MAX_FEATURES_TFIDF,
    NGRAM_RANGE,
    MIN_DF,
    REPORTS_DIR,
)
from src.data_loader import (
    load_raw_data,
    inspect_dataset,
    save_dataset_summary,
    prepare_combined_text_series,
)
from src.preprocessing import clean_text
from src.evaluate import (
    calculate_metrics,
    plot_confusion_matrix,
    plot_model_comparison,
)


def build_pipeline(classifier, max_features: int = MAX_FEATURES_TFIDF) -> Pipeline:
    """Build a unified scikit-learn Pipeline with TF-IDF vectorization and classifier."""
    vectorizer = TfidfVectorizer(
        preprocessor=clean_text,
        max_features=max_features,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        sublinear_tf=True,
    )
    return Pipeline([
        ("tfidf", vectorizer),
        ("classifier", classifier),
    ])


def run_eda_and_save_visualizations(df: pd.DataFrame, summary: Dict[str, Any]):
    """Generate and save verified exploratory data analysis visualizations."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from src.config import THEME

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Class Distribution Plot
    fig, ax = plt.subplots(figsize=(6, 4.5), facecolor=THEME["obsidian_black"])
    ax.set_facecolor(THEME["deep_navy"])
    counts = [summary["class_distribution"]["legitimate_0"], summary["class_distribution"]["fraudulent_1"]]
    bars = ax.bar(["Legitimate (0)", "Fraudulent (1)"], counts, color=[THEME["emerald"], THEME["threat_red"]], width=0.5)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:,}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4), textcoords="offset points", ha="center", color=THEME["text_primary"], weight="bold")
    ax.set_title("Dataset Target Class Distribution", color=THEME["neon_cyan"], fontsize=12)
    ax.set_ylabel("Count", color=THEME["text_secondary"])
    ax.tick_params(colors=THEME["text_secondary"])
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "eda_class_distribution.png", dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)

    # 2. Missing Values Plot
    missing = {k: v for k, v in summary["missing_percentages"].items() if v > 0}
    if missing:
        sorted_missing = dict(sorted(missing.items(), key=lambda x: x[1], reverse=True))
        fig, ax = plt.subplots(figsize=(8, 5), facecolor=THEME["obsidian_black"])
        ax.set_facecolor(THEME["deep_navy"])
        ax.barh(list(sorted_missing.keys())[::-1], list(sorted_missing.values())[::-1], color=THEME["warning_amber"], height=0.6)
        ax.set_title("Missing Values by Feature (%)", color=THEME["neon_cyan"], fontsize=12)
        ax.set_xlabel("Percentage Missing (%)", color=THEME["text_secondary"])
        ax.tick_params(colors=THEME["text_secondary"])
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "eda_missing_values.png", dpi=200, facecolor=fig.get_facecolor())
        plt.close(fig)

    # 3. Text Length Distribution Plot
    X_text, y = prepare_combined_text_series(df)
    word_lengths_legit = X_text[y == 0].apply(lambda s: len(str(s).split()))
    word_lengths_fraud = X_text[y == 1].apply(lambda s: len(str(s).split()))

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=THEME["obsidian_black"])
    ax.set_facecolor(THEME["deep_navy"])
    sns.kdeplot(word_lengths_legit, ax=ax, color=THEME["emerald"], label="Legitimate", fill=True, alpha=0.3, clip=(0, 1500))
    sns.kdeplot(word_lengths_fraud, ax=ax, color=THEME["threat_red"], label="Fraudulent", fill=True, alpha=0.3, clip=(0, 1500))
    ax.set_title("Advertisement Word Count Distribution", color=THEME["neon_cyan"], fontsize=12)
    ax.set_xlabel("Word Count", color=THEME["text_secondary"])
    ax.set_ylabel("Density", color=THEME["text_secondary"])
    ax.tick_params(colors=THEME["text_secondary"])
    ax.legend(facecolor=THEME["elevated_panel"], edgecolor=THEME["card_border"], labelcolor=THEME["text_primary"])
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "eda_text_length_distribution.png", dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


def train_and_evaluate_all_models() -> Dict[str, Any]:
    """
    Execute full training workflow:
    1. Load real dataset.
    2. Inspect and record actual EDA statistics.
    3. Split train/test (stratified).
    4. Train Logistic Regression, Multinomial Naive Bayes, Linear SVM (calibrated).
    5. Evaluate comprehensively.
    6. Select and serialize best model pipeline.
    """
    print("[1/6] Loading dataset...")
    df = load_raw_data()
    summary = inspect_dataset(df)
    save_dataset_summary(summary)
    print(f"      Loaded {summary['total_rows']} rows across {summary['total_columns']} columns.")
    print(f"      Legitimate: {summary['class_distribution']['legitimate_0']} ({summary['class_distribution']['legitimate_pct']}%)")
    print(f"      Fraudulent: {summary['class_distribution']['fraudulent_1']} ({summary['class_distribution']['fraudulent_pct']}%)")

    print("[2/6] Generating verified EDA reports...")
    run_eda_and_save_visualizations(df, summary)

    print("[3/6] Preprocessing and feature engineering...")
    X_text, y = prepare_combined_text_series(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X_text,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED,
    )
    print(f"      Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    # Candidate Models Definitions
    # 1. Logistic Regression with class_weight='balanced'
    log_reg = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        C=1.0,
        random_state=RANDOM_SEED,
    )

    # 2. Multinomial Naive Bayes (alpha=0.1)
    nb = MultinomialNB(alpha=0.1)

    # 3. Linear SVM with CalibratedClassifierCV to provide true well-calibrated probabilities
    linear_svc = LinearSVC(
        class_weight="balanced",
        dual="auto",
        C=1.0,
        random_state=RANDOM_SEED,
        max_iter=2000,
    )
    calibrated_svm = CalibratedClassifierCV(
        estimator=linear_svc,
        method="sigmoid",
        cv=3,
    )

    candidate_classifiers = {
        "Logistic Regression": log_reg,
        "Multinomial Naive Bayes": nb,
        "Linear SVM (Calibrated)": calibrated_svm,
    }

    all_metrics = {}
    fitted_pipelines = {}
    comparison_rows = []

    print("[4/6] Training and evaluating candidate models...")
    for name, clf in candidate_classifiers.items():
        print(f"      Training {name}...")
        pipeline = build_pipeline(clf)
        pipeline.fit(X_train, y_train)

        # Predictions on held-out test set
        y_pred = pipeline.predict(X_test)
        
        y_prob = None
        if hasattr(pipeline, "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred, y_prob=y_prob, model_name=name)
        all_metrics[name] = metrics
        fitted_pipelines[name] = pipeline

        # Save confusion matrix plot
        clean_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        cm_path = REPORTS_DIR / f"confusion_matrix_{clean_name}.png"
        plot_confusion_matrix(metrics["confusion_matrix"]["matrix"], name, cm_path)

        comparison_rows.append({
            "model_name": name,
            "accuracy": metrics["accuracy"],
            "fraud_recall": metrics["fraud_recall"],
            "fraud_precision": metrics["fraud_precision"],
            "fraud_f1": metrics["fraud_f1"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
        })
        print(f"      -> {name} | Fraud Recall: {metrics['fraud_recall']:.4f} | Fraud Precision: {metrics['fraud_precision']:.4f} | Fraud F1: {metrics['fraud_f1']:.4f}")

    # Build comparison DataFrame
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(MODEL_COMPARISON_CSV, index=False)
    plot_model_comparison(comparison_df, REPORTS_DIR / "model_comparison.png")

    print("[5/6] Model selection based on Fraud F1-Score & Fraud Recall...")
    # Selection criteria: Primary is Fraud F1-score (harmonic balance of catching scams and avoiding excessive false alarms), secondary is PR-AUC/Fraud Recall.
    best_model_name = comparison_df.sort_values(by=["fraud_f1", "fraud_recall"], ascending=False).iloc[0]["model_name"]
    best_pipeline = fitted_pipelines[best_model_name]
    best_metrics = all_metrics[best_model_name]

    print(f"      Selected Winner: '{best_model_name}'")
    print(f"      Fraud F1: {best_metrics['fraud_f1']} | Fraud Recall: {best_metrics['fraud_recall']} | Fraud Precision: {best_metrics['fraud_precision']}")

    # Save Best Model Pipeline
    BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, BEST_MODEL_PATH)
    print(f"      Saved best fitted pipeline to: {BEST_MODEL_PATH}")

    # Save comprehensive metrics JSON
    output_package = {
        "best_model": best_model_name,
        "selection_criterion": "Highest Fraud F1-Score (harmonic mean of Fraud Recall and Fraud Precision on minority fraudulent class) combined with calibrated probability support.",
        "models": all_metrics,
        "dataset_summary": summary,
    }
    with open(MODEL_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(output_package, f, indent=2)

    print("[6/6] Training workflow completed successfully!")
    return output_package


if __name__ == "__main__":
    train_and_evaluate_all_models()
