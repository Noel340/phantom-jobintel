"""Model evaluation metrics calculation and visualization generation."""

from pathlib import Path
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)

from src.config import REPORTS_DIR, THEME

# Matplotlib dark cybersecurity style settings
plt.style.use("dark_background")


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    model_name: str = "Model",
) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics for binary classification."""
    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Fraudulent class (class 1) specific metrics
    prec_fraud = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec_fraud = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1_fraud = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

    # Legitimate class (class 0) specific metrics
    prec_legit = float(precision_score(y_true, y_pred, pos_label=0, zero_division=0))
    rec_legit = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
    f1_legit = float(f1_score(y_true, y_pred, pos_label=0, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    roc_auc = None
    pr_auc = None
    if y_prob is not None:
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
            pr_auc = float(average_precision_score(y_true, y_prob))
        except Exception:
            pass

    return {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "macro_precision": round(prec_macro, 4),
        "macro_recall": round(rec_macro, 4),
        "macro_f1": round(f1_macro, 4),
        "weighted_f1": round(f1_weighted, 4),
        "fraud_precision": round(prec_fraud, 4),
        "fraud_recall": round(rec_fraud, 4),
        "fraud_f1": round(f1_fraud, 4),
        "legit_precision": round(prec_legit, 4),
        "legit_recall": round(rec_legit, 4),
        "legit_f1": round(f1_legit, 4),
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "matrix": [[tn, fp], [fn, tp]],
        },
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
        "classification_report": report_dict,
    }


def plot_confusion_matrix(
    cm_matrix: list,
    model_name: str,
    output_path: Path,
) -> Path:
    """Plot and save confusion matrix with cybersecurity styling."""
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=THEME["obsidian_black"])
    ax.set_facecolor(THEME["deep_navy"])

    cmap = sns.color_palette("mako", as_cmap=True)
    sns.heatmap(
        cm_matrix,
        annot=True,
        fmt="d",
        cmap=cmap,
        cbar=False,
        xticklabels=["Legitimate", "Fraudulent"],
        yticklabels=["Legitimate", "Fraudulent"],
        ax=ax,
        annot_kws={"size": 14, "weight": "bold", "color": THEME["text_primary"]},
    )

    ax.set_title(f"Confusion Matrix — {model_name}", color=THEME["neon_cyan"], fontsize=12, pad=12)
    ax.set_xlabel("Predicted Label", color=THEME["text_secondary"], fontsize=10)
    ax.set_ylabel("True Label", color=THEME["text_secondary"], fontsize=10)
    ax.tick_params(colors=THEME["text_secondary"])

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def plot_model_comparison(
    comparison_df,
    output_path: Path,
) -> Path:
    """Plot multi-metric comparison bar chart across models."""
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=THEME["obsidian_black"])
    ax.set_facecolor(THEME["deep_navy"])

    metrics_to_plot = ["fraud_recall", "fraud_precision", "fraud_f1", "macro_f1"]
    metric_labels = ["Fraud Recall", "Fraud Precision", "Fraud F1", "Macro F1"]

    x = np.arange(len(comparison_df))
    width = 0.18

    colors = [THEME["neon_cyan"], THEME["electric_blue"], THEME["emerald"], THEME["warning_amber"]]

    for i, (metric, label, color) in enumerate(zip(metrics_to_plot, metric_labels, colors)):
        vals = comparison_df[metric].values
        rects = ax.bar(x + (i - 1.5) * width, vals, width, label=label, color=color, alpha=0.9)
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color=THEME["text_primary"],
            )

    ax.set_title("Model Performance Comparison (Target: Fraudulent Postings)", color=THEME["neon_cyan"], fontsize=13, pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df["model_name"], color=THEME["text_primary"], fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score (0.0 – 1.0)", color=THEME["text_secondary"], fontsize=10)
    ax.legend(facecolor=THEME["elevated_panel"], edgecolor=THEME["card_border"], labelcolor=THEME["text_primary"])
    ax.grid(axis="y", linestyle="--", alpha=0.2, color="#ffffff")
    ax.tick_params(colors=THEME["text_secondary"])

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path
