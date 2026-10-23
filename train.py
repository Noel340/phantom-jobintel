"""Root entrypoint to execute PHANTOM // JOBINTEL model training."""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train_model import train_and_evaluate_all_models


def main():
    print("==================================================================")
    print("  PHANTOM // JOBINTEL — MODEL TRAINING & EVALUATION PIPELINE")
    print("==================================================================")
    try:
        results = train_and_evaluate_all_models()
        best = results["best_model"]
        metrics = results["models"][best]
        print("\nSUCCESS: Model training completed.")
        print(f"Selected Champion Model : {best}")
        print(f"Overall Accuracy        : {metrics['accuracy'] * 100:.2f}%")
        print(f"Fraud Class Recall      : {metrics['fraud_recall'] * 100:.2f}%")
        print(f"Fraud Class Precision   : {metrics['fraud_precision'] * 100:.2f}%")
        print(f"Fraud Class F1-Score    : {metrics['fraud_f1'] * 100:.2f}%")
        print(f"Pipeline Artifact       : models/best_model_pipeline.joblib")
        print("==================================================================")
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
