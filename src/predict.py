"""Prediction engine, input validation, and indicator checks for PHANTOM // JOBINTEL."""

from pathlib import Path
from typing import Dict, Any, Optional, List
import joblib
import re

from src.config import BEST_MODEL_PATH
from src.preprocessing import clean_text, format_job_input_for_inference
from src.explainability import explain_instance_prediction

# Rule-based suspicious pattern definitions (Independent of ML model)
SUSPICIOUS_HEURISTICS = [
    {
        "id": "FREE_EMAIL_CONTACT",
        "title": "Unprofessional Email Contact",
        "pattern": re.compile(r"\b[A-Za-z0-9._%+-]+@(gmail\.com|yahoo\.com|hotmail\.com|outlook\.com|yopmail\.com|protonmail\.com)\b", re.IGNORECASE),
        "description": "Recruiter uses a free public email address instead of a registered corporate domain.",
        "severity": "HIGH",
    },
    {
        "id": "OFF_PLATFORM_CHAT",
        "title": "Off-Platform / Encrypted Messaging Interview",
        "pattern": re.compile(r"\b(telegram|whatsapp|signal|google\s*hangouts|wire\s*transfer)\b", re.IGNORECASE),
        "description": "Directs applicant to conduct interviews or communication over Telegram or WhatsApp.",
        "severity": "HIGH",
    },
    {
        "id": "UPFRONT_FEE_OR_EQUIPMENT",
        "title": "Payment / Equipment Check Scam Pattern",
        "pattern": re.compile(r"\b(buy\s*(?:your\s*)?(?:own\s*)?equipment|send\s*(?:a\s*)?check|cashier'?s\s*check|wire\s*funds|application\s*fee|registration\s*fee)\b", re.IGNORECASE),
        "description": "Mentions upfront payments, purchasing equipment through specified vendors, or receiving checks.",
        "severity": "CRITICAL",
    },
    {
        "id": "CRYPTOCURRENCY_OR_INVESTMENT",
        "title": "Cryptocurrency / Financial Transfer Scheme",
        "pattern": re.compile(r"\b(crypto|bitcoin|usdt|binance|wallet\s*address|package\s*handling|reshipping)\b", re.IGNORECASE),
        "description": "Job involves cryptocurrency transfers, mysterious parcel forwarding, or high-risk financial movement.",
        "severity": "CRITICAL",
    },
    {
        "id": "URGENT_NO_INTERVIEW",
        "title": "Immediate Hire Without Interview",
        "pattern": re.compile(r"\b(no\s*interview\s*required|start\s*today|immediate\s*placement|guaranteed\s*income|earn\s*\$?\d{3,}\s*(?:a\s*day|per\s*day))\b", re.IGNORECASE),
        "description": "Claims of immediate employment with zero technical/behavioral screening or unrealistic daily pay.",
        "severity": "MEDIUM",
    },
]


def check_suspicious_indicators(raw_text: str) -> List[Dict[str, str]]:
    """
    Perform deterministic heuristic scan for known employment scam red flags.
    These are purely rule-based and strictly decoupled from the ML classifier.
    """
    detected = []
    for heuristic in SUSPICIOUS_HEURISTICS:
        match = heuristic["pattern"].search(raw_text)
        if match:
            detected.append({
                "id": heuristic["id"],
                "title": heuristic["title"],
                "description": heuristic["description"],
                "severity": heuristic["severity"],
                "matched_snippet": match.group(0),
            })
    return detected


class JobIntelPredictor:
    """Production predictor wrapper for PHANTOM // JOBINTEL."""

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path if model_path is not None else BEST_MODEL_PATH
        self.pipeline = None
        self._load_pipeline()

    def _load_pipeline(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained model pipeline not found at '{self.model_path}'. "
                "Please run 'python train.py' to generate the model first."
            )
        self.pipeline = joblib.load(self.model_path)

    def predict(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run inference on job posting data.
        
        Args:
            job_data: Dictionary containing job fields (title, description, etc.) or 'raw_text'.
            
        Returns:
            Structured prediction dictionary.
        """
        if self.pipeline is None:
            self._load_pipeline()

        # Format inputs consistently with training
        combined_text = format_job_input_for_inference(job_data)
        cleaned = clean_text(combined_text)

        if not cleaned or len(cleaned.split()) < 3:
            return {
                "success": False,
                "error": "The provided job posting is too short or empty. Please provide more content for a reliable assessment.",
                "prediction_label": None,
                "confidence": None,
            }

        # Predict class
        pred = int(self.pipeline.predict([cleaned])[0])

        # Predict probability if supported
        fraud_prob = None
        legit_prob = None
        if hasattr(self.pipeline, "predict_proba"):
            probs = self.pipeline.predict_proba([cleaned])[0]
            legit_prob = round(float(probs[0]), 4)
            fraud_prob = round(float(probs[1]), 4)

        # Confidence metric
        confidence = fraud_prob if pred == 1 else legit_prob

        # Heuristic indicator warnings
        indicators = check_suspicious_indicators(combined_text)

        # Feature explanation
        explanation = explain_instance_prediction(self.pipeline, cleaned, top_n=6)

        # Verdict
        if pred == 1:
            verdict = "POTENTIALLY FRAUDULENT"
            verdict_severity = "HIGH"
            verdict_color = "#FF4D6D"
            summary_message = "This advertisement exhibits statistical language patterns frequently associated with fraudulent or deceptive job postings."
        else:
            verdict = "LIKELY LEGITIMATE"
            verdict_severity = "LOW"
            verdict_color = "#00F5A0"
            summary_message = "This advertisement matches stylistic and structural patterns characteristic of legitimate recruitment listings."

        return {
            "success": True,
            "prediction": pred,
            "verdict": verdict,
            "verdict_severity": verdict_severity,
            "verdict_color": verdict_color,
            "confidence": confidence,
            "probability_distribution": {
                "legitimate": legit_prob,
                "fraudulent": fraud_prob,
            },
            "summary_message": summary_message,
            "suspicious_indicators": indicators,
            "explanation": explanation,
            "input_length_words": len(cleaned.split()),
            "input_length_chars": len(cleaned),
            "verification_steps": [
                "Verify the job listing on the employer's official corporate website careers page.",
                "Search the hiring entity in state/national corporate registries (e.g. SEC EDGAR, MCA, Companies House).",
                "Cross-check the recruiter's name and email domain on LinkedIn.",
                "Never send funds, purchase gift cards, or cash advance checks for equipment purchases.",
            ],
            "limitations_disclaimer": "This analysis is an AI-assisted statistical heuristic trained on historical recruitment data. It does not constitute legal, financial, or employment verification. Advanced scams can mirror legitimate language, and genuine listings may trigger warnings if poorly written.",
        }


# Global lazy-loaded instance
_global_predictor: Optional[JobIntelPredictor] = None


def get_predictor() -> JobIntelPredictor:
    """Get or initialize singleton JobIntelPredictor."""
    global _global_predictor
    if _global_predictor is None:
        _global_predictor = JobIntelPredictor()
    return _global_predictor
