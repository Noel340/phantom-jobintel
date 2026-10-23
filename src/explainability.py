"""Explainability and feature importance analysis for PHANTOM // JOBINTEL."""

from typing import List, Dict, Any, Tuple
import numpy as np


def extract_top_global_features(
    pipeline,
    top_n: int = 20,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract top globally indicative features for fraudulent vs legitimate classes
    from a trained linear model pipeline.
    """
    vectorizer = pipeline.named_steps.get("tfidf")
    classifier = pipeline.named_steps.get("classifier")

    if not vectorizer or not classifier:
        return {"fraud_features": [], "legit_features": []}

    feature_names = np.array(vectorizer.get_feature_names_out())

    # Handle calibrated classifier vs direct linear classifier
    coef = None
    if hasattr(classifier, "coef_"):
        coef = classifier.coef_[0]
    elif hasattr(classifier, "calibrated_classifiers_"):
        # Average coefficients across calibrated folds if linear base estimator
        coefs = []
        for clf in classifier.calibrated_classifiers_:
            base = getattr(clf, "estimator", getattr(clf, "base_estimator", None))
            if base is not None and hasattr(base, "coef_"):
                coefs.append(base.coef_[0])
        if coefs:
            coef = np.mean(coefs, axis=0)

    if coef is None:
        return {"fraud_features": [], "legit_features": []}

    # Top fraudulent features (highest positive coefficients)
    top_fraud_idx = np.argsort(coef)[-top_n:][::-1]
    fraud_features = [
        {"feature": str(feature_names[i]), "weight": round(float(coef[i]), 4)}
        for i in top_fraud_idx
    ]

    # Top legitimate features (most negative coefficients)
    top_legit_idx = np.argsort(coef)[:top_n]
    legit_features = [
        {"feature": str(feature_names[i]), "weight": round(float(abs(coef[i])), 4)}
        for i in top_legit_idx
    ]

    return {
        "fraud_features": fraud_features,
        "legit_features": legit_features,
    }


def explain_instance_prediction(
    pipeline,
    raw_text: str,
    top_n: int = 8,
) -> Dict[str, Any]:
    """
    Compute token-level contribution for a specific input advertisement.
    Calculates TF-IDF value multiplied by linear coefficient for each word present.
    """
    vectorizer = pipeline.named_steps.get("tfidf")
    classifier = pipeline.named_steps.get("classifier")

    if not vectorizer or not classifier:
        return {"suspicious_tokens": [], "legitimate_tokens": []}

    # Extract coefficients
    coef = None
    if hasattr(classifier, "coef_"):
        coef = classifier.coef_[0]
    elif hasattr(classifier, "calibrated_classifiers_"):
        coefs = []
        for clf in classifier.calibrated_classifiers_:
            base = getattr(clf, "estimator", getattr(clf, "base_estimator", None))
            if base is not None and hasattr(base, "coef_"):
                coefs.append(base.coef_[0])
        if coefs:
            coef = np.mean(coefs, axis=0)

    if coef is None:
        return {"suspicious_tokens": [], "legitimate_tokens": []}

    feature_names = np.array(vectorizer.get_feature_names_out())
    tfidf_vec = vectorizer.transform([raw_text]).tocoo()

    contributions = []
    for idx, val in zip(tfidf_vec.col, tfidf_vec.data):
        score = val * coef[idx]
        contributions.append((feature_names[idx], score, val))

    # Sort descending by contribution score
    contributions.sort(key=lambda x: x[1], reverse=True)

    suspicious_tokens = [
        {"word": token, "impact": round(float(score), 4), "tfidf": round(float(tfidf), 3)}
        for token, score, tfidf in contributions
        if score > 0.05
    ][:top_n]

    legitimate_tokens = [
        {"word": token, "impact": round(float(abs(score)), 4), "tfidf": round(float(tfidf), 3)}
        for token, score, tfidf in reversed(contributions)
        if score < -0.05
    ][:top_n]

    return {
        "suspicious_tokens": suspicious_tokens,
        "legitimate_tokens": legitimate_tokens,
    }
