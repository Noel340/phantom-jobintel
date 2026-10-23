"""Text cleaning, normalization, and scikit-learn compatible preprocessors."""

import re
from typing import Dict, Any, Union
import html


# Regex patterns for normalization
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
PHONE_PATTERN = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
CURRENCY_PATTERN = re.compile(r"[\$€£¥]\s*\d+(?:[,\.]\d+)*(?:\s*(?:k|thousand|m|million|usd|cad|eur|gbp))?", re.IGNORECASE)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
EXTRA_WHITESPACE_PATTERN = re.compile(r"\s+")
SPECIAL_CHAR_PATTERN = re.compile(r"[^a-zA-Z0-9_\s.,!?\'\"/$€£\-:]")


def clean_text(text: Union[str, float, None]) -> str:
    """
    Apply robust, leakage-free text normalization.
    
    1. Unescape HTML entities.
    2. Remove HTML tags.
    3. Normalize URLs to token '[URL]'.
    4. Normalize email addresses to token '[EMAIL]'.
    5. Normalize phone numbers to token '[PHONE]'.
    6. Normalize currency/salary figures to token '[CURRENCY]'.
    7. Clean unusual control/special characters.
    8. Lowercase and collapse redundant whitespace.
    """
    if text is None:
        return ""
    text_str = str(text)
    if not text_str.strip() or text_str.lower() == "nan":
        return ""

    # 1. Unescape HTML
    text_str = html.unescape(text_str)

    # 2. Strip HTML tags
    text_str = HTML_TAG_PATTERN.sub(" ", text_str)

    # 3. Tokenize sensitive/structural patterns
    text_str = URL_PATTERN.sub(" URL_TOKEN ", text_str)
    text_str = EMAIL_PATTERN.sub(" EMAIL_TOKEN ", text_str)
    text_str = PHONE_PATTERN.sub(" PHONE_TOKEN ", text_str)
    text_str = CURRENCY_PATTERN.sub(" CURRENCY_TOKEN ", text_str)

    # 4. Remove unreadable noise
    text_str = SPECIAL_CHAR_PATTERN.sub(" ", text_str)

    # 5. Lowercase and normalize whitespace
    text_str = text_str.lower()
    text_str = EXTRA_WHITESPACE_PATTERN.sub(" ", text_str).strip()

    return text_str


def format_job_input_for_inference(job_dict: Dict[str, Any]) -> str:
    """
    Convert single job dictionary (from pasted UI form or URL extraction)
    into the exact text format expected by the trained pipeline.
    """
    parts = []

    title = str(job_dict.get("title", "")).strip()
    if title and title.lower() != "nan":
        parts.append(f"Title: {title}")

    company = str(job_dict.get("company_profile", "") or job_dict.get("company_name", "")).strip()
    if company and company.lower() != "nan":
        parts.append(f"Company Profile: {company}")

    desc = str(job_dict.get("description", "")).strip()
    if desc and desc.lower() != "nan":
        parts.append(f"Description: {desc}")

    req = str(job_dict.get("requirements", "")).strip()
    if req and req.lower() != "nan":
        parts.append(f"Requirements: {req}")

    ben = str(job_dict.get("benefits", "")).strip()
    if ben and ben.lower() != "nan":
        parts.append(f"Benefits: {ben}")

    loc = str(job_dict.get("location", "")).strip()
    if loc and loc.lower() != "nan":
        parts.append(f"Location: {loc}")

    emp = str(job_dict.get("employment_type", "")).strip()
    if emp and emp.lower() != "nan":
        parts.append(f"Employment: {emp}")

    exp = str(job_dict.get("required_experience", "") or job_dict.get("experience", "")).strip()
    if exp and exp.lower() != "nan":
        parts.append(f"Experience: {exp}")

    edu = str(job_dict.get("required_education", "") or job_dict.get("education", "")).strip()
    if edu and edu.lower() != "nan":
        parts.append(f"Education: {edu}")

    ind = str(job_dict.get("industry", "")).strip()
    if ind and ind.lower() != "nan":
        parts.append(f"Industry: {ind}")

    func = str(job_dict.get("function", "") or job_dict.get("department", "")).strip()
    if func and func.lower() != "nan":
        parts.append(f"Function: {func}")

    # Fallback if raw text was supplied
    if not parts and job_dict.get("raw_text"):
        parts.append(str(job_dict["raw_text"]).strip())

    return " \n ".join(parts)
