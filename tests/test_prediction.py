"""Unit tests for prediction rules, heuristics, and inference interface."""

import pytest
from src.predict import check_suspicious_indicators


def test_telegram_interview_indicator():
    text = "Kindly contact our hiring manager on Telegram @recruiter_fast to conduct the interview."
    indicators = check_suspicious_indicators(text)
    assert any(ind["id"] == "OFF_PLATFORM_CHAT" for ind in indicators)


def test_upfront_fee_check_indicator():
    text = "We will send you a cashier's check to buy your equipment before your first day."
    indicators = check_suspicious_indicators(text)
    assert any(ind["id"] == "UPFRONT_FEE_OR_EQUIPMENT" for ind in indicators)


def test_free_email_indicator():
    text = "Please submit your resume directly to hr.talent.acquisitions@gmail.com for fast review."
    indicators = check_suspicious_indicators(text)
    assert any(ind["id"] == "FREE_EMAIL_CONTACT" for ind in indicators)


def test_crypto_scam_indicator():
    text = "Responsibilities include handling cryptocurrency transfers and bitcoin wallet operations."
    indicators = check_suspicious_indicators(text)
    assert any(ind["id"] == "CRYPTOCURRENCY_OR_INVESTMENT" for ind in indicators)


def test_clean_job_ad_has_no_suspicious_indicators():
    text = """
    We are seeking a Backend Software Engineer with 3+ years experience in Python,
    Docker, and PostgreSQL. Apply through our corporate portal at company careers page.
    """
    indicators = check_suspicious_indicators(text)
    assert len(indicators) == 0
