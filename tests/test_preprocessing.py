"""Unit tests for text cleaning and normalization routines."""

import pytest
from src.preprocessing import clean_text, format_job_input_for_inference


def test_html_tag_removal():
    raw = "<p>Looking for a <strong>Senior Developer</strong>. Apply now!</p>"
    cleaned = clean_text(raw)
    assert "<p>" not in cleaned
    assert "<strong>" not in cleaned
    assert "senior developer" in cleaned


def test_email_and_url_tokenization():
    raw = "Send resumes to recruiter@example.com or visit https://careers.company.com/apply"
    cleaned = clean_text(raw)
    assert "email_token" in cleaned
    assert "recruiter@example.com" not in cleaned
    assert "url_token" in cleaned
    assert "https://careers.company.com" not in cleaned


def test_currency_tokenization():
    raw = "Competitive compensation of $120,000 to $150k per year."
    cleaned = clean_text(raw)
    assert "currency_token" in cleaned


def test_empty_and_none_handling():
    assert clean_text(None) == ""
    assert clean_text("") == ""
    assert clean_text("nan") == ""
    assert clean_text("   ") == ""


def test_format_job_input_for_inference():
    sample = {
        "title": "Data Analyst",
        "company_profile": "Global Analytics Leader",
        "description": "Analyze big data pipelines using SQL and Python.",
        "requirements": "BSc in CS or related field.",
        "benefits": "Health insurance and remote setup.",
    }
    formatted = format_job_input_for_inference(sample)
    assert "Title: Data Analyst" in formatted
    assert "Company Profile: Global Analytics Leader" in formatted
    assert "Description: Analyze big data pipelines" in formatted
    assert "Requirements: BSc in CS" in formatted
    assert "Benefits: Health insurance" in formatted
