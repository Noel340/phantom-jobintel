"""Integration and end-to-end tests for PHANTOM // JOBINTEL core flow."""

import pytest
from src.predict import JobIntelPredictor, check_suspicious_indicators
from src.url_analyzer import extract_job_from_html
from bs4 import BeautifulSoup


def test_predictor_empty_input():
    predictor = JobIntelPredictor()
    result = predictor.predict({"title": "", "description": ""})
    assert result["success"] is False
    assert "too short or empty" in result["error"].lower()


def test_predictor_valid_input():
    predictor = JobIntelPredictor()
    job_payload = {
        "title": "Senior Security Operations Center Analyst",
        "company_profile": "Fortune 500 cybersecurity consulting firm protecting critical infrastructure worldwide.",
        "description": "Lead incident triage, analyze packet captures, conduct malware reverse engineering and configure Splunk SIEM.",
        "requirements": "B.Tech or BS in Computer Science, CISSP or CEH, 4+ years in cyber defense.",
        "benefits": "Competitive compensation, 401(k) matching, comprehensive health benefits.",
        "location": "US, NY, New York",
        "employment_type": "Full-time",
    }
    result = predictor.predict(job_payload)
    assert result["success"] is True
    assert result["prediction"] in (0, 1)
    assert result["verdict"] in ("LIKELY LEGITIMATE", "POTENTIALLY FRAUDULENT")
    assert result["confidence"] is not None
    assert 0.0 <= result["confidence"] <= 1.0
    assert "explanation" in result
    assert "verification_steps" in result


def test_url_extraction_to_prediction_pipeline():
    """Verify that extracted job data flows cleanly into the prediction engine."""
    mock_html = """
    <html>
      <body>
        <h1 class="job-title">Senior Python Architect</h1>
        <div class="company">SecureCode Systems</div>
        <div class="job-description">Design scalable microservices with FastAPI, PostgreSQL, and Kafka.</div>
        <div class="requirements">Requirements: 5+ years backend systems experience.</div>
      </body>
    </html>
    """
    soup = BeautifulSoup(mock_html, "html.parser")
    extracted = extract_job_from_html(soup)
    assert extracted["title"] == "Senior Python Architect"

    predictor = JobIntelPredictor()
    res = predictor.predict(extracted)
    assert res["success"] is True
    assert res["prediction"] in (0, 1)
