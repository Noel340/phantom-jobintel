"""Unit tests for URL extraction logic and HTML parsing."""

import pytest
from bs4 import BeautifulSoup
from src.url_analyzer import extract_job_from_json_ld, extract_job_from_html, fetch_and_extract_job


MOCK_JOB_POSTING_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Cybersecurity Analyst - Phantom Defense Systems</title>
    <script type="application/ld+json">
    {
      "@context": "https://schema.org/",
      "@type": "JobPosting",
      "title": "Lead SOC Analyst",
      "description": "<p>Monitor enterprise SIEM, conduct threat hunting, and investigate security incidents.</p>",
      "hiringOrganization": {
        "@type": "Organization",
        "name": "Phantom CyberSec Inc."
      },
      "jobLocation": {
        "@type": "Place",
        "address": {
          "addressLocality": "Austin",
          "addressRegion": "TX",
          "addressCountry": "US"
        }
      },
      "employmentType": "FULL_TIME",
      "responsibilities": "Incident response and packet inspection."
    }
    </script>
</head>
<body>
    <h1>Lead SOC Analyst</h1>
</body>
</html>
"""

MOCK_NON_JSON_LD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>DevOps Engineer - CloudTech</title>
</head>
<body>
    <h1 class="job-title">Senior DevOps Architect</h1>
    <div class="company-name">CloudTech Systems</div>
    <div class="job-description">
        We are seeking a senior architect with Kubernetes, Terraform, and AWS CI/CD pipeline expertise.
    </div>
    <div class="requirements">
        <h3>Requirements</h3>
        <p>5+ years managing production Kubernetes clusters.</p>
    </div>
</body>
</html>
"""


def test_extract_from_json_ld():
    soup = BeautifulSoup(MOCK_JOB_POSTING_HTML, "html.parser")
    result = extract_job_from_json_ld(soup)
    assert result["title"] == "Lead SOC Analyst"
    assert result["company"] == "Phantom CyberSec Inc."
    assert "Monitor enterprise SIEM" in result["description"]
    assert "Austin" in result["location"]


def test_extract_from_semantic_html():
    soup = BeautifulSoup(MOCK_NON_JSON_LD_HTML, "html.parser")
    result = extract_job_from_html(soup)
    assert "Senior DevOps Architect" in result["title"]
    assert "CloudTech Systems" in result["company"]
    assert "Kubernetes" in result["description"]
    assert "5+ years managing production" in result["requirements"]


def test_fetch_and_extract_job_rejects_ssrf():
    result = fetch_and_extract_job("http://169.254.169.254/latest/meta-data/")
    assert result["success"] is False
    assert "Security restriction" in result["error"]
