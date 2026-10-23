"""Secure Job URL Analysis and Webpage Extraction Module."""

import json
import re
from typing import Dict, Any, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from src.security import (
    validate_url,
    sanitize_text,
    MAX_RESPONSE_SIZE_BYTES,
    ALLOWED_CONTENT_TYPES,
)

USER_AGENT = "PHANTOM-JobIntel/1.0 (Cybersecurity Research; Job Post Analyzer)"
REQUEST_TIMEOUT = (5.0, 10.0)  # (connect_timeout, read_timeout)
MAX_REDIRECTS = 3


def extract_job_from_json_ld(soup: BeautifulSoup) -> Dict[str, Any]:
    """Extract job details from schema.org JobPosting JSON-LD scripts."""
    extracted = {
        "title": "",
        "company": "",
        "description": "",
        "requirements": "",
        "benefits": "",
        "location": "",
        "employment_type": "",
    }

    for script in soup.find_all("script", type="application/ld+json"):
        try:
            if not script.string:
                continue
            data = json.loads(script.string.strip())
            
            # Handle list of items or single item
            items = data if isinstance(data, list) else [data]
            for item in items:
                if not isinstance(item, dict):
                    continue
                # Check graph wrapper
                if "@graph" in item and isinstance(item["@graph"], list):
                    items.extend(item["@graph"])
                    continue

                if item.get("@type") == "JobPosting":
                    extracted["title"] = item.get("title", "")
                    
                    # Company
                    hiring_org = item.get("hiringOrganization")
                    if isinstance(hiring_org, dict):
                        extracted["company"] = hiring_org.get("name", "")
                    elif isinstance(hiring_org, str):
                        extracted["company"] = hiring_org
                    
                    # Description
                    desc_html = item.get("description", "")
                    if desc_html:
                        desc_soup = BeautifulSoup(desc_html, "html.parser")
                        extracted["description"] = desc_soup.get_text(separator=" ", strip=True)
                    
                    # Location
                    loc = item.get("jobLocation")
                    if isinstance(loc, dict):
                        addr = loc.get("address")
                        if isinstance(addr, dict):
                            extracted["location"] = f"{addr.get('addressLocality', '')}, {addr.get('addressRegion', '')}, {addr.get('addressCountry', '')}".strip(" ,")
                        elif isinstance(addr, str):
                            extracted["location"] = addr
                    
                    # Employment Type
                    extracted["employment_type"] = item.get("employmentType", "")
                    
                    # Responsibilities / Requirements
                    resp = item.get("responsibilities", "") or item.get("skills", "")
                    if isinstance(resp, list):
                        extracted["requirements"] = " ".join(resp)
                    elif isinstance(resp, str):
                        extracted["requirements"] = resp

                    return extracted
        except Exception:
            continue

    return extracted


def extract_job_from_html(soup: BeautifulSoup) -> Dict[str, Any]:
    """Extract job posting details using heuristic DOM semantic queries."""
    extracted = {
        "title": "",
        "company": "",
        "description": "",
        "requirements": "",
        "benefits": "",
        "location": "",
        "employment_type": "",
    }

    # 1. Title Extraction
    title_el = (
        soup.find("h1", class_=re.compile(r"title|heading|job-name", re.I))
        or soup.find("h1")
        or soup.find("meta", property="og:title")
        or soup.find("title")
    )
    if title_el:
        if title_el.name == "meta":
            extracted["title"] = title_el.get("content", "")
        else:
            extracted["title"] = title_el.get_text(strip=True)

    # 2. Company Extraction
    company_el = (
        soup.find(class_=re.compile(r"company|employer|hiring-org", re.I))
        or soup.find("meta", property="og:site_name")
    )
    if company_el:
        if company_el.name == "meta":
            extracted["company"] = company_el.get("content", "")
        else:
            extracted["company"] = company_el.get_text(strip=True)

    # 3. Description Extraction
    desc_el = (
        soup.find(id=re.compile(r"job-description|jobDescription|description", re.I))
        or soup.find(class_=re.compile(r"job-description|jobDescription|details-content|description", re.I))
        or soup.find("article")
        or soup.find("main")
    )
    if desc_el:
        # Remove noisy tags
        for tag in desc_el.find_all(["script", "style", "nav", "footer", "header", "form", "button"]):
            tag.decompose()
        extracted["description"] = desc_el.get_text(separator=" ", strip=True)
    else:
        # Fallback to meta description
        meta_desc = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", property="og:description")
        if meta_desc and meta_desc.get("content"):
            extracted["description"] = meta_desc["content"].strip()

    # 4. Requirements & Benefits sections if present
    for section in soup.find_all(["div", "section"]):
        heading = section.find(["h2", "h3", "h4"])
        if heading:
            htext = heading.get_text(strip=True).lower()
            if "requirement" in htext or "qualification" in htext or "what you'll need" in htext:
                extracted["requirements"] = section.get_text(separator=" ", strip=True)
            elif "benefit" in htext or "perk" in htext or "what we offer" in htext:
                extracted["benefits"] = section.get_text(separator=" ", strip=True)

    return extracted


def fetch_and_extract_job(url: str) -> Dict[str, Any]:
    """
    Fetch public URL safely, apply SSRF defenses, and extract job advertisement contents.
    
    Returns:
        Structured dictionary with extraction results or safe error details.
    """
    # Step 1: Pre-validation of URL
    is_valid, error = validate_url(url)
    if not is_valid:
        return {
            "success": False,
            "error": f"Security restriction: {error}",
            "source_url": url,
            "job_title": "",
            "company_name": "",
            "location": "",
            "description": "",
            "requirements": "",
            "benefits": "",
            "raw_combined_text": "",
        }

    current_url = url
    session = requests.Session()
    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    })

    try:
        # Defensive fetch with redirect validation loop
        redirect_count = 0
        response = None
        while redirect_count <= MAX_REDIRECTS:
            req_valid, req_error = validate_url(current_url)
            if not req_valid:
                return {
                    "success": False,
                    "error": f"Redirect blocked by security validation: {req_error}",
                    "source_url": url,
                    "job_title": "",
                    "company_name": "",
                    "location": "",
                    "description": "",
                    "requirements": "",
                    "benefits": "",
                    "raw_combined_text": "",
                }

            # Stream request to prevent response bomb / memory exhaustion
            resp = session.get(
                current_url,
                timeout=REQUEST_TIMEOUT,
                allow_redirects=False,
                stream=True,
            )

            # Check status for redirects
            if resp.status_code in (301, 302, 303, 307, 308):
                redirect_count += 1
                location = resp.headers.get("Location")
                if not location:
                    break
                current_url = urljoin(current_url, location)
                continue

            response = resp
            break

        if response is None:
            return {
                "success": False,
                "error": "Failed to complete HTTP request or exceeded redirect limit.",
                "source_url": url,
                "job_title": "",
                "company_name": "",
                "location": "",
                "description": "",
                "requirements": "",
                "benefits": "",
                "raw_combined_text": "",
            }

        if response.status_code != 200:
            return {
                "success": False,
                "error": f"Web server returned HTTP {response.status_code}. The page may be private or expired.",
                "source_url": url,
                "job_title": "",
                "company_name": "",
                "location": "",
                "description": "",
                "requirements": "",
                "benefits": "",
                "raw_combined_text": "",
            }

        # Content-Type verification
        content_type = response.headers.get("Content-Type", "").lower()
        if not any(allowed in content_type for allowed in ALLOWED_CONTENT_TYPES):
            return {
                "success": False,
                "error": f"Unsupported Content-Type '{content_type}'. Only HTML job pages can be processed.",
                "source_url": url,
                "job_title": "",
                "company_name": "",
                "location": "",
                "description": "",
                "requirements": "",
                "benefits": "",
                "raw_combined_text": "",
            }

        # Read response content safely up to MAX_RESPONSE_SIZE_BYTES
        content_bytes = bytearray()
        for chunk in response.iter_content(chunk_size=8192):
            content_bytes.extend(chunk)
            if len(content_bytes) > MAX_RESPONSE_SIZE_BYTES:
                return {
                    "success": False,
                    "error": "Webpage exceeded the maximum permitted size (2 MB).",
                    "source_url": url,
                    "job_title": "",
                    "company_name": "",
                    "location": "",
                    "description": "",
                    "requirements": "",
                    "benefits": "",
                    "raw_combined_text": "",
                }

        html_text = content_bytes.decode(response.encoding or "utf-8", errors="replace")

    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Connection timed out while attempting to reach target server.",
            "source_url": url,
            "job_title": "",
            "company_name": "",
            "location": "",
            "description": "",
            "requirements": "",
            "benefits": "",
            "raw_combined_text": "",
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": f"Failed to retrieve webpage: {str(e)}",
            "source_url": url,
            "job_title": "",
            "company_name": "",
            "location": "",
            "description": "",
            "requirements": "",
            "benefits": "",
            "raw_combined_text": "",
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error during network extraction: {str(e)}",
            "source_url": url,
            "job_title": "",
            "company_name": "",
            "location": "",
            "description": "",
            "requirements": "",
            "benefits": "",
            "raw_combined_text": "",
        }

    # Step 2: HTML Parsing
    soup = BeautifulSoup(html_text, "html.parser")

    # Try JSON-LD first (structured schema.org)
    data = extract_job_from_json_ld(soup)
    method_used = "json_ld"

    # If description or title missing, fall back to semantic DOM
    if not data["title"] or not data["description"]:
        html_data = extract_job_from_html(soup)
        for key in data:
            if not data[key] and html_data.get(key):
                data[key] = html_data[key]
        method_used = "semantic_html"

    # Sanitize extracted strings
    title = sanitize_text(data.get("title", ""))
    company = sanitize_text(data.get("company", ""))
    location = sanitize_text(data.get("location", ""))
    desc = sanitize_text(data.get("description", ""))
    reqs = sanitize_text(data.get("requirements", ""))
    benefits = sanitize_text(data.get("benefits", ""))

    if not title and not desc:
        return {
            "success": False,
            "error": "Could not identify job advertisement text on this page. The site may require JavaScript or authentication. Please copy and paste the job details directly.",
            "source_url": url,
            "job_title": "",
            "company_name": "",
            "location": "",
            "description": "",
            "requirements": "",
            "benefits": "",
            "raw_combined_text": "",
        }

    # Build combined text for ML pipeline
    combined_parts = [
        f"Title: {title}" if title else "",
        f"Company: {company}" if company else "",
        f"Location: {location}" if location else "",
        f"Description: {desc}" if desc else "",
        f"Requirements: {reqs}" if reqs else "",
        f"Benefits: {benefits}" if benefits else "",
    ]
    raw_combined = " \n ".join(p for p in combined_parts if p)

    return {
        "success": True,
        "error": None,
        "source_url": url,
        "extraction_method": method_used,
        "job_title": title,
        "company_name": company,
        "location": location,
        "description": desc,
        "requirements": reqs,
        "benefits": benefits,
        "raw_combined_text": raw_combined,
    }
