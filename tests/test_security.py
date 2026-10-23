"""Unit tests for security, SSRF defenses, and URL input validation."""

import pytest
from src.security import validate_url, is_ip_private_or_reserved, sanitize_text


def test_allowed_public_url(monkeypatch):
    """Verify that legitimate public HTTP/HTTPS URLs pass validation."""
    # Mock DNS lookup to return a public IP (e.g. 93.184.216.34) so tests run reliably offline
    import socket
    def mock_getaddrinfo(host, port, proto=0):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]
    monkeypatch.setattr(socket, "getaddrinfo", mock_getaddrinfo)

    valid, err = validate_url("https://www.example.com")
    assert valid is True
    assert err is None


def test_block_localhost():
    """Verify localhost is strictly blocked."""
    valid, err = validate_url("http://localhost:8000/job")
    assert valid is False
    assert "restricted" in err.lower() or "denied" in err.lower()


def test_block_loopback_ip():
    """Verify loopback 127.0.0.1 is blocked."""
    valid, err = validate_url("http://127.0.0.1/admin")
    assert valid is False
    assert "restricted" in err.lower() or "denied" in err.lower()


def test_block_private_rfc1918_ips():
    """Verify private subnet IPs are identified and blocked."""
    assert is_ip_private_or_reserved("192.168.1.1") is True
    assert is_ip_private_or_reserved("10.0.0.1") is True
    assert is_ip_private_or_reserved("172.16.0.1") is True
    assert is_ip_private_or_reserved("169.254.169.254") is True  # Cloud metadata
    assert is_ip_private_or_reserved("8.8.8.8") is False  # Public DNS


def test_block_disallowed_schemes():
    """Verify dangerous or non-web schemes are rejected."""
    for scheme_url in ["ftp://example.com/job", "file:///etc/passwd", "gopher://evil.com"]:
        valid, err = validate_url(scheme_url)
        assert valid is False
        assert "unsupported protocol" in err.lower()


def test_empty_and_oversized_url():
    """Verify empty or excessively long URLs are caught."""
    valid, err = validate_url("")
    assert valid is False
    assert "empty" in err.lower()

    valid, err = validate_url("https://example.com/" + ("a" * 2500))
    assert valid is False
    assert "limit" in err.lower()


def test_text_sanitization():
    """Verify control characters and null bytes are stripped."""
    dirty_input = "Senior Engineer\x00\x08 with <script>alert(1)</script> skills."
    cleaned = sanitize_text(dirty_input)
    assert "\x00" not in cleaned
    assert "\x08" not in cleaned
    assert "Senior Engineer" in cleaned
