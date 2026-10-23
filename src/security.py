"""Security validation and SSRF defenses for PHANTOM // JOBINTEL."""

import ipaddress
import socket
from urllib.parse import urlparse
from typing import Tuple, Optional


class SecurityValidationError(ValueError):
    """Custom exception raised when URL or input violates security policy."""
    pass


# Disallowed special hostnames
BLOCKED_HOSTNAMES = {
    "localhost",
    "127.0.0.1",
    "::1",
    "0.0.0.0",
    "metadata.google.internal",
    "instance-data",
}

# Maximum allowed webpage download size (2 MB)
MAX_RESPONSE_SIZE_BYTES = 2 * 1024 * 1024

# Allowed URL schemes
ALLOWED_SCHEMES = {"http", "https"}

# Allowed Content Types for extraction
ALLOWED_CONTENT_TYPES = [
    "text/html",
    "application/xhtml+xml",
    "text/plain",
]


def is_ip_private_or_reserved(ip_str: str) -> bool:
    """Check if an IP string belongs to private, loopback, link-local, or reserved ranges."""
    try:
        ip = ipaddress.ip_address(ip_str)
        return (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        )
    except ValueError:
        return True


def validate_url(url: str) -> Tuple[bool, Optional[str]]:
    """
    Validate a public URL against SSRF and protocol manipulation attacks.
    
    Returns:
        (is_valid: bool, error_message: Optional[str])
    """
    if not url or not isinstance(url, str):
        return False, "URL cannot be empty."

    stripped_url = url.strip()
    if len(stripped_url) > 2048:
        return False, "URL length exceeds safe limit (2048 characters)."

    try:
        parsed = urlparse(stripped_url)
    except Exception as e:
        return False, f"Malformed URL format: {str(e)}"

    # 1. Scheme check
    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        return False, f"Unsupported protocol '{parsed.scheme}'. Only HTTP and HTTPS are permitted."

    # 2. Hostname presence
    hostname = parsed.hostname
    if not hostname:
        return False, "URL missing valid hostname."

    hostname_lower = hostname.lower()

    # 3. Blocked hostnames check
    if hostname_lower in BLOCKED_HOSTNAMES or hostname_lower.endswith(".local") or hostname_lower.endswith(".internal"):
        return False, f"Access to host '{hostname}' is restricted by security policy."

    # 4. Resolve DNS and check all resolved IP addresses
    try:
        addr_info = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
        if not addr_info:
            return False, f"Could not resolve domain '{hostname}'."

        for item in addr_info:
            sockaddr = item[4]
            ip_str = sockaddr[0]
            if is_ip_private_or_reserved(ip_str):
                return False, f"Target resolved to restricted IP address ({ip_str}). Access denied."

    except socket.gaierror:
        return False, f"Domain name resolution failed for '{hostname}'."
    except Exception as e:
        return False, f"Network validation failed: {str(e)}"

    return True, None


def sanitize_text(text: Optional[str], max_length: int = 100000) -> str:
    """Sanitize user-provided text inputs, stripping harmful control characters."""
    if text is None:
        return ""
    # Strip null bytes and non-printable control chars (except newline/tab/carriage return)
    cleaned = "".join(ch for ch in str(text) if ch in ('\n', '\r', '\t') or (' ' <= ch <= '~') or ord(ch) > 127)
    return cleaned[:max_length].strip()
