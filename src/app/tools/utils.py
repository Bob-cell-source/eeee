"""
Common utilities for tools
"""
import re
from urllib.parse import urlparse
from typing import Optional


def normalize_url(url: str) -> str:
    """Normalize URL by adding https:// if missing"""
    url = url.strip()
    if url and not re.match(r"^https?://", url, flags=re.I):
        url = "https://" + url
    return url


def is_url(s: str) -> bool:
    """Check if string is a valid URL"""
    try:
        parsed = urlparse(s)
        return parsed.scheme in ("http", "https")
    except Exception:
        return False


def validate_non_empty(value: Optional[str], field_name: str = "value") -> str:
    """Validate that a string is not None or empty"""
    if not value or not value.strip():
        raise ValueError(f"{field_name} cannot be empty")
    return value.strip()


def truncate_text(text: str, max_chars: int, suffix: str = "\n\n... (truncated)") -> str:
    """Truncate text to max_chars with suffix"""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + suffix
