"""
Domain validation, normalization, and typosquatting detection for the
X.509 Certificate Security Risk Analyzer.
"""

from __future__ import annotations

import re
from typing import Optional, Set, Tuple
from urllib.parse import urlsplit

_MAX_DOMAIN_LEN = 253

# Strict FQDN regex:
# - Length 1 to 253 total
# - Labels separated by dots, each label 1-63 chars, alphanumeric with optional hyphens (not leading/trailing)
# - Valid TLD 2-63 alphabetic chars
FQDN_REGEX = re.compile(
    r"^(?=.{1,253}$)(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,63}$"
)

# High-profile protected domains for typosquatting checks
PROTECTED_DOMAINS: Set[str] = {
    "google.com",
    "internshala.com",
    "github.com",
    "microsoft.com",
    "amazon.com",
    "apple.com",
    "cloudflare.com",
    "paypal.com",
    "linkedin.com",
    "instagram.com",
    "youtube.com",
    "msrit.edu",
}


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute the Levenshtein edit distance between two strings."""
    if s1 == s2:
        return 0
    if len(s1) == 0:
        return len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def collapse_repeated_chars(s: str) -> str:
    """Collapse consecutive duplicate characters to a single character (e.g. gooogle -> gogle)."""
    return re.sub(r"(.)\1+", r"\1", s)


def extract_hostname(raw: str) -> Tuple[str, bool]:
    """
    Extract hostname from raw user input, stripping scheme, trailing path,
    query string, port, and whitespace.

    Returns:
        (extracted_hostname, has_internal_spaces)
    """
    if not raw:
        return "", False

    stripped = raw.strip()
    if not stripped:
        return "", False

    # Check for internal spaces anywhere before stripping scheme/path
    # (e.g., 'gooogle .com', 'https://goo gle.com')
    has_internal_spaces = bool(re.search(r"\s", stripped))

    # Strip scheme if present
    text = stripped
    for prefix in ("https://", "http://"):
        if text.lower().startswith(prefix):
            text = text[len(prefix):]
            break

    # Strip port, path, query if present
    # Handle '//' prefix if someone passed '//github.com'
    if text.startswith("//"):
        text = text[2:]

    # Extract part before '/', '?', '#', ':'
    # If there's an IPv6 bracket like [::1]:443, handle that carefully
    if text.startswith("["):
        bracket_end = text.find("]")
        if bracket_end != -1:
            host_part = text[1:bracket_end]
        else:
            host_part = text.split("/")[0].split("?")[0].split("#")[0]
    else:
        # Split on path / query / fragment first
        base = text.split("/")[0].split("?")[0].split("#")[0]
        # Split on port if ':' is present
        host_part = base.split(":")[0]

    return host_part.strip().lower()[:_MAX_DOMAIN_LEN], has_internal_spaces


def validate_domain(
    raw: str, allow_private: bool = False
) -> Tuple[bool, str, Optional[str]]:
    """
    Validate raw domain input according to strict FQDN rules.

    Args:
        raw: The user-supplied input (domain or URL).
        allow_private: Whether private/internal hosts (e.g. localhost, local names, private IPs)
                       are permitted.

    Returns:
        (is_valid, normalized_hostname, error_message_or_None)
    """
    default_err = "Invalid domain or URL format. Please enter a valid domain like example.com."
    if not raw or not raw.strip():
        return False, "", default_err

    host, has_internal_spaces = extract_hostname(raw)

    if has_internal_spaces:
        return False, host, default_err

    if not host:
        return False, "", default_err

    # If private hosts are allowed, allow localhost or single-label names or loopback/IPs
    if allow_private:
        # Basic sanity check
        if host == "localhost" or "." not in host or host.startswith("["):
            return True, host, None
        # Could also be an IP or standard domain
        return True, host, None

    # Strict FQDN validation:
    # Must have a valid dot, no internal spaces, valid TLD
    if "." not in host:
        return False, host, default_err

    if not FQDN_REGEX.match(host):
        return False, host, default_err

    return True, host, None


def get_registrable_domain(host: str) -> str:
    """
    Extract the registrable domain (e.g., sub.example.com -> example.com).
    For simplicity and speed without external PSL dependencies, handles standard
    second-level domains (e.g., .com, .org, .co.uk).
    """
    parts = host.lower().split(".")
    if len(parts) <= 2:
        return host.lower()

    # Common multi-part TLD suffixes
    two_part_tlds = {"co.uk", "gov.uk", "ac.uk", "org.uk", "com.au", "net.au", "co.jp"}
    if len(parts) >= 3 and f"{parts[-2]}.{parts[-1]}" in two_part_tlds:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def check_typosquatting(
    host: str,
    protected_domains: Optional[Set[str]] = None,
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Check if the input host or its registrable domain is a suspected typosquat
    of high-profile target domains.

    Returns:
        (is_typosquat, matched_domain_or_None, error_message_or_None)
    """
    if not host:
        return False, None, None

    targets = protected_domains or PROTECTED_DOMAINS
    reg_domain = get_registrable_domain(host)

    for protected in targets:
        # Exact match is legitimate, not typosquatting!
        if host == protected or reg_domain == protected or host.endswith("." + protected):
            continue

        # Check against the registrable domain (e.g. gooogle.com vs google.com)
        # as well as the full host if short
        for candidate in {reg_domain, host}:
            dist = levenshtein_distance(candidate, protected)
            # Edit distance of 1 or 2
            if 0 < dist <= 2:
                msg = f"Invalid or misspelled domain: '{host}'. Did you mean '{protected}'?"
                return True, protected, msg

            # Repeated-character typo check: e.g. gooogle.com -> google.com
            # Check if collapsing repeated characters in the candidate matches the protected domain
            # or if candidate without repeated chars has distance <= 1 from protected
            cand_parts = candidate.split(".")
            prot_parts = protected.split(".")
            if len(cand_parts) == len(prot_parts) and cand_parts[-1] == prot_parts[-1]:
                collapsed_name = collapse_repeated_chars(cand_parts[0])
                collapsed_prot = collapse_repeated_chars(prot_parts[0])
                if collapsed_name == collapsed_prot and cand_parts[0] != prot_parts[0]:
                    msg = f"Invalid or misspelled domain: '{host}'. Did you mean '{protected}'?"
                    return True, protected, msg

    return False, None, None
