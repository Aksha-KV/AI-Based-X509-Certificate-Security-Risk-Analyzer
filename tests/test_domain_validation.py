"""
Tests for domain validation, URL sanitization, and typosquatting detection.
"""

import pytest
from analyzer.domain_validation import (
    validate_domain,
    check_typosquatting,
    extract_hostname,
    levenshtein_distance,
    collapse_repeated_chars,
    PROTECTED_DOMAINS,
)


class TestFQDNValidation:
    """Test strict FQDN validation and URL sanitization."""

    @pytest.mark.parametrize("valid_input,expected_host", [
        ("example.com", "example.com"),
        ("google.com", "google.com"),
        ("https://github.com/", "github.com"),
        ("http://sub.domain.org/path?q=1#hash", "sub.domain.org"),
        ("  api.cloudflare.com:443  ", "api.cloudflare.com"),
        ("internshala.com", "internshala.com"),
        ("my-site.co.uk", "my-site.co.uk"),
    ])
    def test_valid_fqdn_inputs(self, valid_input, expected_host):
        valid, host, err = validate_domain(valid_input, allow_private=False)
        assert valid is True
        assert host == expected_host
        assert err is None

    @pytest.mark.parametrize("invalid_input", [
        ("google"),                   # Bare word, no TLD dot
        ("gooogle .com"),             # Internal space before dot
        ("https://goo gle.com"),      # Internal space inside host
        ("test..com"),                # Double dot
        ("-start.com"),               # Leading hyphen in label
        ("end-.com"),                 # Trailing hyphen in label
        ("domain.c"),                 # Single char TLD
        ("domain.123"),               # Numeric TLD
        (""),                         # Empty string
        ("   "),                      # Whitespace only
    ])
    def test_invalid_fqdn_inputs(self, invalid_input):
        valid, host, err = validate_domain(invalid_input, allow_private=False)
        assert valid is False
        assert "Invalid domain or URL format" in err

    def test_allow_private_hosts(self):
        # Without allow_private, localhost and single words fail
        valid, _, err = validate_domain("localhost", allow_private=False)
        assert valid is False
        assert err is not None

        # With allow_private=True, localhost succeeds
        valid, host, err = validate_domain("localhost", allow_private=True)
        assert valid is True
        assert host == "localhost"
        assert err is None


class TestTyposquattingDetection:
    """Test catching typosquatted/misspelled domains."""

    @pytest.mark.parametrize("typo_domain,expected_target", [
        ("intershala.com", "internshala.com"),
        ("gooogle.com", "google.com"),
        ("githbub.com", "github.com"),
        ("microssoft.com", "microsoft.com"),
        ("appple.com", "apple.com"),
        ("paypall.com", "paypal.com"),
        ("cloudflar.com", "cloudflare.com"),
        ("sub.gooogle.com", "google.com"),
    ])
    def test_detects_typosquats(self, typo_domain, expected_target):
        valid, host, _ = validate_domain(typo_domain)
        assert valid is True
        is_typo, matched, msg = check_typosquatting(host)
        assert is_typo is True
        assert matched == expected_target
        assert f"Did you mean '{expected_target}'?" in msg

    @pytest.mark.parametrize("legitimate_domain", [
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
        "sub.google.com",
        "mail.google.com",
        "api.github.com",
        "example.com",
    ])
    def test_legitimate_domains_not_flagged(self, legitimate_domain):
        valid, host, _ = validate_domain(legitimate_domain)
        assert valid is True
        is_typo, matched, msg = check_typosquatting(host)
        assert is_typo is False
        assert matched is None
        assert msg is None


class TestLevenshteinDistance:
    """Test basic Levenshtein algorithm correctness."""

    def test_levenshtein(self):
        assert levenshtein_distance("kitten", "sitting") == 3
        assert levenshtein_distance("google", "gooogle") == 1
        assert levenshtein_distance("internshala", "intershala") == 1
        assert levenshtein_distance("apple", "apple") == 0
        assert collapse_repeated_chars("goooogle") == "gogle"
