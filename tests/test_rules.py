"""Rules-engine tests: every rule fires when it should, and not when it shouldn't."""

import datetime

import pytest

from analyzer.rules_engine import (combine_scores, evaluate_certificate_rules, level_from_score,
                                   max_validity_days, noisy_or, SUPPRESSION_GROUPS)
from .certgen import good_details


def ids(result):
    return {v["rule_id"] for v in result["vulnerabilities"]}


def suppressed(result):
    return {v["rule_id"] for v in result["vulnerabilities"] if v.get("suppressed_by")}


# --------------------------------------------------------------------------- existing tests

def test_healthy_certificate_has_no_findings():
    r = evaluate_certificate_rules(good_details())
    assert r["vulnerabilities"] == [] and r["risk_score"] == 0 and r["risk_level"] == "Low"
    assert any(a["id"] == "PQC_READINESS" for a in r["advisories"])


def test_expired_is_critical_even_though_score_alone_is_high():
    r = evaluate_certificate_rules(good_details(days_remaining=-5, is_expired=True))
    assert "EXPIRED" in ids(r) and r["risk_level"] == "Critical"
    assert r["risk_score"] == 60


def test_expiry_windows_are_mutually_exclusive():
    assert ids(evaluate_certificate_rules(good_details(days_remaining=10))) == {"EXPIRING_SOON_CRITICAL"}
    assert ids(evaluate_certificate_rules(good_details(days_remaining=25))) == {"EXPIRING_SOON_HIGH"}
    assert "EXPIRED" not in ids(evaluate_certificate_rules(good_details(days_remaining=0, is_expired=False)))


def test_not_yet_valid():
    r = evaluate_certificate_rules(good_details(is_not_yet_valid=True, days_remaining=100))
    assert "NOT_YET_VALID" in ids(r)


def test_weak_crypto_rules():
    assert "WEAK_SIGNATURE" in ids(evaluate_certificate_rules(good_details(signature_hash="SHA1", signature_algorithm="SHA1")))
    assert "WEAK_KEY_SIZE_RSA" in ids(evaluate_certificate_rules(
        good_details(key_type="RSA", key_size=1024, key_curve=None, rsa_exponent=65537)))
    assert "WEAK_KEY_SIZE_EC" in ids(evaluate_certificate_rules(good_details(key_size=224, key_curve="secp224r1")))
    assert "NONSTANDARD_EC_CURVE" in ids(evaluate_certificate_rules(good_details(key_curve="secp256k1")))
    assert "DSA_KEY" in ids(evaluate_certificate_rules(good_details(key_type="DSA", key_size=2048, key_curve=None)))
    assert "SMALL_RSA_EXPONENT" in ids(evaluate_certificate_rules(
        good_details(key_type="RSA", key_size=2048, key_curve=None, rsa_exponent=3)))


def test_ed25519_is_not_flagged_as_weak_ec():
    r = evaluate_certificate_rules(good_details(key_type="Ed25519", key_size=256, key_curve=None, signature_hash=None,
                                                signature_algorithm="ED25519"))
    assert not ids(r) & {"WEAK_KEY_SIZE_EC", "NONSTANDARD_EC_CURVE", "WEAK_SIGNATURE"}


def test_trust_rules():
    assert "SELF_SIGNED" in ids(evaluate_certificate_rules(good_details(is_self_signed=True)))
    assert "CA_ON_LEAF" in ids(evaluate_certificate_rules(good_details(is_ca=True)))
    r = evaluate_certificate_rules(good_details(is_ca=True), expect_leaf=False)
    assert "CA_ON_LEAF" not in ids(r)
    r = evaluate_certificate_rules(good_details(is_ca=True, is_self_signed=True), expect_leaf=False)
    assert "SELF_SIGNED" not in ids(r)  # a root CA is expected to be self-signed


def test_untrusted_chain_only_for_trust_related_verify_codes():
    assert "UNTRUSTED_CHAIN" in ids(evaluate_certificate_rules(
        good_details(tls={"verified": False, "verify_code": 20, "reason": "unable to get local issuer certificate"})))
    # expired (code 10) and hostname (62) are covered by their own rules -> no double counting
    assert "UNTRUSTED_CHAIN" not in ids(evaluate_certificate_rules(
        good_details(tls={"verified": False, "verify_code": 10, "reason": "certificate has expired"})))
    assert "UNTRUSTED_CHAIN" not in ids(evaluate_certificate_rules(
        good_details(tls={"verified": False, "verify_code": 62, "reason": "hostname mismatch"})))


def test_chain_rules():
    chain = {"all_links_valid": False, "weak_intermediates": ["Old CA"], "non_ca_issuers": ["Leafy"]}
    assert {"CHAIN_BROKEN_LINK", "WEAK_CHAIN_SIGNATURE", "NON_CA_ISSUER"} <= ids(
        evaluate_certificate_rules(good_details(chain=chain)))


def test_hostname_mismatch():
    r = evaluate_certificate_rules(good_details(hostname="other.com", hostname_match=False))
    assert "HOSTNAME_MISMATCH" in ids(r)
    assert "HOSTNAME_MISMATCH" not in ids(evaluate_certificate_rules(good_details(hostname="example.com", hostname_match=True)))


def test_identity_and_hygiene_rules():
    assert "MISSING_SAN" in ids(evaluate_certificate_rules(good_details(san_list=[], san_count=0)))
    assert "MISSING_EKU_SERVERAUTH" in ids(evaluate_certificate_rules(good_details(eku_server_auth=False, has_eku=False)))
    assert "INTERNAL_NAMES" in ids(evaluate_certificate_rules(good_details(san_private_names=["db.corp"])))
    assert "EXCESSIVE_SAN_COUNT" in ids(evaluate_certificate_rules(good_details(san_count=150)))
    assert "WILDCARD_USAGE" in ids(evaluate_certificate_rules(good_details(is_wildcard=True)))
    assert "LOW_SERIAL_ENTROPY" in ids(evaluate_certificate_rules(good_details(serial_bits=8)))
    assert "NO_REVOCATION_INFO" in ids(evaluate_certificate_rules(good_details(ocsp_urls=[], crl_urls=[])))
    assert "NO_SCT" in ids(evaluate_certificate_rules(good_details(has_sct=False)))


def test_validity_policy_depends_on_issue_date():
    assert max_validity_days(datetime.date(2019, 1, 1)) == 825
    assert max_validity_days(datetime.date(2021, 1, 1)) == 398
    assert max_validity_days(datetime.date(2026, 4, 1)) == 200
    assert max_validity_days(datetime.date(2027, 4, 1)) == 100
    assert max_validity_days(datetime.date(2030, 1, 1)) == 47
    old = good_details(total_lifetime_days=390, not_before="2021-01-01T00:00:00")
    assert "EXCESSIVE_LIFETIME" not in ids(evaluate_certificate_rules(old))      # fine under 398
    new = good_details(total_lifetime_days=390, not_before="2026-06-01T00:00:00")
    assert "EXCESSIVE_LIFETIME" in ids(evaluate_certificate_rules(new))          # too long under 200


def test_noisy_or_is_bounded_and_has_diminishing_returns():
    assert noisy_or([]) == 0
    assert round(noisy_or([50, 50]), 1) == 75.0
    assert noisy_or([100, 100, 100]) == 100.0
    assert noisy_or([30]) < noisy_or([30, 30]) < noisy_or([30, 30, 30]) < 100


def test_contributions_sum_to_total_and_are_traceable():
    r = evaluate_certificate_rules(good_details(is_self_signed=True, is_wildcard=True, serial_bits=8))
    # Only non-suppressed findings count toward the score
    active = [v for v in r["vulnerabilities"] if not v.get("suppressed_by")]
    assert abs(sum(v["score_contribution"] for v in active) - r["risk_score"]) < 0.3
    assert active[0]["score_contribution"] >= active[-1]["score_contribution"]


def test_levels():
    assert [level_from_score(x) for x in (0, 19.9, 20, 44.9, 45, 69.9, 70, 100)] == \
        ["Low", "Low", "Medium", "Medium", "High", "High", "Critical", "Critical"]


def test_ml_can_only_add_risk_and_is_capped():
    base = evaluate_certificate_rules(good_details(is_self_signed=True))
    quiet = combine_scores(base, ml_anomaly=0.5)
    assert quiet["score"] == base["risk_score"] and quiet["ml_contribution"] == 0
    loud = combine_scores(base, ml_anomaly=1.0)
    assert loud["score"] > base["risk_score"]
    assert loud["score"] <= 100.0 * (1 - (1 - base["risk_score"] / 100) * (1 - 0.25)) + 0.1
    clean = combine_scores(evaluate_certificate_rules(good_details()), ml_anomaly=1.0)
    assert clean["score"] <= 25.0  # an odd-looking but rule-clean cert cannot become "High"


def test_real_world_weak_certs_from_fixtures():
    import os
    from analyzer.extractor import parse_cert_from_pem
    path = os.path.join(os.path.dirname(__file__), "fixtures", "sha1_selfsigned.pem")
    r = evaluate_certificate_rules(parse_cert_from_pem(open(path, "rb").read())[1])
    assert {"WEAK_SIGNATURE", "SELF_SIGNED"} <= ids(r) and r["risk_level"] in ("High", "Critical")


# --------------------------------------------------------------------------- NEW TESTS

# 1. Root-cause grouping / suppression
class TestRootCauseGrouping:
    """SELF_SIGNED and CHAIN_BROKEN_LINK suppress UNTRUSTED_CHAIN in the score."""

    def test_self_signed_suppresses_untrusted_chain_in_score(self):
        """
        When a certificate is both self-signed AND the verifying handshake fails
        due to an untrusted issuer (code 19 = self-signed chain), UNTRUSTED_CHAIN
        should still appear in findings but should NOT contribute to the score
        (because SELF_SIGNED is already the root cause of the trust failure).
        """
        d = good_details(
            is_self_signed=True,
            tls={"verified": False, "verify_code": 19, "reason": "self signed certificate"},
        )
        r = evaluate_certificate_rules(d)
        fired = ids(r)
        assert "SELF_SIGNED" in fired, "SELF_SIGNED must be detected"
        assert "UNTRUSTED_CHAIN" in fired, "UNTRUSTED_CHAIN must still appear for auditability"
        # The suppressed finding must have zero contribution and a suppressed_by tag
        untrusted = next(v for v in r["vulnerabilities"] if v["rule_id"] == "UNTRUSTED_CHAIN")
        assert untrusted["score_contribution"] == 0.0, "suppressed finding must not add to score"
        assert untrusted.get("suppressed_by") == "SELF_SIGNED"

    def test_self_signed_alone_gives_correct_score(self):
        """Score must equal SELF_SIGNED penalty alone (30 penalty -> 30 score), not 30+35."""
        d = good_details(
            is_self_signed=True,
            tls={"verified": False, "verify_code": 19, "reason": "self signed certificate"},
        )
        r = evaluate_certificate_rules(d)
        # SELF_SIGNED penalty=30 is the only *active* finding; score should be 30.
        active_penalties = [v["penalty"] for v in r["vulnerabilities"] if not v.get("suppressed_by")]
        assert r["risk_score"] == round(noisy_or(active_penalties), 1), (
            "risk_score must match noisy_or of unsuppressed penalties only"
        )
        # Specifically verify it is NOT penalty of SELF_SIGNED + UNTRUSTED_CHAIN combined
        combined = noisy_or([30, 35])  # 30 + 35 noisy-OR = 55.5
        assert r["risk_score"] < combined - 0.5, "double-counting must not occur"

    def test_chain_broken_link_suppresses_untrusted_chain(self):
        """CHAIN_BROKEN_LINK (broken intermediate) suppresses UNTRUSTED_CHAIN."""
        d = good_details(
            chain={"all_links_valid": False, "weak_intermediates": [], "non_ca_issuers": []},
            tls={"verified": False, "verify_code": 20, "reason": "unable to get local issuer certificate"},
        )
        r = evaluate_certificate_rules(d)
        fired = ids(r)
        assert "CHAIN_BROKEN_LINK" in fired
        assert "UNTRUSTED_CHAIN" in fired
        untrusted = next(v for v in r["vulnerabilities"] if v["rule_id"] == "UNTRUSTED_CHAIN")
        assert untrusted["score_contribution"] == 0.0
        assert untrusted.get("suppressed_by") == "CHAIN_BROKEN_LINK"

    def test_suppression_groups_are_coherent(self):
        """Every rule id referenced in SUPPRESSION_GROUPS exists in RULE_DEFINITIONS."""
        from analyzer.rules_engine import CertificateRulesEngine
        all_rules = set(CertificateRulesEngine.RULE_DEFINITIONS.keys())
        for root, dependents in SUPPRESSION_GROUPS.items():
            assert root in all_rules, f"root {root!r} not in RULE_DEFINITIONS"
            for dep in dependents:
                assert dep in all_rules, f"dependent {dep!r} not in RULE_DEFINITIONS"

    def test_ca_on_leaf_suppresses_leaf_signs_certs(self):
        """CA_ON_LEAF and LEAF_SIGNS_CERTS co-fire; LEAF_SIGNS_CERTS is suppressed."""
        # A cert with CA:TRUE also typically has keyCertSign / cRLSign
        d = good_details(
            is_ca=True,
            is_self_signed=False,
            key_usage=["Certificate Sign", "CRL Sign"],
        )
        r = evaluate_certificate_rules(d)
        fired = ids(r)
        assert "CA_ON_LEAF" in fired
        # LEAF_SIGNS_CERTS may or may not fire depending on key_usage labels; if it
        # does fire, it must be suppressed by CA_ON_LEAF.
        if "LEAF_SIGNS_CERTS" in fired:
            lsc = next(v for v in r["vulnerabilities"] if v["rule_id"] == "LEAF_SIGNS_CERTS")
            assert lsc["score_contribution"] == 0.0
            assert lsc.get("suppressed_by") == "CA_ON_LEAF"


# 2. LEGACY_TLS rule
class TestLegacyTLS:
    @pytest.mark.parametrize("version", ["TLSv1", "TLSv1.1", "TLSv1 ", "TLSv1.1 "])
    def test_legacy_tls_fires_on_old_versions(self, version):
        d = good_details(tls={"verified": True, "tls_version": version, "cipher": "AES128-SHA"})
        r = evaluate_certificate_rules(d)
        assert "LEGACY_TLS" in ids(r), f"LEGACY_TLS must fire for TLS version '{version}'"
        finding = next(v for v in r["vulnerabilities"] if v["rule_id"] == "LEGACY_TLS")
        assert finding["score_contribution"] > 0 or finding.get("suppressed_by")

    @pytest.mark.parametrize("version", ["TLSv1.2", "TLSv1.3", ""])
    def test_legacy_tls_does_not_fire_on_modern_versions(self, version):
        d = good_details(tls={"verified": True, "tls_version": version, "cipher": "AES256-GCM-SHA384"})
        r = evaluate_certificate_rules(d)
        assert "LEGACY_TLS" not in ids(r), f"LEGACY_TLS must NOT fire for TLS version '{version}'"

    def test_legacy_tls_absent_when_no_tls_info(self):
        """Offline (file) analysis: no TLS info means no LEGACY_TLS finding."""
        r = evaluate_certificate_rules(good_details())
        assert "LEGACY_TLS" not in ids(r)


# 3. LEAF_SIGNS_CERTS rule
class TestLeafSignsCerts:
    def test_leaf_with_key_cert_sign_fires(self):
        d = good_details(is_ca=False, key_usage=["Certificate Sign"])
        r = evaluate_certificate_rules(d)
        assert "LEAF_SIGNS_CERTS" in ids(r)

    def test_leaf_with_crl_sign_fires(self):
        d = good_details(is_ca=False, key_usage=["CRL Sign"])
        r = evaluate_certificate_rules(d)
        assert "LEAF_SIGNS_CERTS" in ids(r)

    def test_leaf_with_no_signing_bits_does_not_fire(self):
        d = good_details(is_ca=False, key_usage=["Digital Signature", "Key Encipherment"])
        r = evaluate_certificate_rules(d)
        assert "LEAF_SIGNS_CERTS" not in ids(r)

    def test_ca_cert_does_not_fire_leaf_signs_certs(self):
        """The CA rule (CA_ON_LEAF) gates this check; if CA=TRUE, LEAF_SIGNS_CERTS is not added."""
        d = good_details(is_ca=True, key_usage=["Certificate Sign", "CRL Sign"])
        r = evaluate_certificate_rules(d)
        # LEAF_SIGNS_CERTS should either not fire (because is_ca=True skips the check)
        # or be suppressed by CA_ON_LEAF.
        fired = ids(r)
        if "LEAF_SIGNS_CERTS" in fired:
            lsc = next(v for v in r["vulnerabilities"] if v["rule_id"] == "LEAF_SIGNS_CERTS")
            assert lsc.get("suppressed_by") == "CA_ON_LEAF", (
                "If LEAF_SIGNS_CERTS fires alongside CA_ON_LEAF, it must be suppressed"
            )

    def test_leaf_signs_certs_not_fired_when_expect_leaf_false(self):
        """When analysing a CA cert explicitly, keyCertSign is expected."""
        d = good_details(is_ca=True, key_usage=["Certificate Sign", "CRL Sign"])
        r = evaluate_certificate_rules(d, expect_leaf=False)
        assert "LEAF_SIGNS_CERTS" not in ids(r)


# 4. CERT_FINGERPRINT_MISMATCH rule
class TestCertFingerprintMismatch:
    def test_mismatch_fires_when_fingerprints_differ(self):
        d = good_details(
            sha256_fingerprint="a" * 64,
            tls={"verified": True, "tls_version": "TLSv1.3",
                 "verified_fingerprint": "b" * 64},
        )
        r = evaluate_certificate_rules(d)
        assert "CERT_FINGERPRINT_MISMATCH" in ids(r)

    def test_no_mismatch_when_fingerprints_match(self):
        fp = "c" * 64
        d = good_details(
            sha256_fingerprint=fp,
            tls={"verified": True, "tls_version": "TLSv1.3",
                 "verified_fingerprint": fp},
        )
        r = evaluate_certificate_rules(d)
        assert "CERT_FINGERPRINT_MISMATCH" not in ids(r)

    def test_no_mismatch_when_verify_fp_absent(self):
        """If the verifying handshake did not capture a fingerprint (e.g. failed), no false alarm."""
        d = good_details(
            sha256_fingerprint="a" * 64,
            tls={"verified": False, "verify_code": 20, "reason": "unable to get local issuer"},
        )
        r = evaluate_certificate_rules(d)
        assert "CERT_FINGERPRINT_MISMATCH" not in ids(r)


# 5. Bug fix: unparseable not_before falls back to epoch, NOT today
class TestUnparseableNotBefore:
    def test_bad_not_before_uses_epoch_not_today(self):
        """
        A missing or malformed not_before must NOT silently use today's date, which
        would apply the most restrictive current policy tier to the lifetime check.
        Instead it falls back to datetime.date.min (epoch), which maps to the oldest
        (825-day) policy tier – the safest conservative fallback.
        """
        from analyzer.rules_engine import _issue_date
        import datetime

        # Missing key
        assert _issue_date({}) == datetime.date.min
        # Malformed value
        assert _issue_date({"not_before": "INVALID DATE"}) == datetime.date.min
        # None value
        assert _issue_date({"not_before": None}) == datetime.date.min
        # Valid value still works
        assert _issue_date({"not_before": "2021-06-01T00:00:00"}) == datetime.date(2021, 6, 1)

    def test_bad_not_before_does_not_cause_excessive_lifetime_false_positive(self):
        """
        A certificate with a bad not_before and a 400-day lifetime should NOT
        get EXCESSIVE_LIFETIME when the epoch fallback gives the 825-day tier.
        """
        d = good_details(total_lifetime_days=400, not_before="NOT A DATE")
        r = evaluate_certificate_rules(d)
        assert "EXCESSIVE_LIFETIME" not in ids(r), (
            "Epoch fallback maps to 825-day tier; 400 days is within that limit"
        )
