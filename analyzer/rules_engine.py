"""
Deterministic, explainable security rules for X.509 certificates.

Design notes
------------
* Every finding carries a rule id, severity, category, penalty, a human-readable
  detail, remediation and the standard it comes from.
* Findings are combined with a *noisy-OR*:  score = 100 * (1 - prod(1 - p_i/100)).
  Unlike a plain sum this is bounded in [0, 100], never exceeds 100 by accident,
  and has diminishing returns (five minor issues are not worse than one critical).
* Root-cause grouping: correlated findings that share an underlying cause are
  suppressed from the score to avoid double-counting.  The suppressed finding still
  appears in the report with ``score_contribution=0`` and
  ``suppressed_by=<root_rule_id>`` for full auditability.  Current groups:
    - SELF_SIGNED suppresses UNTRUSTED_CHAIN (a self-signed leaf is by definition
      not in a trusted chain; counting both would be double jeopardy).
    - CHAIN_BROKEN_LINK suppresses UNTRUSTED_CHAIN for the same reason.
    - CA_ON_LEAF suppresses LEAF_SIGNS_CERTS (the CA-flag check is higher-penalty
      and already describes the same misuse).
* Each finding reports its *marginal contribution* to the final score, so the
  score can be traced back to reasons.
* Availability problems (expiry) are tagged separately from security problems.
* The penalty values are expert-chosen priors, NOT fitted to data. Calibrating
  them against labelled certificates is listed as future work in the README.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional, Set

SEVERITY_ORDER = {"Info": 0, "Low": 1, "Medium": 2, "High": 3, "Critical": 4}

# Maximum certificate validity for publicly trusted TLS certificates, by the date
# the certificate was issued (notBefore). Source: CA/Browser Forum Baseline
# Requirements, ballot SC-081v3 for the 2026+ steps.  Re-check against the current
# BRs before relying on it; private CAs may legitimately use other lifetimes.
VALIDITY_POLICY = [
    (datetime.date(2029, 3, 15), 47),
    (datetime.date(2027, 3, 15), 100),
    (datetime.date(2026, 3, 15), 200),
    (datetime.date(2020, 9, 1), 398),
    (datetime.date.min, 825),
]

ALLOWED_EC_CURVES = {"secp256r1", "secp384r1", "secp521r1"}

# ---------------------------------------------------------------------------
# Root-cause suppression groups.
# Each entry maps a root-cause rule id to the set of rule ids that are
# suppressed (zero-scored) when the root-cause fires.
# ---------------------------------------------------------------------------
SUPPRESSION_GROUPS: Dict[str, Set[str]] = {
    "SELF_SIGNED":       {"UNTRUSTED_CHAIN"},
    "CHAIN_BROKEN_LINK": {"UNTRUSTED_CHAIN"},
    "CA_ON_LEAF":        {"LEAF_SIGNS_CERTS"},
}


def max_validity_days(issued: datetime.date) -> int:
    """Maximum validity (days) allowed for a certificate issued on `issued`."""
    for start, days in VALIDITY_POLICY:
        if issued >= start:
            return days
    return 825


def level_from_score(score: float) -> str:
    if score >= 70:
        return "Critical"
    if score >= 45:
        return "High"
    if score >= 20:
        return "Medium"
    return "Low"


def noisy_or(penalties: List[float]) -> float:
    """Combine independent penalties (0-100) into a single 0-100 score."""
    remaining = 1.0
    for p in penalties:
        remaining *= 1.0 - max(0.0, min(100.0, p)) / 100.0
    return 100.0 * (1.0 - remaining)


class CertificateRulesEngine:
    """Rule-based security analyzer for X.509 certificates."""

    RULE_DEFINITIONS: Dict[str, Dict[str, Any]] = {
        "EXPIRED": dict(
            title="Certificate expired", penalty=60, severity="Critical", category="availability",
            basis="RFC 5280 s6.1 (validity)",
            description="The certificate is past its notAfter date; clients will reject it.",
            remediation="Issue and deploy a new certificate immediately; set up automated renewal (ACME)."),
        "EXPIRING_SOON_CRITICAL": dict(
            title="Certificate expires within 14 days", penalty=25, severity="High", category="availability",
            basis="Operational best practice",
            description="Imminent outage risk if renewal is not completed.",
            remediation="Renew now and verify that automated renewal hooks (e.g. certbot) actually run."),
        "EXPIRING_SOON_HIGH": dict(
            title="Certificate expires within 30 days", penalty=10, severity="Medium", category="availability",
            basis="Operational best practice",
            description="Renewal window is closing.",
            remediation="Schedule renewal in the next maintenance window."),
        "NOT_YET_VALID": dict(
            title="Certificate not yet valid", penalty=45, severity="High", category="availability",
            basis="RFC 5280 s6.1 (validity)",
            description="notBefore is in the future: clock skew or premature deployment.",
            remediation="Check server clock / NTP, or wait until the certificate becomes valid."),
        "WEAK_SIGNATURE": dict(
            title="Weak signature hash (SHA-1 / MD5 / MD2)", penalty=40, severity="High", category="cryptography",
            basis="CA/B Forum BR; NIST SP 800-131A",
            description="Collision attacks make forged certificates practical for these hashes.",
            remediation="Re-issue with SHA-256 or stronger."),
        "WEAK_KEY_SIZE_RSA": dict(
            title="RSA key shorter than 2048 bits", penalty=35, severity="High", category="cryptography",
            basis="NIST SP 800-57 Pt.1; CA/B Forum BR",
            description="Below 112-bit security strength; factoring is feasible.",
            remediation="Generate a new key: RSA >= 2048 (3072 preferred) or ECDSA P-256/P-384."),
        "WEAK_KEY_SIZE_EC": dict(
            title="EC key smaller than 256 bits", penalty=35, severity="High", category="cryptography",
            basis="NIST SP 800-57 Pt.1",
            description="Curve is too small for modern security requirements.",
            remediation="Re-issue with P-256 or P-384."),
        "NONSTANDARD_EC_CURVE": dict(
            title="EC curve not allowed for the Web PKI", penalty=15, severity="Medium", category="cryptography",
            basis="CA/B Forum BR s6.1.5",
            description="Only P-256, P-384 and P-521 are accepted by browsers for public TLS.",
            remediation="Re-issue using secp256r1 or secp384r1."),
        "DSA_KEY": dict(
            title="DSA public key", penalty=30, severity="High", category="cryptography",
            basis="FIPS 186-5; CA/B Forum BR",
            description="DSA is deprecated and unsupported by modern TLS stacks.",
            remediation="Replace with an RSA >= 2048 or ECDSA key."),
        "SMALL_RSA_EXPONENT": dict(
            title="RSA public exponent below 65537", penalty=15, severity="Medium", category="cryptography",
            basis="CA/B Forum BR s6.1.6",
            description="Small exponents (e.g. 3) enable several padding/signature attacks.",
            remediation="Generate a new key with exponent 65537."),
        "SELF_SIGNED": dict(
            title="Self-signed certificate", penalty=30, severity="High", category="trust",
            basis="RFC 5280; CA/B Forum BR",
            description="Issuer equals subject and the signature verifies under its own key; no third party vouches for it.",
            remediation="Use a certificate from a public CA, or distribute your private root to clients."),
        "UNTRUSTED_CHAIN": dict(
            title="Chain does not validate against the system trust store", penalty=35, severity="High", category="trust",
            basis="RFC 5280 s6 (path validation)",
            description="A verifying TLS handshake failed because the issuer is unknown/untrusted or intermediates are missing.",
            remediation="Serve the full chain (leaf + intermediates) and use a CA trusted by your clients."),
        "CHAIN_BROKEN_LINK": dict(
            title="Chain contains a link whose signature does not verify", penalty=35, severity="High", category="trust",
            basis="RFC 5280 s6 (path validation)",
            description="At least one certificate in the served chain was not issued by the next one.",
            remediation="Re-assemble the chain in the correct order with the right intermediates."),
        "WEAK_CHAIN_SIGNATURE": dict(
            title="Intermediate certificate signed with a weak hash", penalty=25, severity="High", category="cryptography",
            basis="CA/B Forum BR; NIST SP 800-131A",
            description="An intermediate in the chain uses SHA-1/MD5, weakening the whole path.",
            remediation="Ask the CA for a current intermediate and replace it in your chain."),
        "NON_CA_ISSUER": dict(
            title="Chain issuer is not a CA", penalty=30, severity="High", category="trust",
            basis="RFC 5280 s4.2.1.9 (basicConstraints)",
            description="A certificate in the chain issued another certificate but is not marked CA:TRUE.",
            remediation="Fix the chain; clients will reject it."),
        "CA_ON_LEAF": dict(
            title="End-entity certificate is marked CA:TRUE", penalty=30, severity="High", category="trust",
            basis="RFC 5280 s4.2.1.9; CA/B Forum BR",
            description="A server certificate that can itself sign certificates turns a key leak into a full MITM capability.",
            remediation="Re-issue the certificate with basicConstraints CA:FALSE."),
        "HOSTNAME_MISMATCH": dict(
            title="Hostname not covered by the certificate", penalty=45, severity="High", category="trust",
            basis="RFC 9525 (service identity)",
            description="The requested hostname does not match any SAN entry.",
            remediation="Re-issue the certificate with the correct SAN entries or fix the DNS/virtual-host mapping."),
        "MISSING_SAN": dict(
            title="No Subject Alternative Name", penalty=20, severity="Medium", category="compliance",
            basis="RFC 9525; CA/B Forum BR",
            description="Modern clients ignore the CN and require SAN for hostname validation.",
            remediation="Re-issue the certificate with all hostnames in the SAN extension."),
        "MISSING_EKU_SERVERAUTH": dict(
            title="Extended Key Usage lacks serverAuth", penalty=15, severity="Medium", category="compliance",
            basis="CA/B Forum BR s7.1.2",
            description="TLS server certificates must permit serverAuth; some clients reject others.",
            remediation="Re-issue with EKU = serverAuth (and clientAuth only if needed)."),
        "EXCESSIVE_LIFETIME": dict(
            title="Validity longer than the Baseline Requirements allow", penalty=20, severity="Medium", category="compliance",
            basis="CA/B Forum BR s6.3.2 (limit depends on issue date)",
            description="Long-lived certificates widen the window in which a stolen key is usable.",
            remediation="Use short-lived certificates issued via ACME automation."),
        "LOW_SERIAL_ENTROPY": dict(
            title="Serial number shorter than 64 bits", penalty=10, severity="Low", category="compliance",
            basis="CA/B Forum BR s7.1 (>= 64 bits CSPRNG output)",
            description="Predictable serials weaken defences against hash-collision attacks on signatures.",
            remediation="Ask the CA to issue serials with at least 64 bits of random output."),
        "NO_REVOCATION_INFO": dict(
            title="No OCSP or CRL pointer", penalty=10, severity="Low", category="compliance",
            basis="RFC 5280 s4.2.1.13 / s4.2.2.1",
            description="Clients have no standard way to learn that this certificate was revoked.",
            remediation="Obtain certificates that carry a CRL distribution point and/or OCSP responder."),
        "NO_SCT": dict(
            title="No embedded Certificate Transparency SCT", penalty=5, severity="Low", category="transparency",
            basis="RFC 9162; browser CT policies",
            description="Publicly trusted certificates are expected to be logged. SCTs can also be delivered "
                        "via TLS extension or OCSP, so this can be a false positive.",
            remediation="Use a CA that embeds SCTs, or deliver them via the TLS extension."),
        "INTERNAL_NAMES": dict(
            title="Internal / private names or IPs in certificate", penalty=15, severity="Medium", category="compliance",
            basis="CA/B Forum BR (internal names deprecated since 2015)",
            description="Private names cannot be validated by a public CA and leak internal topology.",
            remediation="Use a private CA for internal names; keep public certificates to public DNS names."),
        "EXCESSIVE_SAN_COUNT": dict(
            title="Very large number of SAN entries (> 100)", penalty=10, severity="Low", category="hygiene",
            basis="Security best practice",
            description="One key compromise would impact every listed hostname.",
            remediation="Split the certificate by service / trust boundary."),
        "WILDCARD_USAGE": dict(
            title="Wildcard certificate", penalty=10, severity="Low", category="hygiene",
            basis="Security best practice",
            description="If the private key leaks, every subdomain can be impersonated.",
            remediation="Prefer per-host certificates and keep wildcard keys on as few systems as possible."),
        # ----- NEW RULES -----
        "LEGACY_TLS": dict(
            title="Legacy TLS version (TLS 1.0 or 1.1) negotiated", penalty=25, severity="High", category="cryptography",
            basis="NIST SP 800-131A Rev. 2; RFC 8996; PCI DSS 4.0",
            description="TLS 1.0 and TLS 1.1 have known weaknesses (BEAST, POODLE, weak cipher suites). "
                        "Modern clients and standards mandate TLS 1.2 or later.",
            remediation="Disable TLS 1.0/1.1 on the server and configure a minimum of TLS 1.2."),
        "LEAF_SIGNS_CERTS": dict(
            title="Leaf certificate has keyCertSign or cRLSign key usage", penalty=30, severity="High", category="trust",
            basis="RFC 5280 s4.2.1.3; CA/B Forum BR",
            description="A server leaf certificate with keyCertSign or cRLSign in its KeyUsage extension "
                        "can be used to forge certificates or CRLs if the private key is compromised.",
            remediation="Re-issue without keyCertSign or cRLSign in the KeyUsage extension."),
        "CERT_FINGERPRINT_MISMATCH": dict(
            title="Verifying and non-verifying handshakes returned different certificates", penalty=35, severity="High",
            category="trust",
            basis="RFC 5246; security best practice",
            description="The certificate retrieved in the inspection (non-verifying) handshake has a different "
                        "SHA-256 fingerprint to the one returned by the verifying handshake. This can indicate "
                        "a TLS interception proxy, a misconfigured load balancer, or an active MitM attack.",
            remediation="Investigate whether TLS inspection is active on the network path. If unexpected, "
                        "treat the connection as compromised."),
    }

    # --------------------------------------------------------------------- #
    def evaluate(self, cert_details: Dict[str, Any], expect_leaf: bool = True) -> Dict[str, Any]:
        """
        expect_leaf - True for a server (end-entity) certificate. Set False when you are
        deliberately inspecting a CA certificate so CA-specific expectations are skipped.
        """
        d = cert_details
        findings: List[Dict[str, Any]] = []
        passed: List[str] = []
        advisories: List[Dict[str, str]] = []

        def add(rule_id: str, detail: str) -> None:
            rule = self.RULE_DEFINITIONS[rule_id]
            findings.append({**rule, "rule_id": rule_id, "detail": detail})

        # ---- validity window -------------------------------------------- #
        days = d.get("days_remaining", 0)
        if d.get("is_expired", days < 0):
            add("EXPIRED", f"Expired {abs(days)} day(s) ago.")
        elif d.get("is_not_yet_valid"):
            add("NOT_YET_VALID", "notBefore is in the future.")
        elif days <= 14:
            add("EXPIRING_SOON_CRITICAL", f"Expires in {days} day(s).")
        elif days <= 30:
            add("EXPIRING_SOON_HIGH", f"Expires in {days} day(s).")
        else:
            passed.append(f"Certificate is valid for {days} more days.")

        # ---- lifetime vs policy ----------------------------------------- #
        lifetime = d.get("total_lifetime_days", 0)
        if expect_leaf:
            issued = _issue_date(d)
            limit = max_validity_days(issued)
            if lifetime > limit:
                add("EXCESSIVE_LIFETIME",
                    f"Lifetime is {lifetime} days; limit for certificates issued {issued.isoformat()} is {limit} days "
                    "(applies to publicly trusted TLS certificates).")
            else:
                passed.append(f"Lifetime {lifetime} days is within the {limit}-day limit for its issue date.")

        # ---- signature --------------------------------------------------- #
        sig_hash = (d.get("signature_hash") or "").upper()
        sig_alg = str(d.get("signature_algorithm", "")).upper().replace("-", "")
        if sig_hash in {"SHA1", "MD5", "MD2"} or any(w in sig_alg for w in ("SHA1", "MD5", "MD2")):
            add("WEAK_SIGNATURE", f"Signature algorithm is {d.get('signature_algorithm')}.")
        else:
            passed.append(f"Strong signature algorithm ({d.get('signature_algorithm')}).")

        # ---- public key -------------------------------------------------- #
        ktype = str(d.get("key_type", "")).upper()
        ksize = d.get("key_size", 0)
        if ktype == "RSA":
            if ksize < 2048:
                add("WEAK_KEY_SIZE_RSA", f"RSA key is {ksize} bits (minimum 2048).")
            else:
                passed.append(f"RSA key size {ksize} bits is acceptable.")
            exp = d.get("rsa_exponent")
            if exp is not None and exp < 65537:
                add("SMALL_RSA_EXPONENT", f"Public exponent is {exp}.")
        elif ktype == "EC":
            curve = d.get("key_curve")
            if ksize < 256:
                add("WEAK_KEY_SIZE_EC", f"EC key is {ksize} bits (minimum 256).")
            elif curve and curve not in ALLOWED_EC_CURVES:
                add("NONSTANDARD_EC_CURVE", f"Curve {curve} is not P-256/P-384/P-521.")
            else:
                passed.append(f"EC key on {curve or 'unknown curve'} ({ksize} bits) is acceptable.")
        elif ktype == "DSA":
            add("DSA_KEY", f"DSA key of {ksize} bits.")
        else:
            passed.append(f"Public key type {d.get('key_type')} ({ksize} bits).")

        # ---- trust ------------------------------------------------------- #
        if d.get("is_self_signed") and not (d.get("is_ca") and not expect_leaf):
            add("SELF_SIGNED", "Subject equals issuer and the signature verifies with the certificate's own key.")
        elif d.get("is_self_signed"):
            passed.append("Self-signed CA certificate (expected for a trust anchor).")
        else:
            passed.append("Certificate is issued by a different entity.")

        if expect_leaf and d.get("is_ca"):
            add("CA_ON_LEAF", "basicConstraints CA:TRUE on a server certificate.")

        # ---- NEW: leaf key usage ----------------------------------------- #
        # keyCertSign (key_cert_sign) and cRLSign (crl_sign) must not appear
        # on an end-entity leaf certificate.  CA_ON_LEAF is higher-penalty and
        # suppresses this finding, but we emit it here so the full picture is
        # visible even if CA_ON_LEAF is not triggered (e.g. a leaf without
        # basicConstraints but with these bits set).
        if expect_leaf and not d.get("is_ca"):
            ku = d.get("key_usage") or []
            # The extractor stores human-readable labels; also check raw flag names
            # for forward compatibility with both label sets.
            has_cert_sign = any(label in ku for label in ("Certificate Sign", "keyCertSign", "key_cert_sign"))
            has_crl_sign  = any(label in ku for label in ("CRL Sign", "cRLSign", "crl_sign"))
            if has_cert_sign or has_crl_sign:
                bits = []
                if has_cert_sign:
                    bits.append("keyCertSign")
                if has_crl_sign:
                    bits.append("cRLSign")
                add("LEAF_SIGNS_CERTS",
                    f"KeyUsage contains {' and '.join(bits)} on a server leaf certificate.")

        tls = d.get("tls") or {}
        if tls.get("verified") is True:
            passed.append("Chain validates against the system trust store.")
        elif tls.get("verified") is False:
            code = tls.get("verify_code")
            # Only add UNTRUSTED_CHAIN for causes not already covered by other rules.
            # 2 unable to get issuer, 19 self-signed in chain, 20 unable to get local issuer,
            # 21 unable to verify first cert, 27 not trusted.
            if code in (2, 19, 20, 21, 27):
                add("UNTRUSTED_CHAIN", f"Verifying handshake failed: {tls.get('reason')} (OpenSSL code {code}).")

        chain = d.get("chain")
        if chain:
            if not chain.get("all_links_valid", True):
                add("CHAIN_BROKEN_LINK", "A certificate in the served chain was not issued by the next one.")
            if chain.get("weak_intermediates"):
                add("WEAK_CHAIN_SIGNATURE",
                    "Weak hash on: " + ", ".join(chain["weak_intermediates"]))
            if chain.get("non_ca_issuers"):
                add("NON_CA_ISSUER", "Issuer without CA:TRUE: " + ", ".join(chain["non_ca_issuers"]))

        # ---- NEW: Legacy TLS negotiated ---------------------------------- #
        tls_version = tls.get("tls_version") or ""
        if tls_version in ("TLSv1", "TLSv1.1", "TLSv1 ", "TLSv1.1 ") or tls_version.strip() in ("TLSv1", "TLSv1.1"):
            add("LEGACY_TLS",
                f"Server negotiated {tls_version.strip()}. TLS 1.2 is the minimum acceptable version.")
        else:
            if tls_version:
                passed.append(f"TLS version {tls_version} is acceptable.")

        # ---- NEW: Fingerprint mismatch (interception indicator) ---------- #
        verify_fp = tls.get("verified_fingerprint")
        probe_fp  = d.get("sha256_fingerprint")
        if verify_fp and probe_fp and verify_fp != probe_fp:
            add("CERT_FINGERPRINT_MISMATCH",
                f"Non-verifying handshake fingerprint: {probe_fp[:16]}… "
                f"vs verifying handshake fingerprint: {verify_fp[:16]}…")

        # ---- identity ---------------------------------------------------- #
        if d.get("hostname") and d.get("hostname_match") is False:
            add("HOSTNAME_MISMATCH",
                f"'{d['hostname']}' is not covered by SAN entries {d.get('san_list') or '[]'}.")
        elif d.get("hostname") and d.get("hostname_match"):
            passed.append(f"Hostname '{d['hostname']}' is covered by the certificate.")

        if not d.get("san_list"):
            add("MISSING_SAN", "No SAN entries found.")
        else:
            passed.append(f"Contains {d.get('san_count')} SAN entr{'y' if d.get('san_count') == 1 else 'ies'}.")

        if expect_leaf and not d.get("eku_server_auth"):
            detail = "No EKU extension." if not d.get("has_eku") else f"EKU is {d.get('eku')}."
            add("MISSING_EKU_SERVERAUTH", detail)

        if d.get("san_private_names") and not d.get("is_self_signed"):
            add("INTERNAL_NAMES", "Private/internal entries: " + ", ".join(d["san_private_names"][:5]))

        if d.get("san_count", 0) > 100:
            add("EXCESSIVE_SAN_COUNT", f"{d['san_count']} SAN entries.")

        if d.get("is_wildcard"):
            add("WILDCARD_USAGE", "Wildcard name present (*.domain).")

        # ---- hygiene ----------------------------------------------------- #
        if d.get("serial_bits", 128) < 64:
            add("LOW_SERIAL_ENTROPY", f"Serial number has {d.get('serial_bits')} bits.")

        if expect_leaf and not d.get("is_self_signed"):
            if not d.get("ocsp_urls") and not d.get("crl_urls"):
                add("NO_REVOCATION_INFO", "Neither OCSP nor CRL distribution point present.")
            if not d.get("has_sct"):
                add("NO_SCT", "No embedded SCT extension found.")

        # ---- advisories (do not affect the score) ------------------------ #
        if ktype in {"RSA", "EC", "DSA", "ED25519", "ED448"}:
            advisories.append({
                "id": "PQC_READINESS", "severity": "Info",
                "title": "Not post-quantum safe (expected today)",
                "detail": f"{d.get('key_type')} keys can be broken by a large quantum computer (Shor). "
                          "No action is required for most certificates now, but keep an inventory, "
                          "favour crypto-agility, and watch NIST FIPS 203/204/205 and hybrid certificates."})
        if expect_leaf and not d.get("is_self_signed") and lifetime > 100:
            advisories.append({
                "id": "SHORT_LIVED_TREND", "severity": "Info",
                "title": "Validity limits are shrinking",
                "detail": "The Baseline Requirements schedule lowers the maximum lifetime to 200, 100 and later "
                          "47 days. Make sure issuance and renewal are automated."})
        if tls.get("verified") is None and d.get("tls") is not None:
            advisories.append({
                "id": "TLS_NOT_VERIFIED", "severity": "Info", "title": "Trust-store verification unavailable",
                "detail": "The verifying handshake could not complete, so chain trust was not assessed."})

        # ---- root-cause grouping / suppression --------------------------- #
        # Build the set of rule_ids that fired.
        fired_ids: Set[str] = {f["rule_id"] for f in findings}
        # Determine which findings should be suppressed (zero-scored) because
        # a higher-priority root-cause finding already explains the same problem.
        suppressed: Dict[str, str] = {}  # rule_id -> suppressed_by
        for root_id, dependents in SUPPRESSION_GROUPS.items():
            if root_id in fired_ids:
                for dep in dependents:
                    if dep in fired_ids and dep not in suppressed:
                        suppressed[dep] = root_id

        # ---- scoring ----------------------------------------------------- #
        ordered = sorted(findings, key=lambda f: (-f["penalty"], f["rule_id"]))
        cumulative = 0.0
        applied: List[float] = []
        for f in ordered:
            if f["rule_id"] in suppressed:
                # Zero-scored; mark suppression for transparency
                f["score_contribution"] = 0.0
                f["suppressed_by"] = suppressed[f["rule_id"]]
            else:
                applied.append(f["penalty"])
                new_total = noisy_or(applied)
                f["score_contribution"] = round(new_total - cumulative, 1)
                cumulative = new_total

        score = round(cumulative, 1)

        top_severity = max((SEVERITY_ORDER[f["severity"]] for f in findings), default=0)
        level = level_from_score(score)
        if top_severity >= SEVERITY_ORDER["Critical"]:
            level = "Critical"

        return {
            "risk_score": score,
            "risk_level": level,
            "total_penalty": sum(f["penalty"] for f in findings),
            "vulnerabilities": ordered,
            "advisories": advisories,
            "remediations": list(dict.fromkeys(f["remediation"] for f in ordered)),
            "passed_checks": passed,
            "issue_count": len(findings),
            "category_counts": _count_by(ordered, "category"),
            "severity_floor": "Critical" if top_severity >= SEVERITY_ORDER["Critical"] else None,
        }


def _issue_date(d: Dict[str, Any]) -> datetime.date:
    """
    Safely extract the notBefore date.  If the field is missing or unparseable
    (e.g. due to a legacy certificate with a non-ISO format), fall back to the
    epoch rather than today – using today would silently apply the most lenient
    policy tier to every certificate with a bad date.
    """
    try:
        raw = d["not_before"]
        return datetime.datetime.fromisoformat(raw).date()
    except (KeyError, TypeError, ValueError):
        # Return the epoch date so max_validity_days falls through to the 825-day
        # tier (oldest, most permissive), which is the safest conservative fallback.
        return datetime.date.min


def _count_by(findings: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for f in findings:
        out[f[key]] = out.get(f[key], 0) + 1
    return out


def combine_scores(rule_result: Dict[str, Any], ml_anomaly: float, ml_weight: float = 0.25) -> Dict[str, Any]:
    """
    Blend the deterministic rule score with the ML anomaly signal.

    * The rules always dominate: the ML can only ADD risk, never remove it.
    * ML only counts for the most unusual ~10% of certificates
      (anomaly percentile 0.90 -> 0, 1.00 -> 1) and is capped at `ml_weight`.
    """
    rule_score = rule_result["risk_score"]
    effective = max(0.0, (ml_anomaly - 0.90) / 0.10)
    final = 100.0 * (1.0 - (1.0 - rule_score / 100.0) * (1.0 - ml_weight * effective))
    final = round(min(100.0, max(0.0, final)), 1)
    level = level_from_score(final)
    if rule_result.get("severity_floor") == "Critical":
        level = "Critical"
    return {
        "score": final,
        "level": level,
        "rule_score": rule_score,
        "ml_anomaly_percentile": round(ml_anomaly, 4),
        "ml_effective_weight": round(ml_weight * effective, 4),
        "ml_contribution": round(final - rule_score, 1),
    }


def evaluate_certificate_rules(cert_details: Dict[str, Any], expect_leaf: bool = True) -> Dict[str, Any]:
    return CertificateRulesEngine().evaluate(cert_details, expect_leaf=expect_leaf)
