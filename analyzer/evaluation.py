"""
Evaluation harness: run the analyzer on hosts with KNOWN problems and measure
whether the expected findings appear.

`BADSSL_CASES` uses the public badssl.com test hosts. The harness takes a `fetch`
callable so it can be unit-tested offline; scripts/evaluate_badssl.py plugs in the
real network fetcher.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Set

from .pipeline import analyze

# expect: rule ids that MUST be reported.  clean=True: no High/Critical finding allowed.
# known_miss=True: a problem this tool does not claim to detect (reported honestly, not scored).
BADSSL_CASES: List[Dict[str, Any]] = [
    {"host": "expired.badssl.com", "expect": {"EXPIRED"}},
    {"host": "wrong.host.badssl.com", "expect": {"HOSTNAME_MISMATCH"}},
    {"host": "self-signed.badssl.com", "expect": {"SELF_SIGNED"}},
    {"host": "untrusted-root.badssl.com", "expect": {"UNTRUSTED_CHAIN"}},
    {"host": "incomplete-chain.badssl.com", "expect": {"UNTRUSTED_CHAIN"}},
    {"host": "sha1-intermediate.badssl.com", "expect": {"WEAK_CHAIN_SIGNATURE"}},
    {"host": "1000-sans.badssl.com", "expect": {"EXCESSIVE_SAN_COUNT"}},
    {"host": "badssl.com", "clean": True},
    {"host": "rsa2048.badssl.com", "clean": True},
    {"host": "rsa4096.badssl.com", "clean": True},
    {"host": "ecc256.badssl.com", "clean": True},
    {"host": "ecc384.badssl.com", "clean": True},
    {"host": "no-common-name.badssl.com", "clean": True},
    {"host": "revoked.badssl.com", "expect": {"REVOKED"}, "known_miss": True},
]


def evaluate(cases: List[Dict[str, Any]], fetch: Callable[[str], Dict[str, Any]],
             ml_model=None) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    tp = fn = clean_ok = clean_total = unexpected_high = errors = 0

    for case in cases:
        host = case["host"]
        try:
            report = analyze(fetch(host), ml_model)
        except Exception as exc:
            errors += 1
            rows.append({"host": host, "result": "ERROR", "found": "", "missing": "", "note": str(exc)[:80]})
            continue
        found: Set[str] = {v["rule_id"] for v in report["rules"]["vulnerabilities"]}
        severe = {v["rule_id"] for v in report["rules"]["vulnerabilities"] if v["severity"] in ("High", "Critical")}
        expect: Set[str] = set(case.get("expect", set()))

        if case.get("known_miss"):
            rows.append({"host": host, "result": "KNOWN-MISS", "found": ",".join(sorted(found)),
                         "missing": ",".join(sorted(expect)), "note": "not covered by this tool (e.g. live revocation)"})
            continue

        if case.get("clean"):
            clean_total += 1
            ok = not severe
            clean_ok += ok
            rows.append({"host": host, "result": "PASS" if ok else "FALSE-POSITIVE", "found": ",".join(sorted(found)),
                         "missing": "", "note": "" if ok else "High/Critical on a healthy host"})
            continue

        hit, miss = expect & found, expect - found
        tp += len(hit)
        fn += len(miss)
        unexpected = severe - expect
        unexpected_high += len(unexpected)
        rows.append({"host": host, "result": "PASS" if not miss else "MISS", "found": ",".join(sorted(found)),
                     "missing": ",".join(sorted(miss)), "note": f"extra High/Critical: {sorted(unexpected)}" if unexpected else ""})

    recall = tp / (tp + fn) if (tp + fn) else None
    return {
        "rows": rows,
        "summary": {
            "expected_findings_detected": tp, "expected_findings_missed": fn,
            "recall": None if recall is None else round(recall, 3),
            "healthy_hosts_passed": clean_ok, "healthy_hosts_total": clean_total,
            "unexpected_high_or_critical_on_bad_hosts": unexpected_high, "fetch_errors": errors,
        },
    }
