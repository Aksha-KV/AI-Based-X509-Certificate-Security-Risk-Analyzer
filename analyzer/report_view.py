"""Presentation helpers (pure Python / pandas) so the UI layer stays thin and testable."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pandas as pd

from .ml_model import ML_OUTLIER_THRESHOLD

# Show ML factors only when the certificate is genuinely unusual.
# Uses the same unified threshold as the model itself to avoid divergence.
ML_EXPLAIN_THRESHOLD = ML_OUTLIER_THRESHOLD - 0.15  # 0.80: factors shown at 80th percentile+


def findings_table(report: Dict[str, Any]) -> pd.DataFrame:
    rows = [{
        "Severity": v["severity"], "Category": v["category"], "Rule": v["rule_id"], "Finding": v["title"],
        "Detail": v["detail"], "Score +": v["score_contribution"], "Basis": v["basis"],
        **( {"Suppressed by": v["suppressed_by"]} if v.get("suppressed_by") else {} ),
    } for v in report["rules"]["vulnerabilities"]]
    cols = ["Severity", "Category", "Rule", "Finding", "Detail", "Score +", "Basis"]
    if any("Suppressed by" in r for r in rows):
        cols.append("Suppressed by")
    return pd.DataFrame(rows, columns=cols)


def ml_factors_table(report: Dict[str, Any]) -> pd.DataFrame:
    ml = report.get("ml")
    if not ml or ml["ml_anomaly_percentile"] < ML_EXPLAIN_THRESHOLD:
        return pd.DataFrame(columns=["Feature", "This certificate", "Typical", "Share of anomaly"])
    return pd.DataFrame([{
        "Feature": f["label"], "This certificate": f["value"], "Typical": f["typical"],
        "Share of anomaly": f"{f['share']:.0%}",
    } for f in ml["ml_top_factors"]])


def chain_table(report: Dict[str, Any]) -> pd.DataFrame:
    chain = report["cert"].get("chain")
    if not chain:
        return pd.DataFrame(columns=["#", "Subject", "Issuer", "Signature hash", "Signed by next"])
    return pd.DataFrame([{
        "#": i, "Subject": link["subject"], "Issuer": link["issuer"],
        "Signature hash": link["signature_hash"] or "-",
        "Signed by next": {True: "yes", False: "NO", None: "n/a"}[link["issued_by_next_verified"]],
    } for i, link in enumerate(chain["links"])])


def ml_caption(report: Dict[str, Any]) -> str:
    ml = report.get("ml")
    if not ml:
        return "ML anomaly model disabled."
    pct = ml["ml_anomaly_percentile"]
    info = ml["model_info"]
    ref = "a synthetic placeholder baseline" if info.get("is_synthetic") else info.get("source", "a reference corpus")
    return (f"More unusual than {pct:.0%} of {info.get('n_reference', '?')} reference certificates "
            f"(reference: {ref}). This is a rarity measure, not a probability of compromise.")


def export_json(report: Dict[str, Any]) -> str:
    return json.dumps(report, indent=2, default=str)
