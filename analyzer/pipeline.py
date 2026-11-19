"""Single entry point that runs rules + ML and produces the final explainable report."""

from __future__ import annotations

import datetime
from typing import Any, Dict, Optional

from .ml_model import CertificateAnomalyModel
from .rules_engine import combine_scores, evaluate_certificate_rules


def analyze(cert_details: Dict[str, Any], ml_model: Optional[CertificateAnomalyModel] = None,
            expect_leaf: bool = True) -> Dict[str, Any]:
    """
    Run the deterministic rules, then (optionally) the anomaly model, and combine them.
    Rules always dominate; the ML can only add a bounded amount of risk.
    """
    rules = evaluate_certificate_rules(cert_details, expect_leaf=expect_leaf)
    if ml_model is not None:
        ml = ml_model.assess(cert_details)
        final = combine_scores(rules, ml["ml_anomaly_percentile"])
    else:
        ml = None
        final = combine_scores(rules, 0.0)

    top = rules["vulnerabilities"][:3]
    reasons = [f"{v['title']} (+{v['score_contribution']})" for v in top]
    if final["ml_contribution"] > 0 and ml:
        reasons.append(f"Unusual certificate structure (+{final['ml_contribution']})")
    return {
        "analyzed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "expect_leaf": expect_leaf,
        "cert": cert_details,
        "rules": rules,
        "ml": ml,
        "final": {**final, "summary_reasons": reasons},
    }
