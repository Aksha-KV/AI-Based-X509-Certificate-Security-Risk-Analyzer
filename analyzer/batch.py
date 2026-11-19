"""Scan many endpoints or certificate files and rank them by risk."""

from __future__ import annotations

import csv
import io
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, Iterable, List, Optional

from .extractor import fetch_cert_from_domain, parse_cert_from_pem
from .ml_model import CertificateAnomalyModel
from .pipeline import analyze

CSV_FIELDS = ["target", "status", "risk_level", "risk_score", "rule_score", "ml_anomaly_percentile",
              "issues", "days_remaining", "key", "signature", "issuer_cn", "error"]


def _row_from_report(target: str, report: Dict[str, Any]) -> Dict[str, Any]:
    c, r, f = report["cert"], report["rules"], report["final"]
    return {
        "target": target, "status": "ok",
        "risk_level": f["level"], "risk_score": f["score"], "rule_score": f["rule_score"],
        "ml_anomaly_percentile": f["ml_anomaly_percentile"],
        "issues": ";".join(v["rule_id"] for v in r["vulnerabilities"]),
        "days_remaining": c["days_remaining"],
        "key": f"{c['key_type']}-{c['key_size']}", "signature": c["signature_algorithm"],
        "issuer_cn": c["issuer_dict"]["common_name"] or c["issuer_dict"]["organization"], "error": "",
    }


def _error_row(target: str, exc: Exception) -> Dict[str, Any]:
    row = {k: "" for k in CSV_FIELDS}
    row.update(target=target, status="error", error=f"{type(exc).__name__}: {exc}")
    return row


def scan_targets(
    targets: Iterable[str],
    ml_model: Optional[CertificateAnomalyModel] = None,
    workers: int = 8,
    timeout: float = 8,
    allow_private: bool = False,
    save_pem_dir: Optional[str] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> List[Dict[str, Any]]:
    """Scan host[:port] targets concurrently. Returns rows sorted by risk (errors last)."""
    items = [t.strip() for t in targets if t.strip() and not t.strip().startswith("#")]
    rows: List[Dict[str, Any]] = []

    def work(target: str) -> Dict[str, Any]:
        try:
            cert, details = fetch_cert_from_domain(target, timeout=timeout, allow_private=allow_private)
            if save_pem_dir:
                from cryptography.hazmat.primitives import serialization
                os.makedirs(save_pem_dir, exist_ok=True)
                name = "".join(ch if ch.isalnum() or ch in ".-_" else "_" for ch in target)
                with open(os.path.join(save_pem_dir, f"{name}.pem"), "wb") as fh:
                    fh.write(cert.public_bytes(serialization.Encoding.PEM))
            return _row_from_report(target, analyze(details, ml_model))
        except Exception as exc:  # one bad host must not stop the scan
            return _error_row(target, exc)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(work, t) for t in items]
        for done, fut in enumerate(as_completed(futures), 1):
            rows.append(fut.result())
            if progress:
                progress(done, len(items))

    rows.sort(key=lambda r: (r["status"] != "ok", -(r["risk_score"] or 0) if r["status"] == "ok" else 0, r["target"]))
    return rows


def rows_to_csv(rows: List[Dict[str, Any]]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()
