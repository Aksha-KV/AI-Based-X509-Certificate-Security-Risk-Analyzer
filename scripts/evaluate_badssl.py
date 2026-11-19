"""Evaluate detection against the public badssl.com hosts (needs internet).

  python scripts/evaluate_badssl.py --csv eval_badssl.csv
Reports recall on the known-bad hosts, the pass rate on known-good hosts, and the cases the tool
does not cover (e.g. revoked.badssl.com needs live OCSP/CRL checking).
"""
import _path  # noqa: F401
import argparse
import csv

from analyzer import fetch_cert_from_domain
from analyzer.evaluation import BADSSL_CASES, evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--csv")
ap.add_argument("--timeout", type=float, default=10)
args = ap.parse_args()

result = evaluate(BADSSL_CASES, lambda host: fetch_cert_from_domain(host, timeout=args.timeout)[1])
for r in result["rows"]:
    print(f"{r['result']:<15} {r['host']:<34} found={r['found'] or '-'}  {('missing=' + r['missing']) if r['missing'] else ''} {r['note']}")
print("\nSUMMARY", result["summary"])
if args.csv:
    with open(args.csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["host", "result", "found", "missing", "note"])
        w.writeheader(); w.writerows(result["rows"])
