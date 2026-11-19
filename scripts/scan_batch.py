"""Scan a list of hosts and write a risk-ranked CSV (optionally saving certificates as a corpus).

  python scripts/scan_batch.py domains.txt --csv results.csv
  python scripts/scan_batch.py tranco_top500.txt --save-pem data/corpus --csv corpus_scan.csv
"""
import _path  # noqa: F401
import argparse

from analyzer import rows_to_csv, scan_targets
from analyzer.ml_model import CertificateAnomalyModel

ap = argparse.ArgumentParser()
ap.add_argument("targets_file", help="text file, one host[:port] per line ('#' comments allowed)")
ap.add_argument("--csv", default="scan_results.csv")
ap.add_argument("--save-pem", metavar="DIR", help="also save each leaf certificate as PEM (builds a training corpus)")
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--timeout", type=float, default=8)
ap.add_argument("--allow-private", action="store_true", help="permit loopback/RFC1918 targets")
ap.add_argument("--no-ml", action="store_true")
args = ap.parse_args()

targets = open(args.targets_file, encoding="utf-8").read().splitlines()
model = None if args.no_ml else CertificateAnomalyModel()
rows = scan_targets(targets, ml_model=model, workers=args.workers, timeout=args.timeout,
                    allow_private=args.allow_private, save_pem_dir=args.save_pem,
                    progress=lambda d, n: print(f"\r{d}/{n}", end="", flush=True))
print()
with open(args.csv, "w", encoding="utf-8", newline="") as fh:
    fh.write(rows_to_csv(rows))
ok = sum(r["status"] == "ok" for r in rows)
print(f"{ok}/{len(rows)} scanned OK -> {args.csv}")
for r in rows[:10]:
    if r["status"] == "ok":
        print(f"  {r['risk_level']:<8} {r['risk_score']:>5}  {r['target']}  {r['issues']}")
