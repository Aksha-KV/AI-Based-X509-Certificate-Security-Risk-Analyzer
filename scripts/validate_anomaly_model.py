"""
Anomaly-injection validation: compare Isolation Forest vs. Local Outlier Factor
on a set of normal and deliberately injected anomalous certificates.

Usage
-----
    python scripts/validate_anomaly_model.py [--corpus data/corpus]

The script:
1. Builds a normal reference set (real corpus or synthetic baseline).
2. Generates a small set of injected anomalies (self-signed, legacy key,
   extreme lifetime, etc.).
3. Scores every certificate under both Isolation Forest (IF) and LOF.
4. Computes ROC-AUC for each model (1=anomaly, 0=normal label).
5. Prints a comparison table.

Requires: scikit-learn >= 1.0, numpy.
"""

import _path  # noqa: F401 – adds the project root to sys.path

import argparse
import sys

import numpy as np

try:
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.metrics import roc_auc_score
    from sklearn.ensemble import IsolationForest
except ImportError:
    print("ERROR: scikit-learn is required. Install it with: pip install scikit-learn", file=sys.stderr)
    sys.exit(1)

from analyzer.ml_model import (
    CertificateAnomalyModel,
    extract_feature_vector,
    load_corpus,
    synthetic_baseline,
    MIN_CORPUS,
    ML_OUTLIER_THRESHOLD,
)
from analyzer.extractor import parse_cert_from_pem

# ---------------------------------------------------------------------------
# Synthetic anomaly injection helpers
# ---------------------------------------------------------------------------

def _anomaly_self_signed_weak() -> np.ndarray:
    """Very weak, self-signed, short-serial cert with no SAN."""
    from tests.certgen import make_cert, pem, new_key
    from cryptography.hazmat.primitives import hashes
    cert, _ = make_cert(
        cn="bad.internal", san=None, key=new_key("rsa1024"),
        days=3650, serial=3, eku_server=False, ocsp=False, crl=False,
        sign_hash=hashes.SHA1,
    )
    _, d = parse_cert_from_pem(pem(cert))
    return extract_feature_vector(d)


def _anomaly_extreme_san() -> np.ndarray:
    """Certificate with 500 SAN entries."""
    from tests.certgen import make_cert, pem
    names = [f"host{i}.example.com" for i in range(500)]
    cert, _ = make_cert(cn=names[0], san=names, days=90, serial=None)
    _, d = parse_cert_from_pem(pem(cert))
    return extract_feature_vector(d)


def _anomaly_unrecognized_extensions() -> np.ndarray:
    """Certificate loaded with many unrecognized extensions."""
    from tests.certgen import make_cert, pem
    cert, _ = make_cert(extra_unrecognized=10)
    _, d = parse_cert_from_pem(pem(cert))
    return extract_feature_vector(d)


def _anomaly_long_lifetime() -> np.ndarray:
    """10-year certificate."""
    from tests.certgen import make_cert, pem
    cert, _ = make_cert(days=3650)
    _, d = parse_cert_from_pem(pem(cert))
    return extract_feature_vector(d)


def _normal_typical() -> np.ndarray:
    """Modern, typical publicly trusted TLS certificate."""
    from tests.certgen import make_cert, pem
    cert, _ = make_cert(days=90)
    _, d = parse_cert_from_pem(pem(cert))
    return extract_feature_vector(d)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Validate anomaly models with injected anomalies")
    ap.add_argument("--corpus", help="directory with .pem/.crt/.cer/.der files")
    ap.add_argument("--n-normal", type=int, default=500,
                    help="size of the synthetic normal set when no corpus is given")
    ap.add_argument("--n-anomalies", type=int, default=40,
                    help="number of synthetic anomaly instances to inject per anomaly type")
    args = ap.parse_args()

    # ----- reference population -----
    print("Building reference set …", flush=True)
    if args.corpus:
        X_ref = load_corpus(args.corpus)
        if len(X_ref) < MIN_CORPUS:
            print(f"  corpus has only {len(X_ref)} certs (< {MIN_CORPUS}); falling back to synthetic baseline.")
            X_ref = synthetic_baseline(args.n_normal)
        else:
            print(f"  loaded {len(X_ref)} certificates from {args.corpus}")
    else:
        X_ref = synthetic_baseline(args.n_normal)
        print(f"  generated {len(X_ref)} synthetic normal certificates")

    # ----- inject anomalies -----
    print("Injecting anomaly instances …", flush=True)
    anomaly_factories = [
        ("self-signed-weak",          _anomaly_self_signed_weak),
        ("extreme-san-count",         _anomaly_extreme_san),
        ("many-unrecognized-exts",    _anomaly_unrecognized_extensions),
        ("10-year-lifetime",          _anomaly_long_lifetime),
    ]
    X_anomaly_parts = []
    for name, factory in anomaly_factories:
        rows = []
        for _ in range(args.n_anomalies):
            try:
                rows.append(factory())
            except Exception as exc:
                print(f"  WARNING: {name} factory raised {exc!r}", file=sys.stderr)
        if rows:
            X_anomaly_parts.append(np.asarray(rows))
            print(f"  injected {len(rows)} × {name}")

    if not X_anomaly_parts:
        print("ERROR: no anomaly instances were generated.", file=sys.stderr)
        sys.exit(1)

    X_anomaly = np.vstack(X_anomaly_parts)

    # Also inject a few of the "normal" (real cert) samples for balance
    X_normal_test = np.asarray([_normal_typical() for _ in range(min(80, len(X_ref)))])

    X_all = np.vstack([X_normal_test, X_anomaly])
    y_all = np.array([0] * len(X_normal_test) + [1] * len(X_anomaly))

    # ----- train Isolation Forest -----
    print("\nFitting Isolation Forest …", flush=True)
    iso = IsolationForest(n_estimators=200, contamination="auto", random_state=42, n_jobs=1)
    iso.fit(X_ref)
    # score_samples: lower = more anomalous
    iso_scores = iso.score_samples(X_all)
    iso_pred = -iso_scores  # higher = more anomalous

    # ----- LOF (novelty=True → trained on reference, applied to test) -----
    print("Fitting Local Outlier Factor …", flush=True)
    # LOF novelty mode: fit on reference, predict on test set
    lof = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination="auto")
    lof.fit(X_ref)
    lof_scores = lof.score_samples(X_all)
    lof_pred = -lof_scores  # higher = more anomalous

    # ----- ROC-AUC -----
    print("\n--- Results ---")
    try:
        iso_auc = roc_auc_score(y_all, iso_pred)
        lof_auc = roc_auc_score(y_all, lof_pred)
        print(f"  Isolation Forest  ROC-AUC: {iso_auc:.4f}")
        print(f"  Local Outlier Factor ROC-AUC: {lof_auc:.4f}")
    except ValueError as exc:
        print(f"  Could not compute ROC-AUC: {exc}", file=sys.stderr)
        iso_auc = lof_auc = float("nan")

    # ----- Per-model threshold-based summary -----
    # Use the same ML_OUTLIER_THRESHOLD from the production model by applying
    # it to percentile-converted scores.
    def _percentile_scores(raw: np.ndarray, ref: np.ndarray) -> np.ndarray:
        sorted_ref = np.sort(raw[:len(ref)])
        idx = np.searchsorted(sorted_ref, raw, side="left")
        return idx / max(1, len(sorted_ref))

    iso_ref_scores = np.sort(iso.score_samples(X_ref))
    lof_ref_scores = np.sort(lof.score_samples(X_ref))

    def _pct(raw_score: float, sorted_ref_scores: np.ndarray) -> float:
        idx = np.searchsorted(sorted_ref_scores, raw_score, side="left")
        return 1.0 - idx / len(sorted_ref_scores)

    iso_pcts = np.array([_pct(s, iso_ref_scores) for s in iso.score_samples(X_all)])
    lof_pcts = np.array([_pct(s, lof_ref_scores) for s in lof.score_samples(X_all)])

    for model_name, pcts in [("IsolationForest", iso_pcts), ("LOF", lof_pcts)]:
        pred_labels = (pcts >= ML_OUTLIER_THRESHOLD).astype(int)
        tp = int(np.sum((pred_labels == 1) & (y_all == 1)))
        fp = int(np.sum((pred_labels == 1) & (y_all == 0)))
        fn = int(np.sum((pred_labels == 0) & (y_all == 1)))
        tn = int(np.sum((pred_labels == 0) & (y_all == 0)))
        precision = tp / max(1, tp + fp)
        recall    = tp / max(1, tp + fn)
        f1        = 2 * precision * recall / max(1e-9, precision + recall)
        print(f"\n  {model_name} @ threshold {ML_OUTLIER_THRESHOLD:.0%}:")
        print(f"    TP={tp} FP={fp} FN={fn} TN={tn}")
        print(f"    Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}")

    print(f"\n  → Recommended model: {'LOF' if lof_auc > iso_auc else 'IsolationForest'} "
          f"(higher AUC: {max(iso_auc, lof_auc):.4f})")
    print("  (The production model uses IsolationForest; LOF is compared here as an alternative.)")


if __name__ == "__main__":
    main()
