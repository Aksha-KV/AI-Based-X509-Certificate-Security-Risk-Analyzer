"""Train the anomaly model on a directory of real certificates.

  python scripts/train_model.py --corpus data/corpus
Without --corpus (or with fewer than 200 certificates) it trains the synthetic placeholder baseline.
"""
import _path  # noqa: F401
import argparse

from analyzer.ml_model import DEFAULT_MODEL_PATH, MIN_CORPUS, CertificateAnomalyModel, load_corpus

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", help="directory with .pem/.crt/.cer/.der files (searched recursively)")
ap.add_argument("--out", default=DEFAULT_MODEL_PATH)
args = ap.parse_args()

if args.corpus:
    n = len(load_corpus(args.corpus))
    print(f"parsed {n} certificates from {args.corpus}")
    if n < MIN_CORPUS:
        print(f"WARNING: fewer than {MIN_CORPUS} certificates - falling back to the synthetic baseline.")
model = CertificateAnomalyModel(model_path=args.out, corpus_dir=args.corpus, retrain=True)
print("trained:", {k: v for k, v in model.meta.items() if k != "feature_names"})
