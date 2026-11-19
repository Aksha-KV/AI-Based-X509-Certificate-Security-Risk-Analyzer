"""
Unsupervised anomaly detection for X.509 certificates.

What this is (and is not)
-------------------------
* An Isolation Forest learns what a *typical* certificate looks like from a
  reference set and scores how unusual a new certificate is. It does NOT predict
  "probability of compromise" and it is not trained on labelled attacks.
* Features are structural and time-independent (no "days remaining"), so the same
  certificate gets the same anomaly score tomorrow.
* The reference set is either
    - a real corpus of PEM/DER files you collected (recommended; see
      scripts/scan_batch.py --save-pem and scripts/train_model.py), or
    - a synthetic baseline of well-formed modern TLS certificates (fallback).
  The provenance is stored in the model and shown in the UI. A synthetic baseline
  is a placeholder, not evidence.
* The anomaly score is the percentile of the reference set that is *more normal*
  than the certificate (0.97 = more unusual than 97% of reference certificates).
* Every score comes with the features that drove it (occlusion attribution:
  replace one feature with the reference median and measure how much more normal
  the certificate becomes).

ML improvements (v2)
--------------------
* Removed redundant rule-based proxy features that duplicate deterministic rules
  (``is_self_signed``, ``is_ca``, ``san_private_count``).
* Added structural features that the rules do not already penalise:
  - key-usage bits (key_cert_sign, crl_sign)
  - SAN type distribution (ip_san_ratio, wildcard_san_ratio)
  - AKI / SKI presence flags (structural correctness indicator)
  - policy OID count (certificate policy richness)
  - one-hot encoded key type (rsa / ec / other) instead of a single ordinal
* Unified the outlier-threshold constant ``ML_OUTLIER_THRESHOLD`` and the
  ``ML_EXPLAIN_THRESHOLD`` in report_view.py both derive from it.
"""

from __future__ import annotations

import datetime
import glob
import hashlib
import os
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import IsolationForest

from .extractor import parse_cert_from_pem

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "cert_anomaly_model.joblib")

# Unified outlier threshold: a certificate is an outlier when its anomaly
# percentile is at or above this value.  report_view.py imports this constant
# to avoid having two independent thresholds diverge over time.
ML_OUTLIER_THRESHOLD = 0.95

FEATURE_NAMES = [
    # --- key type (one-hot, three buckets: RSA / EC / other) ---
    "key_type_rsa",
    "key_type_ec",
    "key_type_other",
    # --- key strength ---
    "security_strength_bits",
    "rsa_exponent_log2",
    # --- signature ---
    "sig_hash_bits",
    # --- lifetime ---
    "lifetime_days",
    # --- SAN ---
    "san_count",
    "has_san",
    "wildcard_san_ratio",   # fraction of SAN entries that are wildcards
    "ip_san_ratio",         # fraction of SAN entries that are IP addresses
    # --- extensions (structural) ---
    "extension_count",
    "critical_extension_count",
    "unrecognized_extension_count",
    "has_aki",              # AuthorityKeyIdentifier present
    "has_ski",              # SubjectKeyIdentifier present
    "policy_oid_count",     # number of CertificatePolicy OIDs
    # --- key usage bits (structural, NOT rule-equivalent) ---
    "ku_key_cert_sign",     # keyCertSign present (structural flag)
    "ku_crl_sign",          # cRLSign present
    # --- serial ---
    "serial_bits",
    # --- CN entropy / length (unchanged) ---
    "cn_entropy",
    "cn_length",
    # --- EKU / revocation / SCT ---
    "has_eku_server_auth",
    "revocation_url_count",
    "has_sct",
    # --- subject ---
    "subject_field_count",
]

FEATURE_LABELS = {
    "key_type_rsa":              "Key type: RSA",
    "key_type_ec":               "Key type: EC",
    "key_type_other":            "Key type: other (DSA / Ed*)",
    "security_strength_bits":    "Key security strength (bits)",
    "rsa_exponent_log2":         "RSA exponent size (log2)",
    "sig_hash_bits":             "Signature hash size (bits)",
    "lifetime_days":             "Validity period (days)",
    "san_count":                 "Number of SAN entries",
    "has_san":                   "Has SAN extension",
    "wildcard_san_ratio":        "Fraction of SANs that are wildcards",
    "ip_san_ratio":              "Fraction of SANs that are IP addresses",
    "extension_count":           "Number of extensions",
    "critical_extension_count":  "Number of critical extensions",
    "unrecognized_extension_count": "Unrecognised extensions",
    "has_aki":                   "Has AuthorityKeyIdentifier",
    "has_ski":                   "Has SubjectKeyIdentifier",
    "policy_oid_count":          "Number of certificate policy OIDs",
    "ku_key_cert_sign":          "KeyUsage: keyCertSign present",
    "ku_crl_sign":               "KeyUsage: cRLSign present",
    "serial_bits":               "Serial number length (bits)",
    "cn_entropy":                "Common-name randomness (entropy)",
    "cn_length":                 "Common-name length",
    "has_eku_server_auth":       "Has serverAuth EKU",
    "revocation_url_count":      "OCSP/CRL pointers",
    "has_sct":                   "Has embedded CT SCT",
    "subject_field_count":       "Subject fields",
}

_HASH_BITS = {"MD2": 128, "MD5": 128, "SHA1": 160, "SHA224": 224, "SHA256": 256,
              "SHA384": 384, "SHA512": 512, "SHA3_256": 256, "SHA3_384": 384, "SHA3_512": 512}
MIN_CORPUS = 200


def extract_feature_vector(d: Dict[str, Any]) -> np.ndarray:
    exp = d.get("rsa_exponent") or 0
    ktype = d.get("key_type", "")
    san_count = d.get("san_count", 0)
    san_dns   = d.get("san_dns",  []) or []
    san_ips   = d.get("san_ips",  []) or []
    san_all   = san_count or len(san_dns) + len(san_ips)

    wildcard_count = sum(1 for s in san_dns if s.startswith("*."))
    wildcard_ratio = wildcard_count / max(1, san_all)
    ip_ratio       = len(san_ips) / max(1, san_all)

    ku = d.get("key_usage") or []
    has_cert_sign = int(any(label in ku for label in
                            ("Certificate Sign", "keyCertSign", "key_cert_sign")))
    has_crl_sign  = int(any(label in ku for label in
                            ("CRL Sign", "cRLSign", "crl_sign")))

    vec = [
        # one-hot key type
        1 if ktype == "RSA"              else 0,
        1 if ktype == "EC"               else 0,
        1 if ktype not in ("RSA", "EC")  else 0,
        # key strength
        d.get("security_strength_bits", 0),
        float(np.log2(exp)) if exp > 0 else 0.0,
        # sig hash
        _HASH_BITS.get((d.get("signature_hash") or "").upper(), 0),
        # lifetime
        d.get("total_lifetime_days", 0),
        # SAN
        san_count,
        1 if d.get("san_list") else 0,
        wildcard_ratio,
        ip_ratio,
        # extensions
        d.get("extension_count", 0),
        d.get("critical_extension_count", 0),
        d.get("unrecognized_extension_count", 0),
        1 if d.get("has_aki") else 0,
        1 if d.get("has_ski") else 0,
        d.get("policy_oid_count", 0),
        # key usage bits
        has_cert_sign,
        has_crl_sign,
        # serial
        d.get("serial_bits", 0),
        # CN
        d.get("cn_entropy", 0.0),
        d.get("cn_length", 0),
        # EKU / revocation / SCT
        1 if d.get("eku_server_auth") else 0,
        len(d.get("ocsp_urls", [])) + len(d.get("crl_urls", [])),
        1 if d.get("has_sct") else 0,
        # subject
        d.get("subject_field_count", 0),
    ]
    return np.asarray(vec, dtype=np.float64)


def synthetic_baseline(n: int = 4000, seed: int = 42) -> np.ndarray:
    """
    Placeholder reference population: well-formed, modern publicly trusted TLS leaf
    certificates. Distributions are rough, hand-written approximations of what CAs issue.
    Replace with a real corpus for any serious evaluation.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        ktype = rng.choice(["RSA", "EC"], p=[0.62, 0.38])
        is_rsa = ktype == "RSA"
        is_ec  = ktype == "EC"
        is_oth = not (is_rsa or is_ec)
        if is_rsa:
            size = rng.choice([2048, 3072, 4096], p=[0.85, 0.03, 0.12])
            strength = {2048: 112, 3072: 128, 4096: 128}[int(size)]
            exp_log2 = np.log2(65537)
        else:
            size = rng.choice([256, 384], p=[0.92, 0.08])
            strength = 128 if size == 256 else 192
            exp_log2 = 0.0
        hash_bits = rng.choice([256, 384], p=[0.9, 0.1])
        lifetime = rng.choice([47, 90, 100, 180, 200, 365, 397], p=[0.02, 0.45, 0.03, 0.05, 0.1, 0.1, 0.25])
        san = int(np.clip(rng.lognormal(0.8, 0.9), 1, 60))
        wc_ratio = float(rng.random() < 0.15) / max(1, san)
        ip_ratio = float(np.clip(rng.binomial(san, 0.03), 0, san)) / max(1, san)
        ext_count = int(rng.integers(8, 13))
        crit = int(rng.choice([2, 3], p=[0.6, 0.4]))
        serial_bits = int(rng.choice([128, 144, 152, 159], p=[0.3, 0.3, 0.2, 0.2]))
        cn_len = int(np.clip(rng.normal(16, 6), 4, 50))
        cn_ent = float(np.clip(rng.normal(3.2, 0.35), 2.0, 4.2))
        rev = int(rng.choice([0, 1, 2, 3], p=[0.03, 0.3, 0.3, 0.37]))
        rows.append([
            int(is_rsa), int(is_ec), int(is_oth),
            strength, exp_log2, hash_bits,
            lifetime, san, 1,
            wc_ratio, ip_ratio,
            ext_count, crit, 0,
            int(rng.random() < 0.90),   # has_aki
            int(rng.random() < 0.80),   # has_ski
            int(rng.choice([0, 1, 2], p=[0.4, 0.4, 0.2])),  # policy_oid_count
            0, 0,                        # ku_key_cert_sign, ku_crl_sign
            serial_bits, cn_ent, cn_len,
            1,                           # has_eku_server_auth
            rev,
            int(rng.random() < 0.95),   # has_sct
            int(rng.choice([1, 2, 3, 5], p=[0.55, 0.15, 0.1, 0.2])),  # subject_field_count
        ])
    return np.asarray(rows, dtype=np.float64)


def load_corpus(corpus_dir: str) -> np.ndarray:
    """Build a feature matrix from every parseable certificate file in a directory."""
    rows = []
    patterns = ("*.pem", "*.crt", "*.cer", "*.der")
    paths = sorted({p for pat in patterns for p in glob.glob(os.path.join(corpus_dir, "**", pat), recursive=True)})
    for path in paths:
        try:
            with open(path, "rb") as fh:
                _, details = parse_cert_from_pem(fh.read())
            rows.append(extract_feature_vector(details))
        except Exception:
            continue
    return np.asarray(rows, dtype=np.float64) if rows else np.empty((0, len(FEATURE_NAMES)))


class CertificateAnomalyModel:
    """Isolation-Forest anomaly detector with provenance and per-certificate explanations."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, corpus_dir: Optional[str] = None,
                 retrain: bool = False):
        self.model_path = model_path
        self.corpus_dir = corpus_dir
        self.forest: Optional[IsolationForest] = None
        self.train_scores: np.ndarray = np.array([])
        self.medians: np.ndarray = np.array([])
        self.meta: Dict[str, Any] = {}
        if retrain or not self._load():
            self.train()

    # ------------------------------------------------------------------ #
    def _load(self) -> bool:
        if not os.path.exists(self.model_path):
            return False
        try:
            # --- Security: verify SHA-256 checksum before loading pickle ---
            sha256_path = self.model_path + ".sha256"
            if os.path.exists(sha256_path):
                expected = open(sha256_path).read().strip()
                actual = _file_sha256(self.model_path)
                if actual != expected:
                    return False  # file tampered; retrain
            bundle = joblib.load(self.model_path)
            meta = bundle["meta"]
            if meta.get("feature_names") != FEATURE_NAMES or meta.get("sklearn_version") != sklearn.__version__:
                return False  # stale model: retrain rather than trust an incompatible pickle
            if self.corpus_dir and meta.get("source") == "synthetic-baseline":
                return False
            self.forest, self.train_scores = bundle["forest"], bundle["train_scores"]
            self.medians, self.meta = bundle["medians"], meta
            return True
        except Exception:
            return False

    def train(self, X: Optional[np.ndarray] = None, source: Optional[str] = None) -> Dict[str, Any]:
        if X is None:
            corpus = load_corpus(self.corpus_dir) if self.corpus_dir and os.path.isdir(self.corpus_dir) else None
            if corpus is not None and len(corpus) >= MIN_CORPUS:
                X, source = corpus, f"corpus:{len(corpus)} certificates from {self.corpus_dir}"
            else:
                X, source = synthetic_baseline(), "synthetic-baseline"
        source = source or "custom"

        self.forest = IsolationForest(n_estimators=200, contamination="auto", random_state=42, n_jobs=1)
        self.forest.fit(X)
        self.train_scores = np.sort(self.forest.score_samples(X))
        self.medians = np.median(X, axis=0)
        self.meta = {
            "source": source,
            "is_synthetic": source == "synthetic-baseline",
            "n_reference": int(len(X)),
            "feature_names": FEATURE_NAMES,
            "sklearn_version": sklearn.__version__,
            "trained_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        }
        try:
            joblib.dump({"forest": self.forest, "train_scores": self.train_scores,
                         "medians": self.medians, "meta": self.meta}, self.model_path)
            # Write SHA-256 checksum alongside the model file
            checksum = _file_sha256(self.model_path)
            with open(self.model_path + ".sha256", "w") as fh:
                fh.write(checksum + "\n")
        except OSError:
            pass  # read-only deployment: keep the in-memory model
        return self.meta

    # ------------------------------------------------------------------ #
    def _percentile(self, raw_scores: np.ndarray) -> np.ndarray:
        """Fraction of the reference set that is more normal than each score."""
        idx = np.searchsorted(self.train_scores, raw_scores, side="left")
        return 1.0 - idx / len(self.train_scores)

    def assess(self, cert_details: Dict[str, Any], top_k: int = 5) -> Dict[str, Any]:
        x = extract_feature_vector(cert_details)
        # Row 0 is the certificate; rows 1..n replace one feature at a time with the reference median.
        matrix = np.tile(x, (len(x) + 1, 1))
        for i in range(len(x)):
            matrix[i + 1, i] = self.medians[i]
        raw = self.forest.score_samples(matrix)
        base = raw[0]
        anomaly = float(self._percentile(np.array([base]))[0])

        gains = raw[1:] - base  # >0: replacing the feature makes the cert look more normal
        positive = np.clip(gains, 0, None)
        total = positive.sum()
        factors: List[Dict[str, Any]] = []
        if total > 0:
            for i in np.argsort(-positive)[:top_k]:
                if positive[i] <= 0 or x[i] == self.medians[i]:
                    continue
                factors.append({
                    "feature": FEATURE_NAMES[i],
                    "label": FEATURE_LABELS[FEATURE_NAMES[i]],
                    "value": round(float(x[i]), 3),
                    "typical": round(float(self.medians[i]), 3),
                    "share": round(float(positive[i] / total), 3),
                })
        return {
            "ml_anomaly_percentile": round(anomaly, 4),
            "ml_is_outlier": anomaly >= ML_OUTLIER_THRESHOLD,
            "ml_raw_score": round(float(base), 4),
            "ml_top_factors": factors,
            "model_info": {k: v for k, v in self.meta.items() if k != "feature_names"},
        }


def _file_sha256(path: str) -> str:
    """Return the hex-encoded SHA-256 digest of the file at *path*."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
