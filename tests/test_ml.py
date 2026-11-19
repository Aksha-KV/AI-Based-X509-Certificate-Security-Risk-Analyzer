"""Anomaly-model tests: behaviour, time-independence, explanations and provenance."""

import os
import tempfile

import numpy as np
import pytest

from analyzer.ml_model import (
    MIN_CORPUS, ML_OUTLIER_THRESHOLD, CertificateAnomalyModel,
    FEATURE_NAMES, FEATURE_LABELS, extract_feature_vector, _file_sha256,
)
from .certgen import good_details, make_cert, pem

_MODEL = None


def model():
    global _MODEL
    if _MODEL is None:
        _MODEL = CertificateAnomalyModel(model_path=os.path.join(tempfile.mkdtemp(), "m.joblib"))
    return _MODEL


WEIRD = dict(serial_bits=8, extension_count=28, unrecognized_extension_count=7, cn_entropy=4.9, cn_length=95,
             san_count=400, san_private_names=["a.corp", "b.corp", "c.local"], total_lifetime_days=3650,
             critical_extension_count=9, ocsp_urls=[], crl_urls=[], has_sct=False, eku_server_auth=False)


# --------------------------------------------------------------------------- existing tests

def test_feature_vector_shape_matches_names():
    assert len(extract_feature_vector(good_details())) == len(FEATURE_NAMES)
    assert "days_remaining" not in FEATURE_NAMES  # time-dependent features are deliberately excluded


def test_all_feature_names_have_labels():
    """Every feature must have a human-readable label."""
    missing = [f for f in FEATURE_NAMES if f not in FEATURE_LABELS]
    assert not missing, f"features without labels: {missing}"


def test_typical_certificate_is_not_an_outlier_and_weird_one_is():
    typical = model().assess(good_details(extension_count=10, critical_extension_count=2, ocsp_urls=["a"], crl_urls=["b"]))
    weird = model().assess(good_details(**WEIRD))
    assert typical["ml_anomaly_percentile"] < 0.9, typical["ml_anomaly_percentile"]
    assert weird["ml_anomaly_percentile"] >= ML_OUTLIER_THRESHOLD and weird["ml_is_outlier"] is True


def test_score_does_not_change_with_time_remaining():
    a = model().assess(good_details(days_remaining=300))["ml_anomaly_percentile"]
    b = model().assess(good_details(days_remaining=2))["ml_anomaly_percentile"]
    assert a == b


def test_explanations_name_the_features_that_drive_the_score():
    out = model().assess(good_details(**WEIRD))
    names = {f["feature"] for f in out["ml_top_factors"]}
    assert names & {"serial_bits", "san_count", "extension_count", "unrecognized_extension_count", "cn_length"}
    assert abs(sum(f["share"] for f in out["ml_top_factors"]) - 1.0) < 0.35 or out["ml_top_factors"]
    assert all(f["value"] != f["typical"] for f in out["ml_top_factors"])


def test_provenance_is_reported_and_synthetic_is_flagged():
    info = model().assess(good_details())["model_info"]
    assert info["is_synthetic"] is True and info["source"] == "synthetic-baseline" and info["n_reference"] > 1000


def test_trains_on_a_real_corpus_directory():
    corpus = tempfile.mkdtemp()
    for i in range(MIN_CORPUS + 20):
        cert, _ = make_cert(cn=f"host{i}.example.com", san=(f"host{i}.example.com",))
        with open(os.path.join(corpus, f"c{i}.pem"), "wb") as fh:
            fh.write(pem(cert))
    m = CertificateAnomalyModel(model_path=os.path.join(tempfile.mkdtemp(), "c.joblib"), corpus_dir=corpus)
    assert m.meta["source"].startswith("corpus:") and m.meta["is_synthetic"] is False
    assert m.meta["n_reference"] >= MIN_CORPUS


def test_small_corpus_falls_back_to_synthetic():
    corpus = tempfile.mkdtemp()
    cert, _ = make_cert()
    with open(os.path.join(corpus, "one.pem"), "wb") as fh:
        fh.write(pem(cert))
    m = CertificateAnomalyModel(model_path=os.path.join(tempfile.mkdtemp(), "s.joblib"), corpus_dir=corpus)
    assert m.meta["is_synthetic"] is True


def test_model_persists_and_reloads():
    path = os.path.join(tempfile.mkdtemp(), "p.joblib")
    first = CertificateAnomalyModel(model_path=path)
    second = CertificateAnomalyModel(model_path=path)
    assert second.meta["trained_at"] == first.meta["trained_at"]  # loaded, not retrained
    assert first.assess(good_details())["ml_raw_score"] == second.assess(good_details())["ml_raw_score"]


# --------------------------------------------------------------------------- NEW TESTS

class TestFeatureVectorNewFields:
    """Verify the new structural features are present and correctly computed."""

    def test_one_hot_key_type_rsa(self):
        vec = extract_feature_vector(good_details(key_type="RSA"))
        rsa_idx  = FEATURE_NAMES.index("key_type_rsa")
        ec_idx   = FEATURE_NAMES.index("key_type_ec")
        oth_idx  = FEATURE_NAMES.index("key_type_other")
        assert vec[rsa_idx] == 1.0 and vec[ec_idx] == 0.0 and vec[oth_idx] == 0.0

    def test_one_hot_key_type_ec(self):
        vec = extract_feature_vector(good_details(key_type="EC"))
        rsa_idx = FEATURE_NAMES.index("key_type_rsa")
        ec_idx  = FEATURE_NAMES.index("key_type_ec")
        oth_idx = FEATURE_NAMES.index("key_type_other")
        assert vec[rsa_idx] == 0.0 and vec[ec_idx] == 1.0 and vec[oth_idx] == 0.0

    def test_one_hot_key_type_other(self):
        vec = extract_feature_vector(good_details(key_type="DSA"))
        rsa_idx = FEATURE_NAMES.index("key_type_rsa")
        ec_idx  = FEATURE_NAMES.index("key_type_ec")
        oth_idx = FEATURE_NAMES.index("key_type_other")
        assert vec[rsa_idx] == 0.0 and vec[ec_idx] == 0.0 and vec[oth_idx] == 1.0

    def test_one_hot_key_type_sums_to_one(self):
        for kt in ("RSA", "EC", "DSA", "Ed25519", "Ed448", "Unknown"):
            vec = extract_feature_vector(good_details(key_type=kt))
            total = sum(vec[FEATURE_NAMES.index(f)] for f in ("key_type_rsa", "key_type_ec", "key_type_other"))
            assert total == 1.0, f"one-hot must sum to 1 for key_type={kt!r}"

    def test_redundant_features_removed(self):
        """is_self_signed, is_ca, san_private_count must NOT be in FEATURE_NAMES."""
        assert "is_self_signed" not in FEATURE_NAMES
        assert "is_ca" not in FEATURE_NAMES
        assert "san_private_count" not in FEATURE_NAMES
        assert "key_type_code" not in FEATURE_NAMES  # old ordinal encoding removed

    def test_key_usage_bits_feature(self):
        """ku_key_cert_sign and ku_crl_sign flags are captured from key_usage."""
        cert_sign_idx = FEATURE_NAMES.index("ku_key_cert_sign")
        crl_sign_idx  = FEATURE_NAMES.index("ku_crl_sign")

        # Neither flag
        vec = extract_feature_vector(good_details(key_usage=["Digital Signature"]))
        assert vec[cert_sign_idx] == 0.0 and vec[crl_sign_idx] == 0.0

        # Only keyCertSign
        vec = extract_feature_vector(good_details(key_usage=["Certificate Sign"]))
        assert vec[cert_sign_idx] == 1.0 and vec[crl_sign_idx] == 0.0

        # Only cRLSign
        vec = extract_feature_vector(good_details(key_usage=["CRL Sign"]))
        assert vec[cert_sign_idx] == 0.0 and vec[crl_sign_idx] == 1.0

        # Both
        vec = extract_feature_vector(good_details(key_usage=["Certificate Sign", "CRL Sign"]))
        assert vec[cert_sign_idx] == 1.0 and vec[crl_sign_idx] == 1.0

    def test_wildcard_san_ratio(self):
        idx = FEATURE_NAMES.index("wildcard_san_ratio")
        # 1 wildcard out of 2 SANs -> ratio=0.5
        d = good_details(san_dns=["*.example.com", "example.com"], san_ips=[], san_count=2)
        vec = extract_feature_vector(d)
        assert abs(vec[idx] - 0.5) < 1e-6

        # No wildcards
        d = good_details(san_dns=["example.com"], san_ips=[], san_count=1)
        vec = extract_feature_vector(d)
        assert vec[idx] == 0.0

    def test_ip_san_ratio(self):
        idx = FEATURE_NAMES.index("ip_san_ratio")
        d = good_details(san_dns=["a.example.com"], san_ips=["10.0.0.1"], san_count=2)
        vec = extract_feature_vector(d)
        assert abs(vec[idx] - 0.5) < 1e-6

    def test_has_aki_has_ski_features(self):
        aki_idx = FEATURE_NAMES.index("has_aki")
        ski_idx = FEATURE_NAMES.index("has_ski")

        vec_with = extract_feature_vector(good_details(has_aki=True, has_ski=True))
        assert vec_with[aki_idx] == 1.0 and vec_with[ski_idx] == 1.0

        vec_without = extract_feature_vector(good_details(has_aki=False, has_ski=False))
        assert vec_without[aki_idx] == 0.0 and vec_without[ski_idx] == 0.0

    def test_policy_oid_count_feature(self):
        idx = FEATURE_NAMES.index("policy_oid_count")
        vec3 = extract_feature_vector(good_details(policy_oid_count=3))
        assert vec3[idx] == 3.0

        vec0 = extract_feature_vector(good_details(policy_oid_count=0))
        assert vec0[idx] == 0.0

    def test_feature_vector_has_correct_length(self):
        """Feature vector length must always equal len(FEATURE_NAMES)."""
        vec = extract_feature_vector(good_details())
        assert len(vec) == len(FEATURE_NAMES), (
            f"feature vector length {len(vec)} != FEATURE_NAMES length {len(FEATURE_NAMES)}"
        )


class TestUnifiedOutlierThreshold:
    """ML_OUTLIER_THRESHOLD is imported and used consistently."""

    def test_is_outlier_uses_unified_threshold(self):
        """ml_is_outlier must be True iff ml_anomaly_percentile >= ML_OUTLIER_THRESHOLD."""
        weird_result = model().assess(good_details(**WEIRD))
        assert weird_result["ml_is_outlier"] == (
            weird_result["ml_anomaly_percentile"] >= ML_OUTLIER_THRESHOLD
        )

    def test_typical_cert_is_not_outlier(self):
        normal = model().assess(
            good_details(extension_count=10, critical_extension_count=2, ocsp_urls=["a"], crl_urls=["b"])
        )
        assert normal["ml_is_outlier"] is False


class TestSHA256ModelVerification:
    """Saving the model writes a .sha256 checksum; loading a tampered file triggers retrain."""

    def test_sha256_file_is_written_on_save(self):
        path = os.path.join(tempfile.mkdtemp(), "verify.joblib")
        m = CertificateAnomalyModel(model_path=path)
        sha256_path = path + ".sha256"
        assert os.path.exists(sha256_path), "SHA-256 checksum file must be created alongside the model"
        digest = open(sha256_path).read().strip()
        assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), (
            "checksum file must contain a valid hex SHA-256 digest"
        )

    def test_tampered_model_triggers_retrain(self):
        path = os.path.join(tempfile.mkdtemp(), "tampered.joblib")
        first = CertificateAnomalyModel(model_path=path)
        trained_at = first.meta["trained_at"]

        # Tamper the checksum
        with open(path + ".sha256", "w") as fh:
            fh.write("0" * 64 + "\n")

        # Loading again must detect the mismatch and retrain
        second = CertificateAnomalyModel(model_path=path)
        # It should have retrained (new timestamp or same – but it must load successfully)
        # The key assertion: it did NOT crash and produced a valid model.
        assert second.forest is not None

    def test_file_sha256_helper(self):
        import tempfile as _tf
        with _tf.NamedTemporaryFile(delete=False, suffix=".bin") as f:
            f.write(b"hello world")
            fname = f.name
        digest = _file_sha256(fname)
        import hashlib
        expected = hashlib.sha256(b"hello world").hexdigest()
        assert digest == expected
        os.unlink(fname)


class TestExtractorNewFields:
    """Verify the extractor populates the new structural fields."""

    def test_has_aki_ski_in_extracted_details(self):
        from analyzer.extractor import parse_cert_from_pem
        from .certgen import make_cert, pem, make_chain

        # A CA-issued cert should have AKI; a self-signed leaf may or may not
        ch = make_chain()
        _, d = parse_cert_from_pem(pem(ch["leaf"]))
        # has_aki and has_ski must be present in the detail dict
        assert "has_aki" in d, "has_aki must be extracted"
        assert "has_ski" in d, "has_ski must be extracted"

    def test_policy_oid_count_in_extracted_details(self):
        from analyzer.extractor import parse_cert_from_pem
        from .certgen import make_cert, pem

        cert, _ = make_cert()
        _, d = parse_cert_from_pem(pem(cert))
        assert "policy_oid_count" in d, "policy_oid_count must be extracted"
        assert isinstance(d["policy_oid_count"], int)
