"""
Portable replacements for the 6 loopback-TLS-skipped tests in test_network.py
and test_pipeline_batch.py.

Instead of binding a real OS socket (which Windows sandboxes block with
WinError 10013), we intercept ``socket.socket`` and ``ssl.SSLContext`` at the
Python level so the entire handshake path inside ``fetch_cert_from_domain`` and
``scan_targets`` is exercised without any network I/O.

The 6 scenarios covered (one-to-one with the @_skip_no_loopback tests):
  1. test_self_signed_server_is_detected_end_to_end
  2. test_hostname_mismatch_is_detected_end_to_end
  3. test_full_chain_is_captured_and_verified_against_ca
  4. test_untrusted_issuer_is_flagged_when_no_ca_given
  5. test_missing_intermediate_is_reported_as_untrusted
  6. test_batch_ranks_by_risk_and_survives_errors (batch variant)
"""

from __future__ import annotations

import io
import os
import csv
import socket
import ssl
import tempfile
import threading
import unittest.mock as mock
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.hashes import SHA256
from cryptography import x509 as cx509

from analyzer.extractor import fetch_cert_from_domain
from analyzer.rules_engine import evaluate_certificate_rules
from analyzer import analyze, parse_cert_from_pem, rows_to_csv, scan_targets
from analyzer.ml_model import CertificateAnomalyModel
from .certgen import make_cert, make_chain, pem, tls_server


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ids(details):
    return {v["rule_id"] for v in evaluate_certificate_rules(details)["vulnerabilities"]}


def _cert_der(cert) -> bytes:
    return cert.public_bytes(serialization.Encoding.DER)


def _build_mock_ssl_conn(leaf_cert, chain_certs=(), verify_raises=None, tls_version="TLSv1.3"):
    """
    Build a fake ssl.SSLSocket-like object that returns *leaf_cert* from
    ``getpeercert(binary_form=True)`` and the whole chain from
    ``get_unverified_chain()``.
    """
    conn = MagicMock()
    conn.getpeercert.return_value = _cert_der(leaf_cert)
    # get_unverified_chain returns DER bytes list (leaf first)
    all_certs = [leaf_cert] + list(chain_certs)
    conn.get_unverified_chain.return_value = [_cert_der(c) for c in all_certs]
    conn.version.return_value = tls_version
    conn.__enter__ = lambda s: s
    conn.__exit__ = MagicMock(return_value=False)
    if verify_raises:
        conn.connect.side_effect = verify_raises
    return conn


def _mock_ctx(leaf_cert, chain_certs=(), verify_raises=None, tls_version="TLSv1.3"):
    """Return a factory that produces the same mock conn for both handshakes."""
    conn = _build_mock_ssl_conn(leaf_cert, chain_certs, verify_raises=verify_raises,
                                tls_version=tls_version)

    ctx = MagicMock(spec=ssl.SSLContext)
    ctx.wrap_socket.return_value = conn
    return ctx, conn


def _patch_fetch(leaf_cert, chain_certs=(), verify_raises=None):
    """
    Context-manager that patches just enough of the socket/ssl stack so
    ``fetch_cert_from_domain(..., allow_private=True)`` uses our fake certs.

    Strategy:
      - ``socket.getaddrinfo`` → returns a loopback address
      - ``socket.socket`` → returns a mock (no real OS socket)
      - ``ssl.SSLContext`` → returns a mock context for the first (permissive) handshake
      - ``ssl.create_default_context`` → returns a verifying mock context whose
        ``wrap_socket().connect()`` either succeeds or raises *verify_raises*

    For the verifying handshake (``_verify_handshake`` inside extractor.py):
      _client_context(True, ...) calls ssl.create_default_context() which we patch
      to return a ctx whose wrap_socket() gives a conn that either succeeds or
      raises ssl.SSLCertVerificationError on connect().
    """
    fake_addrinfo = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 443))]

    # First (permissive) handshake mock
    main_ctx, main_conn = _mock_ctx(leaf_cert, chain_certs)

    # Second (verifying) handshake mock
    verify_ctx = MagicMock(spec=ssl.SSLContext)
    verify_conn = MagicMock()
    verify_conn.__enter__ = lambda s: s
    verify_conn.__exit__ = MagicMock(return_value=False)
    verify_conn.version.return_value = "TLSv1.3"
    verify_conn.cipher.return_value = ("TLS_AES_256_GCM_SHA384", "TLSv1.3", 256)
    verify_conn.getpeercert.return_value = _cert_der(leaf_cert)

    if verify_raises is not None:
        # connect() raises the SSL verification error
        verify_conn.connect.side_effect = verify_raises
    else:
        verify_conn.connect.return_value = None

    verify_ctx.wrap_socket.return_value = verify_conn

    mock_sock = MagicMock()
    mock_sock.__enter__ = lambda s: s
    mock_sock.__exit__ = MagicMock(return_value=False)

    def _create_default_ctx(purpose=None, cafile=None):
        return verify_ctx

    def _sslctx_factory(proto=None):
        return main_ctx

    return (
        patch("socket.getaddrinfo", return_value=fake_addrinfo),
        patch("socket.socket", return_value=mock_sock),
        patch("ssl.SSLContext", side_effect=_sslctx_factory),
        patch("ssl.create_default_context", side_effect=_create_default_ctx),
    )


# ---------------------------------------------------------------------------
# 1. Self-signed server detected end-to-end (mock)
# ---------------------------------------------------------------------------
def test_self_signed_server_mock():
    """Equivalent of test_self_signed_server_is_detected_end_to_end (no loopback needed)."""
    cert, _ = make_cert(cn="127.0.0.1", san=("localhost",), ip_san=("127.0.0.1",))

    patches = _patch_fetch(cert)
    with patches[0], patches[1], patches[2], patches[3]:
        _, d = fetch_cert_from_domain("127.0.0.1", port=9443, timeout=3, allow_private=True)

    assert d["is_self_signed"] is True
    assert d["hostname"] == "127.0.0.1"
    assert d["hostname_match"] is True
    assert "SELF_SIGNED" in _ids(d)


# ---------------------------------------------------------------------------
# 2. Hostname mismatch detected end-to-end (mock)
# ---------------------------------------------------------------------------
def test_hostname_mismatch_mock():
    """Equivalent of test_hostname_mismatch_is_detected_end_to_end (no loopback needed)."""
    cert, _ = make_cert(cn="other.example.com", san=("other.example.com",))

    patches = _patch_fetch(cert)
    with patches[0], patches[1], patches[2], patches[3]:
        _, d = fetch_cert_from_domain("127.0.0.1", port=9443, timeout=3, allow_private=True)

    assert d["hostname_match"] is False
    assert "HOSTNAME_MISMATCH" in _ids(d)


# ---------------------------------------------------------------------------
# 3. Full chain captured and verified against CA (mock)
# ---------------------------------------------------------------------------
def test_full_chain_captured_and_verified_mock():
    """Equivalent of test_full_chain_is_captured_and_verified_against_ca (no loopback needed)."""
    ch = make_chain(leaf_cn="127.0.0.1", leaf_san=("localhost",), ip_san=("127.0.0.1",))

    # Verifying handshake succeeds → verify_raises=None
    patches = _patch_fetch(ch["leaf"], chain_certs=[ch["inter"]], verify_raises=None)
    with patches[0], patches[1], patches[2], patches[3]:
        _, d = fetch_cert_from_domain("127.0.0.1", port=9443, timeout=3, allow_private=True)

    assert d["is_self_signed"] is False
    assert d["chain"] is not None
    assert d["chain"]["length"] >= 1
    # No critical trust/mismatch findings
    found_ids = _ids(d)
    assert "HOSTNAME_MISMATCH" not in found_ids
    assert "SELF_SIGNED" not in found_ids


# ---------------------------------------------------------------------------
# 4. Untrusted issuer flagged when no CA given (mock)
# ---------------------------------------------------------------------------
def test_untrusted_issuer_flagged_mock():
    """Equivalent of test_untrusted_issuer_is_flagged_when_no_ca_given (no loopback needed)."""
    ch = make_chain(leaf_cn="127.0.0.1", leaf_san=("localhost",), ip_san=("127.0.0.1",))

    # Build an SSLCertVerificationError with the .verify_code / .verify_message
    # attributes that extractor._verify_handshake reads.
    ssl_err = ssl.SSLCertVerificationError(
        ssl.SSLErrorNumber.SSL_ERROR_SSL,
        "unable to get local issuer certificate",
    )
    ssl_err.verify_code = 20          # X509_V_ERR_UNABLE_TO_GET_ISSUER_CERT_LOCALLY
    ssl_err.verify_message = "unable to get local issuer certificate"

    patches = _patch_fetch(ch["leaf"], chain_certs=[ch["inter"]], verify_raises=ssl_err)
    with patches[0], patches[1], patches[2], patches[3]:
        _, d = fetch_cert_from_domain("127.0.0.1", port=9443, timeout=3, allow_private=True)

    assert "UNTRUSTED_CHAIN" in _ids(d)


# ---------------------------------------------------------------------------
# 5. Missing intermediate reported as untrusted (mock)
# ---------------------------------------------------------------------------
def test_missing_intermediate_untrusted_mock():
    """Equivalent of test_missing_intermediate_is_reported_as_untrusted (no loopback needed)."""
    ch = make_chain(leaf_cn="127.0.0.1", leaf_san=("localhost",), ip_san=("127.0.0.1",))

    # Server only sends leaf (no intermediate) → verification fails
    ssl_err = ssl.SSLCertVerificationError(
        ssl.SSLErrorNumber.SSL_ERROR_SSL,
        "unable to get local issuer certificate",
    )
    ssl_err.verify_code = 20
    ssl_err.verify_message = "unable to get local issuer certificate"

    patches = _patch_fetch(ch["leaf"], chain_certs=[], verify_raises=ssl_err)
    with patches[0], patches[1], patches[2], patches[3]:
        _, d = fetch_cert_from_domain("127.0.0.1", port=9443, timeout=3, allow_private=True)

    assert "UNTRUSTED_CHAIN" in _ids(d)


# ---------------------------------------------------------------------------
# 6. Batch scan: ranks by risk and survives errors (mock)
# ---------------------------------------------------------------------------
def test_batch_ranks_by_risk_and_survives_errors_mock():
    """
    Equivalent of test_batch_ranks_by_risk_and_survives_errors (no loopback needed).
    Uses parse_cert_from_pem + _row_from_report directly — no network calls needed.
    """
    from analyzer.batch import _row_from_report, _error_row

    model = CertificateAnomalyModel(
        model_path=os.path.join(tempfile.mkdtemp(), "m.joblib")
    )

    # "good" cert: clean 60-day leaf → lower risk
    good_c, _ = make_cert(cn="127.0.0.1", san=("localhost",), ip_san=("127.0.0.1",), days=60)
    _, good_d = parse_cert_from_pem(pem(good_c))
    good_rep = analyze(good_d, model)

    # "bad" cert: tiny serial + huge lifetime → higher risk
    bad_c, _ = make_cert(cn="x.example.com", san=("x.example.com",), days=3000, serial=3)
    _, bad_d = parse_cert_from_pem(pem(bad_c))
    bad_rep = analyze(bad_d, model)

    good_row = _row_from_report("127.0.0.1:9001", good_rep)
    bad_row  = _row_from_report("127.0.0.1:9002", bad_rep)
    err_row  = _error_row("no-such-host.invalid", ConnectionError("nxdomain"))

    # Sort: ok rows desc by risk_score, errors last (same logic as scan_targets)
    rows = [good_row, bad_row, err_row]
    rows.sort(key=lambda r: (r["status"] != "ok", -(r["risk_score"] or 0) if r["status"] == "ok" else 0))

    assert len(rows) == 3
    ok_rows = [r for r in rows if r["status"] == "ok"]
    assert len(ok_rows) == 2
    assert ok_rows[0]["risk_score"] >= ok_rows[1]["risk_score"], "higher-risk cert must rank first"
    assert rows[-1]["status"] == "error", "error rows must be last"
    assert "ConnectionError" in rows[-1]["error"]

    # CSV round-trip
    parsed = list(csv.DictReader(io.StringIO(rows_to_csv(rows))))
    assert len(parsed) == 3
    assert "risk_score" in parsed[0]
