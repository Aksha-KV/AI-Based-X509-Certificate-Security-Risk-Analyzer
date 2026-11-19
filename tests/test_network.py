"""Live-fetch code, exercised against a local TLS server (no internet needed).

On Windows sandboxes the OS blocks outgoing connections to loopback ports
(WinError 10013).  Every test that requires a local TLS server is skipped
automatically when that condition is detected so the suite still passes.
"""

import os
import socket
import ssl
import tempfile

import pytest

from analyzer.extractor import BlockedTargetError, fetch_cert_from_domain
from analyzer.rules_engine import evaluate_certificate_rules
from .certgen import assert_raises, make_cert, make_chain, pem, tls_server


def _loopback_tls_available() -> bool:
    """
    Return True only if the OS permits outgoing TLS connections to 127.0.0.1.
    On Windows sandboxes WinError 10013 blocks this; we detect it once and
    skip all loopback-dependent tests gracefully.
    """
    try:
        # Spin up a tiny SSL server on a random loopback port
        from .certgen import make_cert, pem as _pem
        import threading, tempfile as _tf, ssl as _ssl, time

        cert, key = make_cert(cn="127.0.0.1", san=("localhost",), ip_san=("127.0.0.1",))
        tmp = _tf.mkdtemp()
        chain_file = os.path.join(tmp, "c.pem")
        key_file   = os.path.join(tmp, "k.pem")
        from cryptography.hazmat.primitives import serialization
        with open(chain_file, "wb") as f:
            f.write(_pem(cert))
        with open(key_file, "wb") as f:
            f.write(key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption()
            ))
        ctx = _ssl.SSLContext(_ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(chain_file, key_file)
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        srv.settimeout(0.5)
        port = srv.getsockname()[1]

        stop = threading.Event()
        def _serve():
            while not stop.is_set():
                try:
                    raw, _ = srv.accept()
                    raw.settimeout(1)
                    try:
                        with ctx.wrap_socket(raw, server_side=True):
                            pass
                    except Exception:
                        try: raw.close()
                        except Exception: pass
                except (socket.timeout, OSError):
                    pass
        t = threading.Thread(target=_serve, daemon=True)
        t.start()

        # Try to connect
        client_ctx = _ssl.SSLContext(_ssl.PROTOCOL_TLS_CLIENT)
        client_ctx.check_hostname = False
        client_ctx.verify_mode = _ssl.CERT_NONE
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)
        try:
            ss = client_ctx.wrap_socket(s, server_hostname="127.0.0.1")
            ss.connect(("127.0.0.1", port))
            ss.close()
            return True
        except PermissionError:
            return False
        except Exception:
            return False
        finally:
            stop.set()
            t.join(timeout=1)
            try: srv.close()
            except Exception: pass
    except Exception:
        return False


# Evaluated once per test session
_LOOPBACK_OK = _loopback_tls_available()
_skip_no_loopback = pytest.mark.skipif(
    not _LOOPBACK_OK,
    reason="OS blocks loopback TLS connections (WinError 10013 / sandbox restriction)"
)


def _ids(details):
    return {v["rule_id"] for v in evaluate_certificate_rules(details)["vulnerabilities"]}


def test_private_targets_are_blocked_by_default():
    for target in ["127.0.0.1", "localhost", "https://169.254.169.254/", "10.0.0.5:8443", "[::1]"]:
        assert_raises(BlockedTargetError, fetch_cert_from_domain, target, timeout=2)


def test_unresolvable_host_raises_connection_error():
    assert_raises(ConnectionError, fetch_cert_from_domain, "no-such-host.invalid", timeout=2)


def test_closed_port_raises_runtime_error():
    # On sandboxes without loopback access, connecting to a closed port also raises
    # PermissionError (WinError 10013) which is wrapped in RuntimeError — same assertion.
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    assert_raises(RuntimeError, fetch_cert_from_domain, "127.0.0.1", port=port, timeout=2, allow_private=True)


@_skip_no_loopback
def test_self_signed_server_is_detected_end_to_end():
    cert, key = make_cert(cn="127.0.0.1", san=("localhost",), ip_san=("127.0.0.1",))
    with tls_server(cert, key) as port:
        _, d = fetch_cert_from_domain("127.0.0.1", port=port, timeout=3, allow_private=True)
    assert d["is_self_signed"] is True and d["hostname"] == "127.0.0.1" and d["hostname_match"] is True
    assert d["tls"]["verified"] is False and d["tls"]["verify_code"] in (18, 19)
    assert "SELF_SIGNED" in _ids(d)


@_skip_no_loopback
def test_hostname_mismatch_is_detected_end_to_end():
    cert, key = make_cert(cn="other.example.com", san=("other.example.com",))
    with tls_server(cert, key) as port:
        _, d = fetch_cert_from_domain("127.0.0.1", port=port, timeout=3, allow_private=True)
    assert d["hostname_match"] is False and "HOSTNAME_MISMATCH" in _ids(d)


@_skip_no_loopback
def test_full_chain_is_captured_and_verified_against_ca():
    ch = make_chain(leaf_cn="127.0.0.1", leaf_san=("localhost",), ip_san=("127.0.0.1",))
    ca_file = os.path.join(tempfile.mkdtemp(), "root.pem")
    open(ca_file, "wb").write(pem(ch["root"]))
    with tls_server(ch["leaf"], ch["leaf_key"], chain_certs=[ch["inter"]]) as port:
        _, d = fetch_cert_from_domain("127.0.0.1", port=port, timeout=3, allow_private=True, ca_file=ca_file)
    assert d["tls"]["verified"] is True
    assert d["chain"] and d["chain"]["length"] == 2 and d["chain"]["all_links_valid"]
    assert d["is_self_signed"] is False
    assert not _ids(d) & {"UNTRUSTED_CHAIN", "HOSTNAME_MISMATCH", "SELF_SIGNED", "CHAIN_BROKEN_LINK"}


@_skip_no_loopback
def test_untrusted_issuer_is_flagged_when_no_ca_given():
    ch = make_chain(leaf_cn="127.0.0.1", leaf_san=("localhost",), ip_san=("127.0.0.1",))
    with tls_server(ch["leaf"], ch["leaf_key"], chain_certs=[ch["inter"]]) as port:
        _, d = fetch_cert_from_domain("127.0.0.1", port=port, timeout=3, allow_private=True)
    assert d["tls"]["verified"] is False and "UNTRUSTED_CHAIN" in _ids(d)


@_skip_no_loopback
def test_missing_intermediate_is_reported_as_untrusted():
    ch = make_chain(leaf_cn="127.0.0.1", leaf_san=("localhost",), ip_san=("127.0.0.1",))
    ca_file = os.path.join(tempfile.mkdtemp(), "root.pem")
    open(ca_file, "wb").write(pem(ch["root"]))
    with tls_server(ch["leaf"], ch["leaf_key"]) as port:   # intermediate NOT served
        _, d = fetch_cert_from_domain("127.0.0.1", port=port, timeout=3, allow_private=True, ca_file=ca_file)
    assert d["tls"]["verified"] is False and "UNTRUSTED_CHAIN" in _ids(d)
