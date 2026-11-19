"""Helpers that build certificates, chains and a local TLS server for tests (no network needed)."""

from __future__ import annotations

import contextlib
import datetime
import os
import socket
import ssl
import tempfile
import threading
from typing import Any, Dict, Iterable, Optional

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID, ObjectIdentifier

PEM = serialization.Encoding.PEM
NOW = datetime.datetime.now(datetime.timezone.utc)


def new_key(kind: str = "ec256"):
    if kind == "rsa2048":
        return rsa.generate_private_key(65537, 2048)
    if kind == "rsa1024":
        return rsa.generate_private_key(65537, 1024)
    if kind == "ec256":
        return ec.generate_private_key(ec.SECP256R1())
    if kind == "ec224":
        return ec.generate_private_key(ec.SECP224R1())
    if kind == "secp256k1":
        return ec.generate_private_key(ec.SECP256K1())
    if kind == "ed25519":
        return ed25519.Ed25519PrivateKey.generate()
    raise ValueError(kind)


def make_cert(
    cn: str = "test.example.com",
    san: Optional[Iterable[str]] = ("test.example.com",),
    key=None,
    issuer_cert: Optional[x509.Certificate] = None,
    issuer_key=None,
    days: int = 90,
    start_offset_days: int = 1,
    ca: bool = False,
    eku_server: bool = True,
    ocsp: bool = True,
    crl: bool = True,
    serial: Optional[int] = None,
    sign_hash=hashes.SHA256,
    extra_unrecognized: int = 0,
    ip_san: Iterable[str] = (),
    must_staple: bool = False,
):
    """Return (cert, private_key). Self-signed unless issuer_cert/issuer_key are given."""
    key = key or new_key("ec256")
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])
    issuer_name = issuer_cert.subject if issuer_cert is not None else subject
    signer = issuer_key or key

    not_before = NOW - datetime.timedelta(days=start_offset_days)
    b = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer_name)
        .public_key(key.public_key())
        .serial_number(serial if serial is not None else x509.random_serial_number())
        .not_valid_before(not_before)
        .not_valid_after(not_before + datetime.timedelta(days=days))
        .add_extension(x509.BasicConstraints(ca=ca, path_length=None), critical=True)
    )
    names = [x509.DNSName(s) for s in (san or [])]
    import ipaddress
    names += [x509.IPAddress(ipaddress.ip_address(i)) for i in ip_san]
    if names:
        b = b.add_extension(x509.SubjectAlternativeName(names), critical=False)
    if eku_server and not ca:
        b = b.add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
    if ca:
        b = b.add_extension(
            x509.KeyUsage(False, False, False, False, False, True, True, False, False), critical=True)
    if ocsp and issuer_cert is not None:
        b = b.add_extension(x509.AuthorityInformationAccess([
            x509.AccessDescription(x509.AuthorityInformationAccessOID.OCSP,
                                   x509.UniformResourceIdentifier("http://ocsp.example.test"))]), critical=False)
    if crl and issuer_cert is not None:
        b = b.add_extension(x509.CRLDistributionPoints([x509.DistributionPoint(
            [x509.UniformResourceIdentifier("http://crl.example.test/x.crl")], None, None, None)]), critical=False)
    if must_staple:
        b = b.add_extension(x509.TLSFeature([x509.TLSFeatureType.status_request]), critical=False)
    for i in range(extra_unrecognized):
        b = b.add_extension(
            x509.UnrecognizedExtension(ObjectIdentifier(f"1.3.6.1.4.1.99999.{i + 1}"), b"\x05\x00"), critical=False)

    algo = None if isinstance(signer, ed25519.Ed25519PrivateKey) else sign_hash()
    return b.sign(signer, algo), key


def pem(cert: x509.Certificate) -> bytes:
    return cert.public_bytes(PEM)


def make_chain(leaf_cn: str = "localhost", leaf_san=("localhost",), **leaf_kwargs):
    """root -> intermediate -> leaf. Returns dict with certs and keys."""
    root, root_key = make_cert(cn="Test Root CA", san=None, ca=True, days=3650, ocsp=False, crl=False)
    inter, inter_key = make_cert(cn="Test Intermediate CA", san=None, ca=True, days=1800,
                                 issuer_cert=root, issuer_key=root_key)
    leaf, leaf_key = make_cert(cn=leaf_cn, san=leaf_san, issuer_cert=inter, issuer_key=inter_key, **leaf_kwargs)
    return {"root": root, "inter": inter, "leaf": leaf, "root_key": root_key,
            "inter_key": inter_key, "leaf_key": leaf_key}


def good_details(**overrides) -> Dict[str, Any]:
    """A healthy publicly-trusted-looking leaf as a details dict, for precise rule tests."""
    d: Dict[str, Any] = {
        "days_remaining": 60, "is_expired": False, "is_not_yet_valid": False,
        "total_lifetime_days": 90,
        "not_before": (datetime.datetime.now() - datetime.timedelta(days=30)).isoformat(),
        "key_type": "EC", "key_size": 256, "key_curve": "secp256r1", "rsa_exponent": None,
        "security_strength_bits": 128,
        "signature_algorithm": "SHA256", "signature_hash": "SHA256",
        "is_self_signed": False, "is_ca": False,
        "san_list": ["example.com"], "san_dns": ["example.com"], "san_ips": [], "san_count": 1,
        "san_private_names": [], "is_wildcard": False,
        "eku_server_auth": True, "has_eku": True, "has_sct": True,
        "ocsp_urls": ["http://ocsp.example"], "crl_urls": [],
        "serial_bits": 128, "extension_count": 9, "critical_extension_count": 2,
        "unrecognized_extension_count": 0, "cn_entropy": 3.2, "cn_length": 11,
        "subject_field_count": 1,
        "hostname": None, "hostname_match": None, "chain": None, "tls": None,
        # New structural fields (ML v2)
        "key_usage": ["Digital Signature"],
        "has_aki": True, "has_ski": True, "policy_oid_count": 1,
        # sha256_fingerprint for fingerprint-mismatch rule
        "sha256_fingerprint": "a" * 64,
    }
    d.update(overrides)
    return d


def assert_raises(exc_types, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_types:
        return
    raise AssertionError(f"expected {exc_types} from {fn.__name__}")


@contextlib.contextmanager
def tls_server(leaf: x509.Certificate, leaf_key, chain_certs: Iterable[x509.Certificate] = ()):
    """Serve `leaf` (+ intermediates) on 127.0.0.1:<port> until the block exits."""
    tmp = tempfile.mkdtemp()
    chain_file, key_file = os.path.join(tmp, "chain.pem"), os.path.join(tmp, "key.pem")
    with open(chain_file, "wb") as fh:
        fh.write(pem(leaf))
        for c in chain_certs:
            fh.write(pem(c))
    with open(key_file, "wb") as fh:
        fh.write(leaf_key.private_bytes(PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(chain_file, key_file)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)
    srv.settimeout(0.2)
    port = srv.getsockname()[1]
    stop = threading.Event()

    def loop():
        while not stop.is_set():
            try:
                raw, _ = srv.accept()
            except (socket.timeout, OSError):
                continue
            try:
                raw.settimeout(2)
                with ctx.wrap_socket(raw, server_side=True):
                    pass
            except Exception:
                try:
                    raw.close()
                except Exception:
                    pass

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    try:
        yield port
    finally:
        stop.set()
        t.join(timeout=2)
        srv.close()
