"""Extractor tests: parsing, self-signature verification, hostname matching, chain analysis."""

from analyzer.extractor import (analyze_chain, hostname_matches, is_private_name,
                                parse_cert_from_pem, parse_target)
from .certgen import assert_raises, make_cert, make_chain, new_key, pem


def test_parse_basic_fields():
    cert, _ = make_cert(key=new_key("rsa2048"), san=("a.example.com", "b.example.com"))
    _, d = parse_cert_from_pem(pem(cert))
    assert d["key_type"] == "RSA" and d["key_size"] == 2048 and d["rsa_exponent"] == 65537
    assert d["signature_hash"] == "SHA256" and d["san_count"] == 2
    assert d["security_strength_bits"] == 112
    assert d["serial_bits"] > 64 and d["eku_server_auth"] is True


def test_ec_and_ed25519_keys():
    c, _ = make_cert(key=new_key("ec256"))
    assert parse_cert_from_pem(pem(c))[1]["key_curve"] == "secp256r1"
    c, _ = make_cert(key=new_key("ed25519"))
    d = parse_cert_from_pem(pem(c))[1]
    assert d["key_type"] == "Ed25519" and d["signature_hash"] is None  # no separate hash


def test_self_signed_is_verified_not_just_name_compare():
    c, _ = make_cert()
    assert parse_cert_from_pem(pem(c))[1]["is_self_signed"] is True
    # Same subject/issuer name but signed by a different key -> NOT self-signed.
    other = new_key("ec256")
    fake, _ = make_cert(cn="x.example.com", san=("x.example.com",), key=new_key("ec256"), issuer_key=other)
    d = parse_cert_from_pem(pem(fake))[1]
    assert d["is_self_issued"] is True and d["is_self_signed"] is False


def test_ca_issued_leaf_is_not_self_signed():
    ch = make_chain()
    d = parse_cert_from_pem(pem(ch["leaf"]))[1]
    assert d["is_self_signed"] is False


def _fixture(name):
    import os
    with open(os.path.join(os.path.dirname(__file__), "fixtures", name), "rb") as fh:
        return fh.read()


def test_sha1_and_md5_signatures_detected_on_real_certs():
    # Generated with the openssl CLI (the cryptography library refuses to sign with SHA-1).
    assert parse_cert_from_pem(_fixture("sha1_selfsigned.pem"))[1]["signature_hash"] == "SHA1"
    d = parse_cert_from_pem(_fixture("md5_selfsigned.pem"))[1]
    assert d["signature_hash"] == "MD5" and d["is_self_signed"] is True


def test_extension_statistics_and_serial():
    c, _ = make_cert(extra_unrecognized=3, serial=7)
    d = parse_cert_from_pem(pem(c))[1]
    assert d["unrecognized_extension_count"] == 3
    assert d["serial_bits"] == 3
    assert d["extension_count"] >= 5


def test_must_staple_flag():
    c, _ = make_cert(must_staple=True)
    assert parse_cert_from_pem(pem(c))[1]["has_must_staple"] is True


def test_hostname_matching_rules():
    dns = ["example.com", "*.example.com"]
    assert hostname_matches("example.com", dns, [])
    assert hostname_matches("WWW.Example.com.", dns, [])
    assert not hostname_matches("a.b.example.com", dns, [])      # wildcard covers one label only
    assert not hostname_matches("evil-example.com", dns, [])
    assert not hostname_matches("x.com", ["*.com"], [])          # too-broad wildcard rejected
    assert not hostname_matches("fooexample.com", ["f*.example.com"], [])  # partial wildcard rejected
    assert hostname_matches("10.0.0.5", [], ["10.0.0.5"])
    assert not hostname_matches("10.0.0.6", [], ["10.0.0.5"])


def test_private_name_detection():
    for n in ["localhost", "intranet", "host.local", "printer.corp", "192.168.1.4", "10.1.2.3", "*.internal"]:
        assert is_private_name(n), n
    for n in ["example.com", "*.example.com", "8.8.8.8"]:
        assert not is_private_name(n), n


def test_parse_target_variants():
    assert parse_target("https://Example.COM:8443/path") == ("example.com", 8443)
    assert parse_target("example.com") == ("example.com", 443)
    assert parse_target("[::1]:9443") == ("::1", 9443)
    assert_raises(ValueError, parse_target, "")


def test_chain_analysis_valid_and_broken():
    ch = make_chain()
    good = analyze_chain([ch["leaf"], ch["inter"], ch["root"]])
    assert good["all_links_valid"] and good["root_included"] and good["length"] == 3
    other = make_chain()  # unrelated intermediate
    bad = analyze_chain([ch["leaf"], other["inter"], ch["root"]])
    assert bad["all_links_valid"] is False


def test_pem_bundle_provides_chain():
    ch = make_chain()
    bundle = pem(ch["leaf"]) + pem(ch["inter"]) + pem(ch["root"])
    _, d = parse_cert_from_pem(bundle, hostname="localhost")
    assert d["chain"]["length"] == 3 and d["hostname_match"] is True


def test_invalid_bytes_raise():
    assert_raises(ValueError, parse_cert_from_pem, b"INVALID_CERTIFICATE_CONTENT")
