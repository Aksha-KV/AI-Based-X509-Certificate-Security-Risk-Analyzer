"""
X.509 certificate extraction layer.

Responsibilities
----------------
* Parse PEM / DER certificates (and PEM bundles with a chain).
* Pull a flat, JSON-friendly dictionary of security-relevant features out of a
  certificate (key, signature, validity, SAN, extensions, structure ...).
* Fetch a certificate (and, when possible, the chain the server sent) from a
  live TLS endpoint, with an SSRF guard so the tool is safe to host.
* Optionally perform a *second*, fully verifying handshake so the report can say
  whether the system trust store accepts the chain, and why not.
"""

from __future__ import annotations

import datetime
import ipaddress
import math
import socket
import ssl
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

from cryptography import x509
from cryptography.exceptions import InvalidSignature, UnsupportedAlgorithm
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import dsa, ec, ed448, ed25519, rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, ExtensionOID, NameOID

PRIVATE_SUFFIXES = (
    ".local", ".internal", ".lan", ".corp", ".home", ".intranet",
    ".localdomain", ".localhost", ".test", ".invalid",
)
WEAK_HASHES = {"SHA1", "MD5", "MD2"}


class BlockedTargetError(ValueError):
    """Raised when a target resolves to a private / loopback / reserved address."""


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _name_attr(name: x509.Name, oid: x509.ObjectIdentifier) -> str:
    attrs = name.get_attributes_for_oid(oid)
    return str(attrs[0].value) if attrs else ""


def _name_dict(name: x509.Name) -> Dict[str, str]:
    return {
        "common_name": _name_attr(name, NameOID.COMMON_NAME),
        "organization": _name_attr(name, NameOID.ORGANIZATION_NAME),
        "organizational_unit": _name_attr(name, NameOID.ORGANIZATIONAL_UNIT_NAME),
        "country": _name_attr(name, NameOID.COUNTRY_NAME),
        "state": _name_attr(name, NameOID.STATE_OR_PROVINCE_NAME),
        "locality": _name_attr(name, NameOID.LOCALITY_NAME),
    }


def _name_string(name: x509.Name) -> str:
    parts = [f"{a.rfc4514_attribute_name}={a.value}" for a in name]
    return ", ".join(parts) if parts else "Unknown"


def shannon_entropy(text: str) -> float:
    """Shannon entropy (bits/char). 0.0 for empty strings."""
    if not text:
        return 0.0
    counts = Counter(text)
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def is_private_name(name: str) -> bool:
    """True for names/IPs that must not appear in publicly trusted certificates."""
    n = name.strip().lower().rstrip(".")
    if not n:
        return False
    try:
        ip = ipaddress.ip_address(n)
        return bool(ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved)
    except ValueError:
        pass
    if n == "localhost" or n.endswith(PRIVATE_SUFFIXES):
        return True
    bare = n[2:] if n.startswith("*.") else n
    return "." not in bare  # single-label ("intranet", "printer01")


def hostname_matches(hostname: str, dns_names: List[str], ip_addrs: List[str]) -> bool:
    """
    RFC 9525-style matching against SAN entries only (the CN is ignored).
    Wildcards are honoured only as the complete left-most label.
    """
    host = hostname.strip().lower().rstrip(".")
    try:
        target_ip = ipaddress.ip_address(host)
    except ValueError:
        target_ip = None
    if target_ip is not None:
        for ip in ip_addrs:
            try:
                if ipaddress.ip_address(ip) == target_ip:
                    return True
            except ValueError:
                continue
        return False

    for pattern in dns_names:
        p = pattern.strip().lower().rstrip(".")
        if "*" not in p:
            if p == host:
                return True
            continue
        if not p.startswith("*.") or "*" in p[2:]:
            continue  # partial / multiple wildcards are not accepted
        suffix = p[2:]
        if "." not in suffix:
            continue  # "*.com" style
        first, _, rest = host.partition(".")
        if first and rest == suffix:
            return True
    return False


def _security_strength(key_type: str, key_size: int) -> int:
    """Approximate NIST SP 800-57 security strength in bits."""
    if key_type in ("RSA", "DSA"):
        if key_size < 1024:
            return 40
        if key_size < 2048:
            return 80
        if key_size < 3072:
            return 112
        if key_size < 7680:
            return 128
        if key_size < 15360:
            return 192
        return 256
    if key_type == "EC":
        return min(256, key_size // 2)
    if key_type == "Ed25519":
        return 128
    if key_type == "Ed448":
        return 224
    return 0


def _is_self_signed(cert: x509.Certificate) -> bool:
    """
    True only if issuer == subject *and* the certificate verifies under its own key.
    If the signature cannot be checked at all (unsupported legacy hash) we fall back
    to the name comparison so e.g. MD5 self-signed certs are still recognised.
    """
    if cert.subject != cert.issuer:
        return False
    try:
        cert.verify_directly_issued_by(cert)
        return True
    except InvalidSignature:
        return False
    except Exception:  # unsupported algorithm etc.
        return True


def _signature_info(cert: x509.Certificate) -> Tuple[Optional[str], str, str]:
    """Return (hash_name_or_None, readable algorithm, dotted OID)."""
    oid = cert.signature_algorithm_oid
    oid_name = getattr(oid, "_name", "") or oid.dotted_string
    try:
        h = cert.signature_hash_algorithm
        hash_name = h.name.upper() if h is not None else None
    except (UnsupportedAlgorithm, ValueError):
        hash_name = None  # e.g. MD2 - fall back to the OID name below
    return hash_name, (hash_name or oid_name.upper()), oid.dotted_string


# --------------------------------------------------------------------------- #
# Chain analysis
# --------------------------------------------------------------------------- #
def analyze_chain(chain: List[x509.Certificate]) -> Dict[str, Any]:
    """Check that each certificate was issued by the next one and flag weak links."""
    links: List[Dict[str, Any]] = []
    all_valid = True
    weak_intermediates: List[str] = []
    non_ca_issuers: List[str] = []

    for i, cert in enumerate(chain):
        issuer = chain[i + 1] if i + 1 < len(chain) else None
        verified: Optional[bool] = None
        if issuer is not None:
            try:
                cert.verify_directly_issued_by(issuer)
                verified = True
            except InvalidSignature:
                verified = False
            except Exception:
                verified = None
            if verified is False:
                all_valid = False
            try:
                bc = issuer.extensions.get_extension_for_class(x509.BasicConstraints).value
                if not bc.ca:
                    non_ca_issuers.append(_name_attr(issuer.subject, NameOID.COMMON_NAME) or "unknown")
            except x509.ExtensionNotFound:
                pass
        hash_name, _, _ = _signature_info(cert)
        is_root = _is_self_signed(cert)
        # The leaf (index 0) is scored elsewhere; roots' own signature is irrelevant.
        if i > 0 and not is_root and hash_name in WEAK_HASHES:
            weak_intermediates.append(_name_attr(cert.subject, NameOID.COMMON_NAME) or "unknown")
        links.append({
            "subject": _name_string(cert.subject),
            "issuer": _name_string(cert.issuer),
            "signature_hash": hash_name,
            "issued_by_next_verified": verified,
            "self_signed": is_root,
        })

    return {
        "length": len(chain),
        "links": links,
        "all_links_valid": all_valid,
        "root_included": bool(chain) and links[-1]["self_signed"],
        "weak_intermediates": weak_intermediates,
        "non_ca_issuers": non_ca_issuers,
    }


# --------------------------------------------------------------------------- #
# Main extraction
# --------------------------------------------------------------------------- #
def extract_cert_details(
    cert: x509.Certificate,
    hostname: Optional[str] = None,
    chain: Optional[List[x509.Certificate]] = None,
    tls_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Flatten a certificate into a dictionary of security-relevant features.

    hostname  - the name the client intended to reach (enables hostname-match check)
    chain     - certificates the server sent, leaf first (enables chain checks)
    tls_info  - result of a verifying handshake (see fetch_cert_from_domain)
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    not_before = cert.not_valid_before_utc
    not_after = cert.not_valid_after_utc
    total_lifetime_days = (not_after - not_before).days
    days_remaining = (not_after - now).days

    # ---- public key ------------------------------------------------------- #
    pk = cert.public_key()
    key_type, key_size, curve, rsa_exponent = "Unknown", 0, None, None
    if isinstance(pk, rsa.RSAPublicKey):
        key_type, key_size = "RSA", pk.key_size
        rsa_exponent = pk.public_numbers().e
    elif isinstance(pk, ec.EllipticCurvePublicKey):
        key_type, key_size, curve = "EC", pk.key_size, pk.curve.name
    elif isinstance(pk, dsa.DSAPublicKey):
        key_type, key_size = "DSA", pk.key_size
    elif isinstance(pk, ed25519.Ed25519PublicKey):
        key_type, key_size = "Ed25519", 256
    elif isinstance(pk, ed448.Ed448PublicKey):
        key_type, key_size = "Ed448", 448
    else:
        key_type = pk.__class__.__name__

    sig_hash, sig_alg, sig_oid = _signature_info(cert)

    # ---- names ------------------------------------------------------------ #
    subject_dict, issuer_dict = _name_dict(cert.subject), _name_dict(cert.issuer)
    cn = subject_dict["common_name"]
    self_issued = cert.subject == cert.issuer
    is_self_signed = _is_self_signed(cert)

    # ---- SAN -------------------------------------------------------------- #
    san_dns: List[str] = []
    san_ips: List[str] = []
    try:
        san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        san_dns = list(san.get_values_for_type(x509.DNSName))
        san_ips = [str(ip) for ip in san.get_values_for_type(x509.IPAddress)]
    except x509.ExtensionNotFound:
        pass
    san_list = san_dns + san_ips
    is_wildcard = cn.startswith("*.") or any(s.startswith("*.") for s in san_dns)
    san_private = [s for s in san_list if is_private_name(s)]

    # ---- generic extension statistics ------------------------------------- #
    exts = list(cert.extensions)
    extension_count = len(exts)
    critical_ext_count = sum(1 for e in exts if e.critical)
    unrecognized_ext_count = sum(1 for e in exts if isinstance(e.value, x509.UnrecognizedExtension))

    # ---- AIA / CRL -------------------------------------------------------- #
    ocsp_urls: List[str] = []
    ca_issuers_urls: List[str] = []
    try:
        aia = cert.extensions.get_extension_for_class(x509.AuthorityInformationAccess).value
        for ad in aia:
            if ad.access_method == x509.AuthorityInformationAccessOID.OCSP:
                ocsp_urls.append(ad.access_location.value)
            elif ad.access_method == x509.AuthorityInformationAccessOID.CA_ISSUERS:
                ca_issuers_urls.append(ad.access_location.value)
    except x509.ExtensionNotFound:
        pass

    crl_urls: List[str] = []
    try:
        for dp in cert.extensions.get_extension_for_class(x509.CRLDistributionPoints).value:
            for gn in dp.full_name or []:
                if isinstance(gn.value, str):
                    crl_urls.append(gn.value)
    except x509.ExtensionNotFound:
        pass

    # ---- basic constraints / key usage / EKU ------------------------------ #
    is_ca, path_length, has_basic_constraints = False, None, False
    try:
        bc = cert.extensions.get_extension_for_class(x509.BasicConstraints).value
        is_ca, path_length, has_basic_constraints = bc.ca, bc.path_length, True
    except x509.ExtensionNotFound:
        pass

    key_usage: List[str] = []
    try:
        ku = cert.extensions.get_extension_for_class(x509.KeyUsage).value
        for flag, label in [
            ("digital_signature", "Digital Signature"), ("content_commitment", "Content Commitment"),
            ("key_encipherment", "Key Encipherment"), ("data_encipherment", "Data Encipherment"),
            ("key_agreement", "Key Agreement"), ("key_cert_sign", "Certificate Sign"),
            ("crl_sign", "CRL Sign"),
        ]:
            if getattr(ku, flag):
                key_usage.append(label)
    except x509.ExtensionNotFound:
        pass

    eku_names: List[str] = []
    eku_server_auth = False
    has_eku = False
    try:
        eku = cert.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value
        has_eku = True
        eku_names = [getattr(o, "_name", "") or o.dotted_string for o in eku]
        eku_server_auth = any(
            o in (ExtendedKeyUsageOID.SERVER_AUTH, ExtendedKeyUsageOID.ANY_EXTENDED_KEY_USAGE) for o in eku
        )
    except x509.ExtensionNotFound:
        pass

    has_must_staple = False
    try:
        tf = cert.extensions.get_extension_for_class(x509.TLSFeature).value
        has_must_staple = x509.TLSFeatureType.status_request in tf
    except x509.ExtensionNotFound:
        pass

    has_sct = False
    try:
        cert.extensions.get_extension_for_oid(ExtensionOID.PRECERT_SIGNED_CERTIFICATE_TIMESTAMPS)
        has_sct = True
    except x509.ExtensionNotFound:
        pass

    # ---- AKI / SKI presence (structural correctness indicator) ----------- #
    has_aki = False
    try:
        cert.extensions.get_extension_for_class(x509.AuthorityKeyIdentifier)
        has_aki = True
    except x509.ExtensionNotFound:
        pass

    has_ski = False
    try:
        cert.extensions.get_extension_for_class(x509.SubjectKeyIdentifier)
        has_ski = True
    except x509.ExtensionNotFound:
        pass

    # ---- Certificate Policies OID count ---------------------------------- #
    policy_oid_count = 0
    try:
        cp = cert.extensions.get_extension_for_class(x509.CertificatePolicies).value
        policy_oid_count = sum(1 for _ in cp)
    except x509.ExtensionNotFound:
        pass

    # ---- hostname / chain / TLS ------------------------------------------- #
    hostname_match: Optional[bool] = None
    if hostname:
        hostname_match = hostname_matches(hostname, san_dns, san_ips)

    chain_info = analyze_chain(chain) if chain and len(chain) > 0 else None

    serial_bits = cert.serial_number.bit_length()

    return {
        "subject": _name_string(cert.subject),
        "subject_dict": subject_dict,
        "issuer": _name_string(cert.issuer),
        "issuer_dict": issuer_dict,
        "common_name": cn or (san_list[0] if san_list else "Unknown"),
        "subject_field_count": len(list(cert.subject)),
        "not_before": not_before.isoformat(),
        "not_after": not_after.isoformat(),
        "total_lifetime_days": total_lifetime_days,
        "days_remaining": days_remaining,
        "is_expired": now > not_after,
        "is_not_yet_valid": now < not_before,
        "key_type": key_type,
        "key_size": key_size,
        "key_curve": curve,
        "rsa_exponent": rsa_exponent,
        "security_strength_bits": _security_strength(key_type, key_size),
        "signature_algorithm": sig_alg,
        "signature_hash": sig_hash,
        "signature_oid": sig_oid,
        "is_self_issued": self_issued,
        "is_self_signed": is_self_signed,
        "san_list": san_list,
        "san_dns": san_dns,
        "san_ips": san_ips,
        "san_count": len(san_list),
        "san_private_names": san_private,
        "is_wildcard": is_wildcard,
        "cn_entropy": round(shannon_entropy(cn), 4),
        "cn_length": len(cn),
        "is_ca": is_ca,
        "path_length": path_length,
        "has_basic_constraints": has_basic_constraints,
        "key_usage": key_usage,
        "eku": eku_names,
        "has_eku": has_eku,
        "eku_server_auth": eku_server_auth,
        "has_must_staple": has_must_staple,
        "has_sct": has_sct,
        "has_aki": has_aki,
        "has_ski": has_ski,
        "policy_oid_count": policy_oid_count,
        "extension_count": extension_count,
        "critical_extension_count": critical_ext_count,
        "unrecognized_extension_count": unrecognized_ext_count,
        "ocsp_urls": ocsp_urls,
        "ca_issuers_urls": ca_issuers_urls,
        "crl_urls": crl_urls,
        "serial_number": hex(cert.serial_number),
        "serial_bits": serial_bits,
        "sha256_fingerprint": cert.fingerprint(hashes.SHA256()).hex(),
        "sha1_fingerprint": cert.fingerprint(hashes.SHA1()).hex(),
        "version": f"v{cert.version.value + 1}",
        "hostname": hostname,
        "hostname_match": hostname_match,
        "chain": chain_info,
        "tls": tls_info,
    }


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def parse_cert_from_pem(
    data: bytes | str, hostname: Optional[str] = None
) -> Tuple[x509.Certificate, Dict[str, Any]]:
    """
    Parse PEM (single certificate or bundle) or DER bytes.
    In a bundle the first certificate is the subject; the rest are treated as its chain.
    Raises ValueError if nothing can be parsed.
    """
    if isinstance(data, str):
        data = data.encode("utf-8")

    certs: List[x509.Certificate] = []
    try:
        certs = x509.load_pem_x509_certificates(data)
    except Exception:
        certs = []
    if not certs:
        try:
            certs = [x509.load_der_x509_certificate(data)]
        except Exception:
            certs = []
    if not certs:
        raise ValueError(
            "Failed to parse X.509 certificate. Ensure the file is a valid PEM or DER certificate."
        )

    chain = certs if len(certs) > 1 else None
    return certs[0], extract_cert_details(certs[0], hostname=hostname, chain=chain)


# --------------------------------------------------------------------------- #
# Live TLS fetch
# --------------------------------------------------------------------------- #
def parse_target(raw: str, default_port: int = 443) -> Tuple[str, int]:
    """Accepts 'host', 'host:port', 'https://host:port/path', '[::1]:8443'."""
    text = raw.strip()
    if "://" not in text:
        text = "//" + text
    parts = urlsplit(text)
    host = (parts.hostname or "").strip().lower()
    if not host:
        raise ValueError(f"Could not read a hostname from {raw!r}")
    try:
        port = parts.port or default_port
    except ValueError:
        raise ValueError(f"Invalid port in {raw!r}")
    try:
        host = host.encode("idna").decode("ascii")
    except UnicodeError:
        pass
    return host, port


def _forbidden_ip(ip_text: str) -> bool:
    ip = ipaddress.ip_address(ip_text.split("%")[0])
    return bool(
        ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast
        or ip.is_reserved or ip.is_unspecified
    )


def _resolve(host: str, port: int, allow_private: bool) -> Tuple[int, str]:
    """Resolve once, block internal targets, and return (family, ip) to connect to."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ConnectionError(f"Could not resolve domain name '{host}': {exc}")
    if not infos:
        raise ConnectionError(f"No address records for '{host}'")
    if not allow_private:
        for fam, _t, _p, _c, sockaddr in infos:
            if _forbidden_ip(sockaddr[0]):
                raise BlockedTargetError(
                    f"'{host}' resolves to a private/internal address ({sockaddr[0]}). "
                    "Scanning internal hosts is disabled."
                )
    fam, _t, _p, _c, sockaddr = infos[0]
    return fam, sockaddr[0]


def _client_context(verify: bool, ca_file: Optional[str]) -> ssl.SSLContext:
    if verify:
        return ssl.create_default_context(cafile=ca_file)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    # An inspection tool should be able to look at legacy servers too.
    try:
        ctx.minimum_version = ssl.TLSVersion.TLSv1
        ctx.set_ciphers("DEFAULT:@SECLEVEL=0")
    except (ValueError, ssl.SSLError):
        pass
    return ctx


def _peer_chain(conn: ssl.SSLSocket) -> List[x509.Certificate]:
    """Best-effort retrieval of the chain the server sent (leaf first)."""
    try:
        getter = getattr(conn, "get_unverified_chain", None)
        raw = getter() if getter else conn._sslobj.get_unverified_chain()  # type: ignore[attr-defined]
        out: List[x509.Certificate] = []
        for item in raw or []:
            if isinstance(item, (bytes, bytearray)):
                der = bytes(item)
            else:
                der = item.public_bytes(ssl._ssl.ENCODING_DER)  # type: ignore[attr-defined]
            out.append(x509.load_der_x509_certificate(der))
        return out
    except Exception:
        return []


def _verify_handshake(ip: str, family: int, host: str, port: int, timeout: float,
                      ca_file: Optional[str]) -> Dict[str, Any]:
    """Second, fully verifying handshake - tells us whether the chain is trusted and why not."""
    ctx = _client_context(True, ca_file)
    sock = socket.socket(family, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        conn = ctx.wrap_socket(sock, server_hostname=host)
        conn.connect((ip, port))
        # Capture the fingerprint of the certificate from the verifying handshake
        # so callers can detect if a TLS inspection proxy substituted a different cert.
        der = conn.getpeercert(binary_form=True)
        verified_fp: Optional[str] = None
        if der:
            from cryptography.hazmat.primitives import hashes as _hashes
            from cryptography import x509 as _x509
            try:
                _cert = _x509.load_der_x509_certificate(der)
                verified_fp = _cert.fingerprint(_hashes.SHA256()).hex()
            except Exception:
                pass
        info: Dict[str, Any] = {
            "verified": True,
            "tls_version": conn.version(),
            "cipher": (conn.cipher() or ("",))[0],
        }
        if verified_fp:
            info["verified_fingerprint"] = verified_fp
        conn.close()
        return info
    except ssl.SSLCertVerificationError as exc:
        return {"verified": False, "verify_code": exc.verify_code, "reason": exc.verify_message}
    except (ssl.SSLError, OSError) as exc:
        return {"verified": None, "error": str(exc)}
    finally:
        try:
            sock.close()
        except Exception:
            pass


def fetch_cert_from_domain(
    hostname: str,
    port: int = 443,
    timeout: float = 10,
    allow_private: bool = False,
    verify_tls: bool = True,
    ca_file: Optional[str] = None,
) -> Tuple[x509.Certificate, Dict[str, Any]]:
    """
    Connect to a TLS endpoint (SNI), return the leaf certificate and its feature dict.

    allow_private - permit loopback/RFC1918 targets (off by default: SSRF protection).
    verify_tls    - additionally run a verifying handshake and record the outcome.
    ca_file       - custom CA bundle for the verifying handshake (testing / private PKI).
    """
    host, port = parse_target(hostname, default_port=port)
    family, ip = _resolve(host, port, allow_private)

    sock = socket.socket(family, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        conn = _client_context(False, None).wrap_socket(sock, server_hostname=host)
        conn.connect((ip, port))
        der = conn.getpeercert(binary_form=True)
        if not der:
            raise ValueError(f"No certificate returned by {host}:{port}")
        tls_version = conn.version()
        cipher = (conn.cipher() or ("",))[0]
        chain = _peer_chain(conn)
        cert = x509.load_der_x509_certificate(der)
        conn.close()
    except (socket.timeout, TimeoutError):
        raise TimeoutError(f"Connection timed out while attempting to reach {host}:{port}")
    except BlockedTargetError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Error establishing TLS connection to {host}:{port} - {exc}")
    finally:
        try:
            sock.close()
        except Exception:
            pass

    tls_info: Dict[str, Any] = {"tls_version": tls_version, "cipher": cipher, "verified": None}
    if verify_tls:
        tls_info.update(_verify_handshake(ip, family, host, port, timeout, ca_file))
        tls_info["tls_version"] = tls_version  # keep the version from the permissive probe

    details = extract_cert_details(cert, hostname=host, chain=chain or None, tls_info=tls_info)
    return cert, details
