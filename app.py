"""
AI-Based X.509 Certificate Security Risk Analyzer - Streamlit dashboard.

Rules decide what is wrong; an unsupervised anomaly model flags unusual structure;
the report explains every point of the score.

Security notes
--------------
* All certificate string fields displayed in the UI are sanitised with
  ``_safe_str()`` before rendering to prevent reflected XSS in environments
  where the HTML is later parsed (e.g., Streamlit's ``st.markdown``).
* Domain / hostname inputs are stripped and length-capped before being forwarded
  to the network layer.
"""

import html
import os
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from analyzer import (
    analyze,
    check_typosquatting,
    fetch_cert_from_domain,
    parse_cert_from_pem,
    rows_to_csv,
    scan_targets,
    validate_domain,
)
from analyzer.ml_model import CertificateAnomalyModel
from analyzer.report_view import (chain_table, export_json, findings_table, ml_caption,
                                  ml_factors_table)

st.set_page_config(page_title="X.509 Certificate Risk Analyzer", page_icon="🛡️", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    .badge {padding:6px 14px;border-radius:20px;font-weight:700;font-size:16px;display:inline-block;color:#fff}
    .badge-critical{background:#d62728}.badge-high{background:#ff7f0e}
    .badge-medium{background:#b8860b}.badge-low{background:#2e9e4f}
    </style>
    """,
    unsafe_allow_html=True,
)

# Hosted deployments should leave internal scanning OFF (SSRF protection). Operators can opt in.
ALLOW_PRIVATE_OPTION = os.environ.get("X509_ALLOW_PRIVATE", "") == "1"
_MAX_DOMAIN_LEN = 253  # RFC 1035 label limit


def _safe_str(value: object, max_len: int = 500) -> str:
    """HTML-escape and length-cap an arbitrary certificate field for safe display."""
    text = str(value) if value is not None else ""
    text = text[:max_len]
    return html.escape(text)


def _normalize_hostname(raw: str) -> str:
    """
    Normalize a user-supplied hostname/URL to a bare, lowercase domain name.
    Preserves backward compatibility for callers and internal usage.
    """
    is_valid, host, _ = validate_domain(raw, allow_private=True)
    if host:
        return host[:_MAX_DOMAIN_LEN]
    # Fallback to standard urlsplit stripping if not parsed
    from urllib.parse import urlsplit
    text = raw.strip()
    if not text:
        return ""
    if "://" not in text:
        text = "//" + text
    try:
        parts = urlsplit(text)
        host = (parts.hostname or "").strip().lower()
    except Exception:
        host = ""
    if not host:
        host = raw.strip().lower()
        for prefix in ("https://", "http://"):
            if host.startswith(prefix):
                host = host[len(prefix):]
        host = host.split("/")[0].split(":")[0]
    return host[:_MAX_DOMAIN_LEN]


def _safe_domain(raw: str) -> str:
    """Strip and cap a user-supplied domain/hostname string before network use.

    For the domain-analysis tab (where port handling is done separately) this
    delegates to _normalize_hostname so URLs like 'https://github.com' work.
    For the expected-hostname field in the Upload tab, the same normalization
    removes scheme/path before the hostname-match rule sees it.
    """
    return _normalize_hostname(raw)


@st.cache_resource
def get_model() -> CertificateAnomalyModel:
    return CertificateAnomalyModel()


@st.cache_data(ttl=300, show_spinner=False)
def get_domain_cert_details(safe_domain: str, port: int, allow_private: bool, verify_tls: bool):
    """
    Cache certificate fetch and feature extraction for consistent repeat scans.
    Expires after 5 minutes or can be cleared via Streamlit cache clear.
    """
    _, details = fetch_cert_from_domain(
        safe_domain, port=port, allow_private=allow_private, verify_tls=verify_tls
    )
    return details


model = get_model()

for key, default in {"report": None, "source": ""}.items():
    st.session_state.setdefault(key, default)

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🛡️ X.509 Risk Analyzer")
    st.caption("Explainable rules + unsupervised anomaly detection")

    st.subheader("Settings")
    expect_leaf = st.checkbox("Server (end-entity) certificate", value=True,
                              help="Untick when inspecting a CA certificate so leaf-only expectations are skipped.")
    verify_tls = st.checkbox("Verify chain against system trust store", value=True,
                             help="Runs a second, verifying TLS handshake to the target.")
    allow_private = False
    if ALLOW_PRIVATE_OPTION:
        allow_private = st.checkbox("Allow private / internal hosts", value=False)

    st.subheader("🤖 Anomaly model")
    info = model.meta
    if info.get("is_synthetic"):
        st.warning("Reference set: **synthetic placeholder**. Anomaly scores are indicative only. "
                   "Train on real certificates: `scripts/train_model.py`.")
    else:
        st.success(f"Reference set: {_safe_str(info.get('source'))}")
    st.caption(f"Isolation Forest · {len(info.get('feature_names', []))} structural features · "
               f"{info.get('n_reference', '?')} reference certificates")

    st.subheader("📜 Standards used")
    st.markdown("- RFC 5280 / RFC 9525\n- CA/B Forum Baseline Requirements\n- NIST SP 800-57, SP 800-131A")
    st.caption("The 0-100 score is a communication design, not a probability of compromise.")


def run_analysis(details, source):
    st.session_state.report = analyze(details, model, expect_leaf=expect_leaf)
    st.session_state.source = source


st.title("AI-Based X.509 Certificate Security Risk Analyzer")
st.markdown("Inspect a live TLS endpoint, a certificate file, or a whole list of hosts. "
            "Every point of the score is traceable to a named finding.")

tab_domain, tab_file, tab_batch, tab_report = st.tabs(
    ["🌐 Analyze Domain", "📄 Upload Certificate", "📦 Batch Scan", "📊 Report"])

# ----------------------------------------------------------------------------- domain
with tab_domain:
    c1, c2, c3 = st.columns([3, 1, 1.2])
    domain = c1.text_input("Domain or URL", placeholder="e.g. github.com, expired.badssl.com")
    port = c2.number_input("Port", min_value=1, max_value=65535, value=443)
    go = c3.button("🔍 Inspect", type="primary", use_container_width=True)

    st.caption("Quick targets:")
    presets = ["github.com", "expired.badssl.com", "wrong.host.badssl.com", "self-signed.badssl.com",
               "untrusted-root.badssl.com"]
    for col, name in zip(st.columns(len(presets)), presets):
        if col.button(name, use_container_width=True):
            domain, go = name, True

    if go and domain:
        is_valid, safe_domain, val_err = validate_domain(domain, allow_private=allow_private)
        if not is_valid:
            st.error(f"❌ {val_err}")
        else:
            is_typo, matched_domain, typo_err = check_typosquatting(safe_domain)
            if is_typo:
                st.error(f"❌ {typo_err}")
            else:
                with st.spinner(f"TLS handshake with {_safe_str(safe_domain)}:{int(port)} ..."):
                    try:
                        details = get_domain_cert_details(safe_domain, port=int(port), allow_private=allow_private,
                                                          verify_tls=verify_tls)
                        # Check for hostname mismatch or trust-store verification error
                        tls = details.get("tls") or {}
                        verify_code = tls.get("verify_code")
                        is_mismatch = (details.get("hostname_match") is False) or (verify_code == 62)

                        if is_mismatch:
                            cert_cn = _safe_str(details.get("common_name", "Unknown"))
                            st.error(
                                f"❌ **Certificate Hostname Mismatch**: The TLS certificate returned by "
                                f"`{_safe_str(safe_domain)}:{int(port)}` does not match the requested hostname "
                                f"'{_safe_str(safe_domain)}' (Certificate CN: `{cert_cn}`)."
                            )
                        else:
                            run_analysis(details, f"Domain: {_safe_str(safe_domain)}:{int(port)}")
                            st.success("Certificate fetched. Open the **Report** tab.")
                    except ConnectionError as exc:
                        st.error(
                            f"❌ **Connection Failed**: Could not resolve domain or establish connection to "
                            f"`{_safe_str(safe_domain)}:{int(port)}`. The domain may not exist or is unreachable. ({_safe_str(str(exc))})"
                        )
                    except (TimeoutError, socket.timeout) as exc:
                        st.error(
                            f"❌ **Connection Timed Out**: Connection to `{_safe_str(safe_domain)}:{int(port)}` timed out. "
                            f"Check network connectivity or host availability."
                        )
                    except Exception as exc:
                        st.error(f"❌ {type(exc).__name__}: {_safe_str(str(exc))}")

# ----------------------------------------------------------------------------- file
with tab_file:
    st.caption("Accepts PEM or DER. A PEM bundle (leaf first, then intermediates) also enables chain checks.")
    expected_host = st.text_input("Expected hostname (optional)", placeholder="www.example.com",
                                  help="Enables the hostname-match check.")
    uploaded = st.file_uploader("Certificate file", type=["pem", "crt", "cer", "der"])
    pasted = st.text_area("…or paste PEM text", height=140,
                          placeholder="-----BEGIN CERTIFICATE-----\n...\n-----END CERTIFICATE-----")
    if uploaded is not None:
        try:
            _, details = parse_cert_from_pem(uploaded.read(), hostname=_safe_domain(expected_host) or None)
            run_analysis(details, f"File: {_safe_str(uploaded.name)}")
            st.success(f"Parsed **{_safe_str(uploaded.name)}**. Open the **Report** tab.")
        except Exception as exc:
            st.error(f"❌ {_safe_str(str(exc))}")
    elif pasted.strip() and st.button("Analyze pasted PEM", type="primary"):
        try:
            _, details = parse_cert_from_pem(pasted.strip().encode(), hostname=_safe_domain(expected_host) or None)
            run_analysis(details, "Pasted PEM")
            st.success("Parsed. Open the **Report** tab.")
        except Exception as exc:
            st.error(f"❌ {_safe_str(str(exc))}")

# ----------------------------------------------------------------------------- batch
with tab_batch:
    st.caption("One `host` or `host:port` per line. Results are ranked by risk and can be exported as CSV.")
    targets_text = st.text_area("Targets", height=150, placeholder="github.com\nexpired.badssl.com\nexample.org:443")
    targets_file = st.file_uploader("…or upload a .txt list", type=["txt"], key="targets_file")
    workers = st.slider("Parallel connections", 1, 16, 8)
    if st.button("▶ Scan all", type="primary"):
        lines = targets_text.splitlines()
        if targets_file is not None:
            lines += targets_file.read().decode("utf-8", errors="ignore").splitlines()
        lines = [x for x in lines if x.strip() and not x.strip().startswith("#")]
        if not lines:
            st.warning("Add at least one target.")
        else:
            bar = st.progress(0.0, text="Scanning...")
            rows = scan_targets(lines, ml_model=model, workers=workers, allow_private=allow_private,
                                progress=lambda d, n: bar.progress(d / n, text=f"Scanned {d}/{n}"))
            bar.empty()
            st.session_state.batch_rows = rows
    rows = st.session_state.get("batch_rows")
    if rows:
        df = pd.DataFrame(rows)
        ok = int((df["status"] == "ok").sum())
        m1, m2, m3 = st.columns(3)
        m1.metric("Scanned OK", f"{ok}/{len(df)}")
        m2.metric("High / Critical", int(df["risk_level"].isin(["High", "Critical"]).sum()))
        m3.metric("Expiring ≤ 30 days", int(((df["status"] == "ok") & (pd.to_numeric(df["days_remaining"], errors="coerce") <= 30)).sum()))
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("📥 Download CSV", rows_to_csv(rows), "scan_results.csv", "text/csv")

# ----------------------------------------------------------------------------- report
with tab_report:
    report = st.session_state.report
    if report is None:
        st.info("Analyze a domain or upload a certificate first.")
    else:
        cert, rules, ml, final = report["cert"], report["rules"], report["ml"], report["final"]
        # Sanitise certificate fields before rendering
        common_name = _safe_str(cert.get("common_name", "Unknown"))
        queried_host = _safe_str(cert.get("hostname") or common_name)

        st.markdown(f"### Audit report: `{queried_host}`")
        if cert.get("hostname") and cert.get("common_name") and cert.get("hostname").lower() != cert.get("common_name").lower():
            st.markdown(f"**Certificate CN:** `{common_name}`")

        st.caption(f"Source: {_safe_str(st.session_state.source)} · analysed "
                   f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

        a, b, c, d = st.columns(4)
        a.markdown("##### Risk level")
        level_cls = final['level'].lower()
        a.markdown(f"<span class='badge badge-{level_cls}'>{_safe_str(final['level']).upper()}</span>",
                   unsafe_allow_html=True)
        b.metric("Risk score", f"{final['score']} / 100",
                 delta=f"rules {final['rule_score']} + ML {final['ml_contribution']}", delta_color="off")
        c.metric("Expiry", "EXPIRED" if cert["is_expired"] else f"{cert['days_remaining']} days left")
        d.metric("Key / signature", f"{_safe_str(cert['key_type'])} {cert['key_size']}b",
                 delta=_safe_str(cert["signature_algorithm"]), delta_color="off")
        st.progress(min(1.0, final["score"] / 100.0))

        st.markdown("#### Why this score")
        if final["summary_reasons"]:
            for reason in final["summary_reasons"]:
                st.write(f"• {reason}")
        else:
            st.success("No findings. Certificate conforms to the checks performed.")
        st.caption("Rules decide the score. The anomaly model can only add a bounded amount "
                   "(max 25% of the remaining headroom) and never lowers it.")

        st.markdown("#### ⚠️ Findings")
        ft = findings_table(report)
        if ft.empty:
            st.success("🎉 No security findings.")
        else:
            st.dataframe(ft, use_container_width=True, hide_index=True)
            cats = rules["category_counts"]
            st.caption("By category: " + ", ".join(f"{k} {v}" for k, v in cats.items()))

        st.markdown("#### 🛠️ Remediation")
        if rules["remediations"]:
            for i, r in enumerate(rules["remediations"], 1):
                st.warning(f"**{i}.** {r}")
        else:
            st.info("Nothing to fix.")

        st.markdown("#### 🔗 Trust & chain")
        tls = cert.get("tls") or {}
        if tls.get("verified") is True:
            st.success(f"Chain validates against the system trust store ({_safe_str(tls.get('tls_version'))}).")
        elif tls.get("verified") is False:
            st.error(f"Trust-store verification failed: {_safe_str(tls.get('reason'))} "
                     f"(OpenSSL code {tls.get('verify_code')}).")
        elif tls:
            st.info("Trust-store verification was not possible for this endpoint.")
        else:
            st.caption("Offline analysis: no live trust verification was performed.")
        ct = chain_table(report)
        if not ct.empty:
            st.dataframe(ct, use_container_width=True, hide_index=True)

        st.markdown("#### 🤖 Structural anomaly (unsupervised)")
        if ml is None:
            st.info("ML disabled.")
        else:
            st.progress(float(ml["ml_anomaly_percentile"]))
            st.caption(ml_caption(report))
            ft2 = ml_factors_table(report)
            if not ft2.empty:
                st.write("What makes it unusual:")
                st.dataframe(ft2, use_container_width=True, hide_index=True)

        if rules["advisories"]:
            with st.expander("ℹ️ Advisories (do not affect the score)"):
                for adv in rules["advisories"]:
                    st.markdown(f"**{_safe_str(adv['title'])}**  \n{_safe_str(adv['detail'])}")

        with st.expander(f"✅ Checks passed ({len(rules['passed_checks'])})"):
            for p in rules["passed_checks"]:
                st.write(f"✔ {p}")

        with st.expander("🔍 Full certificate metadata"):
            cc1, cc2 = st.columns(2)
            # subject_dict / issuer_dict: sanitise individual field values
            safe_subj = {k: _safe_str(v) for k, v in cert["subject_dict"].items()}
            safe_issuer = {k: _safe_str(v) for k, v in cert["issuer_dict"].items()}
            cc1.json({"subject": safe_subj, "issuer": safe_issuer})
            cc2.write(f"**Valid:** {cert['not_before']} → {cert['not_after']} ({cert['total_lifetime_days']} days)")
            cc2.write(f"**Serial:** `{_safe_str(cert['serial_number'])}` ({cert['serial_bits']} bits)")
            cc2.write(f"**SHA-256:** `{_safe_str(cert['sha256_fingerprint'])}`")
            cc2.write(f"**EKU:** {', '.join(_safe_str(e) for e in cert['eku']) or 'none'} "
                      f"· **Key usage:** {', '.join(_safe_str(u) for u in cert['key_usage']) or 'none'}")
            cc2.code("\n".join(_safe_str(s) for s in cert["san_list"]) or "no SAN entries")

        st.download_button("📥 Export full report (JSON)", export_json(report),
                           file_name=f"x509_report_{_safe_str(cert.get('common_name', 'cert'))}.json",
                           mime="application/json")
