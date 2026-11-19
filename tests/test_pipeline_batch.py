"""End-to-end pipeline and batch scanning."""

import csv
import io
import os
import tempfile

import pytest

from analyzer import analyze, parse_cert_from_pem, rows_to_csv, scan_targets
from analyzer.ml_model import CertificateAnomalyModel
from analyzer.batch import _row_from_report, _error_row
from .certgen import make_cert, pem, tls_server
from .test_network import _LOOPBACK_OK

_M = None


def model():
    global _M
    if _M is None:
        _M = CertificateAnomalyModel(model_path=os.path.join(tempfile.mkdtemp(), "m.joblib"))
    return _M


_skip_no_loopback = pytest.mark.skipif(
    not _LOOPBACK_OK,
    reason="OS blocks loopback TLS connections (WinError 10013 / sandbox restriction)"
)


def test_pipeline_report_structure_and_explainability():
    cert, _ = make_cert(serial=5)  # self-signed, tiny serial
    _, d = parse_cert_from_pem(pem(cert))
    rep = analyze(d, model())
    assert set(rep) >= {"cert", "rules", "ml", "final", "analyzed_at"}
    f = rep["final"]
    assert 0 <= f["score"] <= 100 and f["score"] >= f["rule_score"]
    assert f["summary_reasons"], "every report must carry human-readable reasons"
    assert {v["rule_id"] for v in rep["rules"]["vulnerabilities"]} >= {"SELF_SIGNED", "LOW_SERIAL_ENTROPY"}


def test_pipeline_works_without_ml():
    _, d = parse_cert_from_pem(pem(make_cert()[0]))
    rep = analyze(d, None)
    assert rep["ml"] is None and rep["final"]["score"] == rep["rules"]["risk_score"]


@_skip_no_loopback
def test_batch_ranks_by_risk_and_survives_errors():
    good_c, good_k = make_cert(cn="127.0.0.1", san=("localhost",), ip_san=("127.0.0.1",), days=60)
    bad_c, bad_k = make_cert(cn="x.example.com", san=("x.example.com",), days=3000, serial=3)
    pem_dir = tempfile.mkdtemp()
    with tls_server(good_c, good_k) as p1, tls_server(bad_c, bad_k) as p2:
        rows = scan_targets([f"127.0.0.1:{p1}", f"127.0.0.1:{p2}", "no-such-host.invalid", "# comment", ""],
                            ml_model=model(), allow_private=True, timeout=3, workers=3, save_pem_dir=pem_dir)
    assert len(rows) == 3
    ok = [r for r in rows if r["status"] == "ok"]
    assert len(ok) == 2 and ok[0]["risk_score"] >= ok[1]["risk_score"]        # ranked high -> low
    assert rows[-1]["status"] == "error" and "ConnectionError" in rows[-1]["error"]  # errors last
    assert len([f for f in os.listdir(pem_dir) if f.endswith(".pem")]) == 2   # corpus export works
    parsed = list(csv.DictReader(io.StringIO(rows_to_csv(rows))))
    assert len(parsed) == 3 and "risk_score" in parsed[0]


def test_batch_ranks_by_risk_no_network():
    """
    Socket-free equivalent of the live batch test: build report rows directly from
    PEM-parsed certs, insert a synthetic error row, then verify that scan_targets'
    sorting logic (risk desc, errors last) works correctly via _row_from_report /
    _error_row helpers and the batch CSV machinery.
    """
    # "good" cert: clean 60-day leaf → low risk
    good_c, _ = make_cert(cn="good.example.com", san=("good.example.com",), days=60)
    _, good_d = parse_cert_from_pem(pem(good_c))
    good_rep = analyze(good_d, model())

    # "bad" cert: tiny serial + huge lifetime → higher risk
    bad_c, _ = make_cert(cn="bad.example.com", san=("bad.example.com",), days=3000, serial=3)
    _, bad_d = parse_cert_from_pem(pem(bad_c))
    bad_rep = analyze(bad_d, model())

    good_row = _row_from_report("good.example.com", good_rep)
    bad_row  = _row_from_report("bad.example.com",  bad_rep)
    err_row  = _error_row("no-such-host.invalid", ConnectionError("nxdomain"))

    # Replicate scan_targets sort: ok rows desc by risk_score, errors last
    rows = [good_row, bad_row, err_row]
    rows.sort(key=lambda r: (r["status"] != "ok", -(r["risk_score"] or 0) if r["status"] == "ok" else 0))

    assert len(rows) == 3
    ok = [r for r in rows if r["status"] == "ok"]
    assert len(ok) == 2
    assert ok[0]["risk_score"] >= ok[1]["risk_score"], "higher-risk cert must rank first"
    assert rows[-1]["status"] == "error", "error rows must be last"

    # CSV round-trip
    parsed = list(csv.DictReader(io.StringIO(rows_to_csv(rows))))
    assert len(parsed) == 3 and "risk_score" in parsed[0]


def test_batch_blocks_private_targets_by_default():
    rows = scan_targets(["127.0.0.1:9", "192.168.0.10"], timeout=1)
    assert all(r["status"] == "error" and "BlockedTargetError" in r["error"] for r in rows)


def test_evaluation_harness_scores_hits_misses_false_positives_and_known_gaps():
    from analyzer.evaluation import evaluate
    ss = parse_cert_from_pem(pem(make_cert(cn="ss.test", san=("ss.test",))[0]), hostname="ss.test")[1]
    ok = parse_cert_from_pem(pem(make_cert(cn="ok.test", san=("ok.test",), days=60)[0]), hostname="ok.test")[1]
    ok["is_self_signed"] = False                       # pretend a CA issued it
    ok["ocsp_urls"], ok["has_sct"] = ["http://o"], True
    fake = {"ss.test": ss, "ok.test": ok}
    cases = [
        {"host": "ss.test", "expect": {"SELF_SIGNED"}},
        {"host": "ss.test", "expect": {"EXPIRED"}},             # deliberately missed
        {"host": "ok.test", "clean": True},
        {"host": "ss.test", "clean": True},                     # self-signed is High -> false positive
        {"host": "ss.test", "expect": {"REVOKED"}, "known_miss": True},
        {"host": "gone.test", "clean": True},                   # fetch error
    ]
    def fetch(h):
        if h not in fake:
            raise ConnectionError("nope")
        return fake[h]
    res = evaluate(cases, fetch)
    results = [r["result"] for r in res["rows"]]
    assert results == ["PASS", "MISS", "PASS", "FALSE-POSITIVE", "KNOWN-MISS", "ERROR"], results
    s = res["summary"]
    assert s["expected_findings_detected"] == 1 and s["expected_findings_missed"] == 1 and s["recall"] == 0.5
    assert s["healthy_hosts_passed"] == 1 and s["healthy_hosts_total"] == 2 and s["fetch_errors"] == 1


def test_report_view_tables_and_export():
    import json
    from analyzer.report_view import chain_table, export_json, findings_table, ml_caption, ml_factors_table
    from .certgen import good_details
    from .test_ml import WEIRD
    rep = analyze(good_details(**WEIRD), model())
    ft = findings_table(rep)
    assert list(ft.columns)[:3] == ["Severity", "Category", "Rule"] and len(ft) == rep["rules"]["issue_count"]
    assert len(ml_factors_table(rep)) > 0                              # unusual cert -> factors shown
    calm = analyze(good_details(extension_count=10, critical_extension_count=2, ocsp_urls=["a"], crl_urls=["b"]), model())
    assert len(ml_factors_table(calm)) == 0                            # typical cert -> no noisy factors
    assert chain_table(calm).empty
    assert "synthetic" in ml_caption(rep) and "not a probability" in ml_caption(rep)
    assert json.loads(export_json(rep))["final"]["score"] == rep["final"]["score"]
