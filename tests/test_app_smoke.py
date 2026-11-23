"""Executes app.py against a stub `streamlit` so every UI code path is run at least once (no browser needed)."""

import os
import runpy
import sys
import types
from unittest.mock import MagicMock

from analyzer import analyze, parse_cert_from_pem
from analyzer.ml_model import CertificateAnomalyModel
from .certgen import make_cert, pem


class _State(dict):
    __getattr__ = dict.get

    def __setattr__(self, k, v):
        self[k] = v


def _stub(session):
    st = MagicMock(name="streamlit")
    st.session_state = session
    st.cache_resource = lambda fn: fn
    st.cache_data = lambda *a, **kw: (lambda fn: fn)
    st.tabs = lambda labels: [MagicMock() for _ in labels]
    st.columns = lambda spec, **kw: [MagicMock() for _ in range(spec if isinstance(spec, int) else len(spec))]
    st.button.return_value = False
    st.checkbox.side_effect = lambda label, value=False, **kw: value
    st.text_input.return_value = ""
    st.text_area.return_value = ""
    st.file_uploader.return_value = None
    st.number_input.return_value = 443
    st.slider.return_value = 8
    return st


def _run(session):
    stub = _stub(session)
    old = sys.modules.get("streamlit")
    sys.modules["streamlit"] = stub
    try:
        runpy.run_path(os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.py"), run_name="__main__")
    finally:
        if old is None:
            sys.modules.pop("streamlit", None)
        else:
            sys.modules["streamlit"] = old
    return stub


def test_app_runs_with_no_report():
    stub = _run(_State())
    assert stub.info.called            # "Analyze a domain or upload a certificate first."


def test_app_renders_a_full_report_and_batch_table():
    cert, _ = make_cert(serial=5, days=900)
    details = parse_cert_from_pem(pem(cert))[1]
    model = CertificateAnomalyModel()
    session = _State(report=analyze(details, model), source="test", batch_rows=[
        {"target": "a.test", "status": "ok", "risk_level": "High", "risk_score": 60, "rule_score": 60,
         "ml_anomaly_percentile": 0.4, "issues": "X", "days_remaining": 10, "key": "RSA-2048",
         "signature": "SHA256", "issuer_cn": "CA", "error": ""},
        {"target": "b.test", "status": "error", "risk_level": "", "risk_score": "", "rule_score": "",
         "ml_anomaly_percentile": "", "issues": "", "days_remaining": "", "key": "", "signature": "",
         "issuer_cn": "", "error": "ConnectionError: x"}])
    stub = _run(session)
    assert stub.download_button.call_count == 2      # CSV + JSON export were both offered
    assert stub.dataframe.called
