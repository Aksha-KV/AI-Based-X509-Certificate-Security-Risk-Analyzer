# 🛡️ AI-Based X.509 Certificate Security Risk Analyzer

Explainable pre-deployment assessment of X.509 / TLS certificates:
**deterministic security rules** decide what is wrong, an **unsupervised anomaly model** flags unusual
structure, and a **traceable 0-100 score** explains every point with a reason and a remediation.

> The score is a *communication design*, not a probability of compromise. The tool complements
> OpenSSL, SSL Labs and certificate linters; it does not replace them.

## What it does

| Capability | Detail |
| :-- | :-- |
| Input | Live TLS endpoint (SNI), PEM/DER file, pasted PEM, PEM bundle with chain, or a **batch list of hosts** |
| Parsing | Key (type, size, curve, exponent), signature hash, validity, SAN, EKU, key usage, basic constraints, OCSP/CRL, SCT, must-staple, serial entropy, extension statistics |
| Trust | Real self-signature verification, **hostname matching (RFC 9525 rules)**, served-chain link verification, weak intermediates, **verifying handshake against the system trust store** |
| Rules | 26 rules with severity, category, standard reference, remediation and per-finding score contribution |
| Scoring | Noisy-OR combination (bounded, diminishing returns) + bounded ML add-on |
| ML | Isolation Forest on 21 time-independent structural features, percentile-based score, per-certificate explanation |
| Output | Dashboard, JSON report, CSV batch export |
| Safety | SSRF guard (private/loopback targets blocked by default), stale-model protection |

## Architecture

```
input (domain | file | bundle | batch)
  -> extractor.py    parse + features + SSRF-safe fetch + chain + verifying handshake
  -> rules_engine.py deterministic findings, noisy-OR score, contributions, advisories
  -> ml_model.py     Isolation Forest anomaly percentile + top contributing features
  -> pipeline.py     combine (rules dominate; ML adds a bounded amount, never subtracts)
  -> report_view.py / app.py / batch.py / scripts/*
```

## Scoring

* Each finding has a penalty `p` (0-100). Findings combine as `score = 100 * (1 - prod(1 - p_i/100))`.
  The score can never exceed 100, and minor issues cannot add up to "Critical".
* Each finding shows its **marginal contribution** to the score (they sum to the rule score).
* A *Critical*-severity finding (e.g. expired) forces the level to Critical regardless of score.
* Levels: Low < 20 <= Medium < 45 <= High < 70 <= Critical.
* Availability issues (expiry) are tagged separately from security issues and weighted lower.
* **Hybrid:** `final = 100 * (1 - (1 - rule/100) * (1 - 0.25 * a))`, where `a` is the anomaly percentile rescaled so that
  the top 10% most unusual certificates map to 0..1 (everything below the 90th percentile contributes 0).
  The ML cannot lower a score and cannot lift a rule-clean certificate above 25.
* The penalty values are expert-chosen priors, **not fitted to data** (see Limitations).

## Rules

| Rule | Finding | Penalty | Severity | Category | Basis |
| :-- | :-- | :-: | :-: | :-- | :-- |
| `EXPIRED` | Certificate expired | 60 | Critical | availability | RFC 5280 s6.1 (validity) |
| `NOT_YET_VALID` | Certificate not yet valid | 45 | High | availability | RFC 5280 s6.1 (validity) |
| `HOSTNAME_MISMATCH` | Hostname not covered by the certificate | 45 | High | trust | RFC 9525 (service identity) |
| `WEAK_SIGNATURE` | Weak signature hash (SHA-1 / MD5 / MD2) | 40 | High | cryptography | CA/B Forum BR; NIST SP 800-131A |
| `WEAK_KEY_SIZE_RSA` | RSA key shorter than 2048 bits | 35 | High | cryptography | NIST SP 800-57 Pt.1; CA/B Forum BR |
| `WEAK_KEY_SIZE_EC` | EC key smaller than 256 bits | 35 | High | cryptography | NIST SP 800-57 Pt.1 |
| `UNTRUSTED_CHAIN` | Chain does not validate against the system trust store | 35 | High | trust | RFC 5280 s6 (path validation) |
| `CHAIN_BROKEN_LINK` | Chain contains a link whose signature does not verify | 35 | High | trust | RFC 5280 s6 (path validation) |
| `DSA_KEY` | DSA public key | 30 | High | cryptography | FIPS 186-5; CA/B Forum BR |
| `SELF_SIGNED` | Self-signed certificate | 30 | High | trust | RFC 5280; CA/B Forum BR |
| `NON_CA_ISSUER` | Chain issuer is not a CA | 30 | High | trust | RFC 5280 s4.2.1.9 (basicConstraints) |
| `CA_ON_LEAF` | End-entity certificate is marked CA:TRUE | 30 | High | trust | RFC 5280 s4.2.1.9; CA/B Forum BR |
| `EXPIRING_SOON_CRITICAL` | Certificate expires within 14 days | 25 | High | availability | Operational best practice |
| `WEAK_CHAIN_SIGNATURE` | Intermediate certificate signed with a weak hash | 25 | High | cryptography | CA/B Forum BR; NIST SP 800-131A |
| `MISSING_SAN` | No Subject Alternative Name | 20 | Medium | compliance | RFC 9525; CA/B Forum BR |
| `EXCESSIVE_LIFETIME` | Validity longer than the Baseline Requirements allow | 20 | Medium | compliance | CA/B Forum BR s6.3.2 (limit depends on issue date) |
| `NONSTANDARD_EC_CURVE` | EC curve not allowed for the Web PKI | 15 | Medium | cryptography | CA/B Forum BR s6.1.5 |
| `SMALL_RSA_EXPONENT` | RSA public exponent below 65537 | 15 | Medium | cryptography | CA/B Forum BR s6.1.6 |
| `MISSING_EKU_SERVERAUTH` | Extended Key Usage lacks serverAuth | 15 | Medium | compliance | CA/B Forum BR s7.1.2 |
| `INTERNAL_NAMES` | Internal / private names or IPs in certificate | 15 | Medium | compliance | CA/B Forum BR (internal names deprecated since 2015) |
| `EXPIRING_SOON_HIGH` | Certificate expires within 30 days | 10 | Medium | availability | Operational best practice |
| `LOW_SERIAL_ENTROPY` | Serial number shorter than 64 bits | 10 | Low | compliance | CA/B Forum BR s7.1 (>= 64 bits CSPRNG output) |
| `NO_REVOCATION_INFO` | No OCSP or CRL pointer | 10 | Low | compliance | RFC 5280 s4.2.1.13 / s4.2.2.1 |
| `EXCESSIVE_SAN_COUNT` | Very large number of SAN entries (> 100) | 10 | Low | hygiene | Security best practice |
| `WILDCARD_USAGE` | Wildcard certificate | 10 | Low | hygiene | Security best practice |
| `NO_SCT` | No embedded Certificate Transparency SCT | 5 | Low | transparency | RFC 9162; browser CT policies |

Maximum validity by issue date (publicly trusted TLS certificates; **re-check against the current Baseline Requirements**):
2020-09-01: 398 days, 2026-03-15: 200 days, 2027-03-15: 100 days, 2029-03-15: 47 days, earlier: 825 days.

Advisories that do **not** change the score: post-quantum readiness, shrinking-validity trend, trust-store check unavailable.

## Machine-learning component (honest description)

* **Model:** `IsolationForest` (unsupervised). It learns what a *typical* certificate looks like and scores rarity.
* **Features (21, time-independent):** key type/strength/exponent, hash size, lifetime, SAN count/wildcard, extension
  count / critical count / unrecognised count, serial bits, CN entropy and length, EKU, revocation pointers, SCT,
  self-signed, CA flag, private SAN names, subject field count. `days_remaining` is intentionally excluded.
* **Score:** percentile of the reference set that is more normal than the certificate. It is a rarity measure.
* **Explanation:** occlusion attribution (replace one feature with the reference median, measure how much more normal the
  certificate becomes). Shown only when the certificate is unusual (>= 80th percentile).
* **Reference data:** by default a *synthetic placeholder* baseline (flagged in the UI and in every report).
  For real results, collect certificates and retrain:

```bash
python scripts/scan_batch.py tranco_top1000.txt --save-pem data/corpus --csv corpus_scan.csv
python scripts/train_model.py --corpus data/corpus      # needs >= 200 certificates
```

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py
python -m pytest tests -v          # or:  python -m tests.run_tests   (no pytest needed)
```

Batch scan from the command line:

```bash
python scripts/scan_batch.py domains.txt --csv results.csv
```

Hosted deployments: internal/private targets are blocked. Operators can opt in with `X509_ALLOW_PRIVATE=1`.

## Evaluation

`analyzer/evaluation.py` + `scripts/evaluate_badssl.py` run the analyzer on public badssl.com hosts with known
defects (expired, wrong host, self-signed, untrusted root, incomplete chain, SHA-1 intermediate, 1000 SANs) and known-good
hosts, and report recall on expected findings, healthy-host pass rate, unexpected High/Critical findings and
known misses (e.g. `revoked.badssl.com` needs live OCSP/CRL checking, which is not implemented).

```bash
python scripts/evaluate_badssl.py --csv eval_badssl.csv     # requires internet
```

Suggested further evaluation for the report: scan a few hundred domains from a popularity list, compare findings with
SSL Labs grades and a certificate linter (e.g. zlint), and run an ablation of rules-only vs ML-only vs hybrid.

## Tests

54 tests: extractor (parsing, real SHA-1/MD5 fixtures, self-signature verification, hostname matching, chains),
every rule, scoring maths, validity policy by date, the ML model (time-independence, explanations, corpus training,
persistence), SSRF protection, a **local TLS server** exercising live fetch / chain capture / trust verification,
batch scanning, the evaluation harness and a UI smoke test.

## Limitations (please read)

* Penalties are priors, not calibrated on labelled data. Scores are for prioritisation, not risk measurement.
* The bundled anomaly reference is synthetic unless you train on a real corpus. A rarity score is not maliciousness.
* No revocation checking (OCSP/CRL URLs are extracted and a missing pointer is flagged, but no live check).
* The Certificate Transparency check only sees *embedded* SCTs; SCTs delivered by TLS extension/OCSP appear as missing.
* Only the served chain is analysed; no path building, name constraints or policy validation beyond the listed checks.
* Validity limits apply to publicly trusted TLS certificates; private CAs may legitimately differ.
* Server-side TLS protocol/cipher configuration is out of scope (use SSL Labs / testssl.sh).

## Future work

Calibrate penalties on labelled data (e.g. logistic regression), SHAP for the tree model, live OCSP/CRL checks,
CT-log lookups, name-constraint and policy validation, certificate-chain graph analytics across an inventory.

## License

MIT - see `LICENSE`.
