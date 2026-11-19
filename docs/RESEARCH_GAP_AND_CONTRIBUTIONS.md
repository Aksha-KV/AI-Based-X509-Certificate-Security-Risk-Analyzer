# Research gap, contributions and evaluation plan (draft for the report/slides)

## Sharper gap statement
1. **Linters report, they do not prioritise.** Certificate linters (zlint, certlint) and scanners return many binary
   pass/fail findings. They do not combine them into one traceable risk score with per-finding contributions and
   remediation.
2. **ML work on certificates mostly targets other problems** (malware/traffic classification, CT-log mining) and often
   relies on labelled data. A pre-deployment auditor needs *unsupervised* rarity detection that is **time-independent**,
   **bounded** (cannot override deterministic rules) and **explained per certificate**.
3. **Tools look at one layer.** OpenSSL shows fields, SSL Labs grades the server configuration. Cross-field reasoning
   (hostname vs SAN, CA flag on a leaf, weak intermediate in the served chain, validity vs issue-date policy) is left to
   the human.

## Contributions of this project (what is actually implemented)
* 26 explainable rules with severity, category, standard reference and remediation; noisy-OR scoring with
  per-finding marginal contributions.
* Date-aware validity policy (825 -> 398 -> 200 -> 100 -> 47 days) instead of a single fixed threshold.
* Real self-signature verification, RFC 9525 hostname matching, served-chain verification, weak-intermediate detection
  and a verifying handshake that records *why* the system trust store rejects a chain.
* Isolation Forest over 21 time-independent structural features with percentile scores, occlusion-based explanations,
  stored provenance, and a bounded add-on to the rule score.
* Batch inventory scanning with risk ranking and CSV export; SSRF-safe fetching.
* Evaluation harness against known-good / known-bad hosts, offline test-bed with a local TLS server.

## Evaluation to run and report (needs internet / real data)
1. `scripts/evaluate_badssl.py` -> recall on known-bad hosts, pass rate on known-good hosts, listed known misses.
2. Scan 300-1000 domains from a popularity list (`scripts/scan_batch.py`), compare findings with SSL Labs grades and a
   linter; report agreement and disagreement cases.
3. Retrain the anomaly model on the scanned corpus (`scripts/train_model.py`); report the outlier rate and manually
   inspect the top-20 outliers.
4. Ablation: rules-only vs ML-only vs hybrid (how often does the ML add a finding the rules missed?).

## Reading list (verify details before citing)
* Durumeric et al., "Analysis of the HTTPS Certificate Ecosystem", IMC 2013.
* Kumar et al., "Tracking Certificate Misissuance in the Wild" (zlint), IEEE S&P 2018.
* Liu, Ting, Zhou, "Isolation Forest", ICDM 2008.
* Chung et al., "Is the Web Ready for OCSP Must-Staple?", IMC 2018.
* RFC 5280, RFC 9525, RFC 6962 / RFC 9162; CA/Browser Forum Baseline Requirements (incl. SC-081);
  NIST SP 800-57 Pt.1, SP 800-131A, FIPS 203/204/205.
