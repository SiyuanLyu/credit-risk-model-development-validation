# Model Development Document
## Retail Credit Card PD / LGD / EAD Model — v0.1 (starter draft)

*Template aligned with the documentation expectations of SR 11-7.
Sections marked TODO are yours to complete as you work through the project;
replace every bracketed placeholder with your real results.*

### 1. Purpose and use
The model estimates probability of default (PD), loss given default (LGD)
and exposure at default (EAD) for a retail credit-card portfolio, for use in
expected-loss measurement (CECL-style) and stress testing (CCAR-style).
It is a development exercise on public data and is not used for real lending
decisions.

### 2. Data
- Source: UCI Machine Learning Repository #350, "Default of Credit Card
  Clients" (Yeh & Lien, 2009) — 30,000 accounts, Apr–Sep 2005 behaviour,
  default observed the following month. Overall default rate: [from
  data_profile.json].
- Development sample: 21,000 accounts (70%); hold-out: 9,000 (30%),
  stratified, seed 42.
- Data-quality findings: TODO — document the EDUCATION/MARRIAGE code
  clean-up and any other anomalies you find, with counts.

### 3. Methodology
- Champion: logistic-regression scorecard on WOE-transformed features
  (10 quantile bins, fit on development sample only), PDO scaling
  (20 points to double the odds, 600 at 1:50).
- Challenger: XGBoost (300 trees, depth 4) on raw features, used for
  benchmarking only.
- EAD: drawn balance + 0.50 CCF × undrawn commitment. LGD: scenario
  parameter 0.85 base (0.70–0.95 sensitivity) — no recovery data available.

### 4. Results
TODO — insert AUC / KS / Gini from outputs/pd_metrics.json and portfolio
EL from outputs/el_summary.csv. Interpret, don't just paste.

### 5. Assumptions and limitations
- Single-period (next-month) default definition; lifetime extension in the
  stress module is a documented simplification.
- LGD is assumed, not estimated (no workout/recovery data).
- Macro overlay coefficient (beta to unemployment) is illustrative until
  estimated in the satellite-model extension.
- TODO — add at least two limitations you discover yourself.

### 6. Implementation and reproducibility
Run `bash run_pipeline.sh`. All parameters are in the script headers.
