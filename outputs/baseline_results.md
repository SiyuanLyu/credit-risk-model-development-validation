# Baseline results (starter run, 2026-11-13)

All figures below come from the first end-to-end run of the pipeline on the
UCI credit-card dataset (30,000 accounts; 21,000 development / 9,000 hold-out;
overall default rate 22.1%). Amounts are NT$ as in the source data.
These are YOUR starting point — re-run, challenge and improve them.

## PD model performance (hold-out test)

| Model | AUC | Gini | KS |
|---|---|---|---|
| Logistic scorecard (champion) | 0.759 | 0.519 | 0.404 |
| XGBoost (challenger) | 0.780 | 0.561 | 0.425 |
| Naive benchmark (PAY_0 only) | 0.682 | — | — |

## Validation (outcomes analysis, champion)
- Stability: PSI (train vs. test PD distribution) = 0.002 → stable.
- Calibration: Brier score 0.142; largest decile deviation 3.1 pp —
  deciles 6–7 *under-predict* defaults (predicted 14.9% / 18.6% vs. observed
  17.7% / 21.8%). This is a genuine finding for your validation report.
- Benchmarking: the challenger beats the champion by ~2 AUC points —
  your report must argue whether the interpretability/stability of the
  scorecard justifies keeping it (in banks, usually yes, with monitoring).

## Expected loss (test portfolio, CCF = 0.50)
| LGD | Portfolio EAD | Expected loss | EL / EAD |
|---|---|---|---|
| 0.70 | NT$988.5M | NT$124.6M | 12.6% |
| 0.85 | NT$988.5M | NT$151.3M | 15.3% |
| 0.95 | NT$988.5M | NT$169.1M | 17.1% |

EL concentrates in low-limit segments (avg PD 31.9% for limits ≤ NT$50k vs.
13.2% above NT$300k) — see el_by_limit_band.csv.

## Stress scenarios (illustrative satellite overlay, beta = 0.25)
| Scenario | Unemployment | Avg lifetime PD | EL / EAD |
|---|---|---|---|
| Baseline | 4.1% | 34.5% | 24.7% |
| Adverse | 7.1% | 50.2% | 37.6% |
| Severely adverse | 9.4% | 63.3% | 49.1% |

The beta is an assumption, not an estimate — replacing it with an estimated
satellite model is the single most valuable upgrade in this project.
