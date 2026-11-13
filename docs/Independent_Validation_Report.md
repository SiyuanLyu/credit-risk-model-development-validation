# Independent Validation Report
## Retail Credit Card PD Model — SR 11-7 Framework — v0.1 (starter draft)

*Write this as if you were a validator in Model Risk Management who did not
build the model. Your job is to challenge it, not to sell it.*

### 1. Scope and approach
Validation covers conceptual soundness, outcomes analysis and benchmarking,
per SR 11-7. Evidence: outputs/validation_metrics.json,
outputs/calibration_deciles.csv, calibration_curve.png, ks_curve.png.

### 2. Conceptual soundness
TODO — assess: Is logistic regression + WOE appropriate for this portfolio?
Are the features economically sensible (sign of coefficients vs. intuition)?
Is the default definition consistent? Would you approve the CCF and LGD
assumptions? Challenge at least the macro-overlay beta.

### 3. Outcomes analysis
- Discrimination: TODO — report test AUC / KS / Gini and judge against
  typical acceptance ranges (e.g. KS > 0.30 for retail scorecards).
- Calibration: TODO — use calibration_deciles.csv: which deciles deviate,
  by how many percentage points; is the model conservative or aggressive?
- Stability: TODO — interpret the PSI value using the thresholds printed
  in validation_metrics.json.

### 4. Benchmarking
TODO — compare champion vs. XGBoost challenger vs. naive PAY_0 benchmark.
If the challenger wins materially, explain why the champion should still
(or should not) be approved: interpretability, stability, governance.

### 5. Findings
| # | Severity | Finding | Recommendation |
|---|----------|---------|----------------|
| F1 | TODO | e.g. LGD is assumed, not estimated | Estimate LGD when recovery data is available; keep sensitivity range in reporting |
| F2 | TODO | (add your own) | |

### 6. Conclusion
TODO — one paragraph: approve / approve with conditions / reject, and why.
