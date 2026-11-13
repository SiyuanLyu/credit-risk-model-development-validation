# ML extension — baseline results (2026-11-13)

Four-way comparison on the same hold-out test set (9,000 accounts):

| Model | Role | Test AUC | Gini | KS |
|---|---|---|---|---|
| Logistic scorecard (WOE) | Champion | 0.759 | 0.519 | 0.404 |
| XGBoost (unconstrained) | Challenger #1 | 0.780 | 0.561 | 0.425 |
| GRU sequence model (raw 6-month behaviour) | Challenger #2 | 0.764 | 0.527 | 0.397 |
| XGBoost + monotonicity constraints | Governance variant | 0.765 | 0.530 | 0.405 |

## What the numbers say (read this before the interviews do)
- The GRU does NOT beat gradient boosting on this dataset. That is the
  expected result at this scale: 30k accounts x 6 monthly steps is small
  for deep learning, and the hand-engineered features already compress
  most of the signal. Deep sequence models earn their keep with
  transaction-level data and longer histories — be ready to say exactly
  that, and to name what extra data would change the conclusion.
- Monotonicity constraints cost ~1.5 AUC points (0.780 -> 0.765). That gap
  is the measurable "price of compliance": guaranteed directionally-sane
  responses to delinquency/utilisation, which validators and regulators
  actively reward. Arguing this trade-off with your own numbers is a
  stronger interview answer than quoting a textbook.
- SHAP global importance (monotone model): max_delinq (0.398),
  PAY_0 (0.374), LIMIT_BAL (0.204), bill_trend (0.182), pay_ratio (0.139)
  — consistent with the WOE/IV ranking in Step 2, a good consistency
  check between two very different importance measures.
- Reason codes work: the 20 highest-PD accounts get plain-language
  adverse-action reasons (outputs/reason_codes_sample.csv), dominated by
  recent and worst delinquency — exactly what an ECOA review expects.

## Deliberate open items (your practice tasks)
- Complete the plain-language mapping for every feature in reason codes
  (bill_trend and AGE still print raw names).
- Try a 12-feature-month transformer encoder or add attention and see if
  it closes the gap to XGBoost; document either outcome honestly.
- Add calibration + PSI for the GRU and monotone models to Step 4 so the
  validation report can judge all challengers on equal footing.
