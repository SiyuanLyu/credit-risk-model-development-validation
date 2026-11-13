"""
Step 8 - ML extension: explainability & governance pack for the ML models
=========================================================================
Banks do not reject ML because it is ML; they reject it when it cannot be
explained, constrained, and monitored. This step produces the evidence a
validator asks for:

1. Monotonicity-constrained XGBoost - domain knowledge hard-coded as
   constraints (more delinquency/utilisation can only raise PD; a higher
   payment ratio can only lower it), then compared with the unconstrained
   challenger from Step 2.
2. SHAP global importance for the constrained model (test sample).
3. Adverse-action reason codes: for the highest-risk accounts, the top
   features pushing PD up, in plain language — the ECOA-style output a
   production credit model must produce.

Outputs
-------
outputs/monotone_xgb_metrics.json
outputs/shap_importance.csv
outputs/shap_summary.png
outputs/reason_codes_sample.csv
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"

FEATURES = [
    "LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "utilisation", "pay_ratio", "max_delinq", "n_delinq", "n_severe",
    "bill_trend", "avg_utilisation",
]
CAT_FEATURES = ["SEX", "EDUCATION", "MARRIAGE"]
ALL_FEATURES = FEATURES + CAT_FEATURES

# +1: feature may only increase PD; -1: only decrease; 0: unconstrained
MONOTONE = {
    "LIMIT_BAL": -1, "AGE": 0,
    "PAY_0": 1, "PAY_2": 1, "PAY_3": 1, "PAY_4": 1, "PAY_5": 1, "PAY_6": 1,
    "utilisation": 1, "pay_ratio": -1, "max_delinq": 1, "n_delinq": 1,
    "n_severe": 1, "bill_trend": 0, "avg_utilisation": 1,
    "SEX": 0, "EDUCATION": 0, "MARRIAGE": 0,
}
READABLE = {
    "PAY_0": "Most recent repayment status (delinquency)",
    "PAY_2": "Repayment status 2 months ago",
    "max_delinq": "Worst delinquency in past 6 months",
    "n_severe": "Months with 2+ months delay",
    "n_delinq": "Months with any payment delay",
    "utilisation": "Current credit-limit utilisation",
    "avg_utilisation": "Average utilisation, 6 months",
    "LIMIT_BAL": "Credit limit",
    "pay_ratio": "Payment-to-bill ratio",
}


def ks_statistic(y: np.ndarray, p: np.ndarray) -> float:
    order = np.argsort(p)[::-1]
    ys = y[order]
    return float(np.max(np.abs(
        np.cumsum(ys) / ys.sum() - np.cumsum(1 - ys) / (1 - ys).sum())))


def main() -> None:
    train = pd.read_csv(PROC / "train.csv")
    test = pd.read_csv(PROC / "test.csv")
    y_tr, y_te = train["default"].to_numpy(), test["default"].to_numpy()
    constraints = tuple(MONOTONE[f] for f in ALL_FEATURES)

    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
        random_state=42, monotone_constraints=constraints,
    )
    model.fit(train[ALL_FEATURES], y_tr)
    p_te = model.predict_proba(test[ALL_FEATURES])[:, 1]
    auc = float(roc_auc_score(y_te, p_te))
    metrics = {
        "model": "XGBoost with monotonicity constraints",
        "test": {"auc": auc, "gini": float(2 * auc - 1),
                 "ks": ks_statistic(y_te, p_te)},
        "reference_unconstrained_xgb_auc": 0.780,
        "constraints": {f: MONOTONE[f] for f in ALL_FEATURES},
    }
    (OUT / "monotone_xgb_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: metrics[k] for k in ("model", "test")}, indent=2))

    # ---- SHAP on a test sample ---------------------------------------------
    sample = test.sample(n=min(2000, len(test)), random_state=42)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(sample[ALL_FEATURES])
    sv = np.asarray(sv)
    if sv.ndim == 3:            # (n, features, classes) in newer shap
        sv = sv[:, :, 1] if sv.shape[2] > 1 else sv[:, :, 0]
    mean_abs = np.abs(sv).mean(axis=0)
    imp = pd.DataFrame({"feature": ALL_FEATURES,
                        "mean_abs_shap": mean_abs}).sort_values(
        "mean_abs_shap", ascending=False)
    imp.to_csv(OUT / "shap_importance.csv", index=False)

    top = imp.head(10).iloc[::-1]
    plt.figure(figsize=(7, 5))
    plt.barh(top["feature"], top["mean_abs_shap"])
    plt.xlabel("mean |SHAP value|")
    plt.title("SHAP global importance - monotone XGBoost (test sample)")
    plt.tight_layout(); plt.savefig(OUT / "shap_summary.png", dpi=150)

    # ---- Reason codes for the 20 highest-risk accounts ----------------------
    test = test.assign(pd_mono=p_te)
    worst = test.nlargest(20, "pd_mono")
    sv_worst = np.asarray(explainer.shap_values(worst[ALL_FEATURES]))
    if sv_worst.ndim == 3:
        sv_worst = sv_worst[:, :, 1] if sv_worst.shape[2] > 1 \
            else sv_worst[:, :, 0]
    rows = []
    for i, (_, acc) in enumerate(worst.iterrows()):
        contrib = pd.Series(sv_worst[i], index=ALL_FEATURES)
        top_up = contrib[contrib > 0].nlargest(4).index
        rows.append({
            "ID": int(acc["ID"]), "pd": round(float(acc["pd_mono"]), 3),
            "reason_1": READABLE.get(top_up[0], top_up[0]) if len(top_up) > 0 else "",
            "reason_2": READABLE.get(top_up[1], top_up[1]) if len(top_up) > 1 else "",
            "reason_3": READABLE.get(top_up[2], top_up[2]) if len(top_up) > 2 else "",
            "reason_4": READABLE.get(top_up[3], top_up[3]) if len(top_up) > 3 else "",
        })
    pd.DataFrame(rows).to_csv(OUT / "reason_codes_sample.csv", index=False)
    print(imp.head(8).round(4).to_string(index=False))
    print(pd.DataFrame(rows).head(5).to_string(index=False))


if __name__ == "__main__":
    main()
