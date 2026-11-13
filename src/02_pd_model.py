"""
Step 2 - PD model development (champion vs. challenger)
=======================================================
Champion  : Logistic-regression credit scorecard on WOE-transformed features
            (the industry-standard, regulator-friendly PD model).
Challenger: XGBoost gradient-boosted trees on the raw features
            (used for benchmarking in the validation step).

Outputs
-------
outputs/pd_metrics.json          AUC / KS / Gini for both models (train & test)
outputs/woe_iv.csv               WOE bins and Information Value per feature
outputs/predictions_train.csv    ID, y, pd_logit, pd_xgb  (train)
outputs/predictions_test.csv     ID, y, pd_logit, pd_xgb  (test / OOT)
outputs/roc_curve.png            ROC comparison plot
outputs/scorecard_points.csv     Scorecard points per WOE bin (PDO scaling)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "utilisation", "pay_ratio", "max_delinq", "n_delinq", "n_severe",
    "bill_trend", "avg_utilisation",
]
CAT_FEATURES = ["SEX", "EDUCATION", "MARRIAGE"]
ALL_FEATURES = FEATURES + CAT_FEATURES


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(y_score)[::-1]
    y_sorted = y_true[order]
    cum_bad = np.cumsum(y_sorted) / max(y_sorted.sum(), 1)
    cum_good = np.cumsum(1 - y_sorted) / max((1 - y_sorted).sum(), 1)
    return float(np.max(np.abs(cum_bad - cum_good)))


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    auc = roc_auc_score(y, p)
    return {"auc": float(auc), "gini": float(2 * auc - 1),
            "ks": ks_statistic(y, p)}


# ---------------------------------------------------------------- WOE binning
def woe_bins(series: pd.Series, y: pd.Series, n_bins: int = 10) -> pd.DataFrame:
    """Quantile-bin a numeric feature and compute WOE / IV against target y."""
    try:
        binned = pd.qcut(series, q=n_bins, duplicates="drop")
    except ValueError:
        binned = pd.cut(series, bins=n_bins)
    tmp = pd.DataFrame({"bin": binned, "y": y})
    grp = tmp.groupby("bin", observed=True)["y"].agg(["count", "sum"])
    grp.columns = ["n", "n_bad"]
    grp["n_good"] = grp["n"] - grp["n_bad"]
    total_bad, total_good = grp["n_bad"].sum(), grp["n_good"].sum()
    grp["bad_rate"] = grp["n_bad"] / grp["n"]
    grp["woe"] = np.log(
        ((grp["n_good"] + 0.5) / (total_good + 0.5 * len(grp)))
        / ((grp["n_bad"] + 0.5) / (total_bad + 0.5 * len(grp)))
    )
    grp["iv_component"] = (
        (grp["n_good"] / total_good - grp["n_bad"] / total_bad) * grp["woe"]
    )
    grp["edges"] = [f"({iv.left:.4g}, {iv.right:.4g}]" for iv in grp.index]
    return grp.reset_index(drop=True)


def apply_woe(series: pd.Series, table: pd.DataFrame) -> np.ndarray:
    """Map raw values to WOE using bin edges stored as intervals."""
    edges = [-np.inf]
    for e in table["edges"]:
        right = float(e.split(",")[1].strip(" ]"))
        edges.append(right)
    edges[-1] = np.inf
    idx = np.digitize(series.to_numpy(dtype=float), edges[1:-1])
    return table["woe"].to_numpy()[idx]


def main() -> None:
    train = pd.read_csv(PROC / "train.csv")
    test = pd.read_csv(PROC / "test.csv")
    y_tr, y_te = train["default"].to_numpy(), test["default"].to_numpy()

    # --- WOE transform (fit on train only; no leakage) ----------------------
    woe_tables, iv_rows = {}, []
    X_tr_woe, X_te_woe = {}, {}
    for f in FEATURES:
        tbl = woe_bins(train[f], train["default"])
        woe_tables[f] = tbl
        iv_rows.append({"feature": f, "iv": float(tbl["iv_component"].sum())})
        X_tr_woe[f] = apply_woe(train[f], tbl)
        X_te_woe[f] = apply_woe(test[f], tbl)
    pd.DataFrame(iv_rows).sort_values("iv", ascending=False).to_csv(
        OUT / "woe_iv.csv", index=False)

    Xtr = pd.DataFrame(X_tr_woe)
    Xte = pd.DataFrame(X_te_woe)
    for c in CAT_FEATURES:  # keep categoricals as dummies for the scorecard
        for df_, target in ((train, Xtr), (test, Xte)):
            d = pd.get_dummies(df_[c], prefix=c, drop_first=True, dtype=float)
            for col in d.columns:
                target[col] = d[col].to_numpy()
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0.0)

    # --- Champion: logistic scorecard ---------------------------------------
    logit = LogisticRegression(max_iter=1000, C=1.0)
    logit.fit(Xtr, y_tr)
    pd_tr_logit = logit.predict_proba(Xtr)[:, 1]
    pd_te_logit = logit.predict_proba(Xte)[:, 1]

    # Scorecard points with standard PDO scaling (PDO=20, 600 @ odds 1:50)
    factor, offset = 20 / np.log(2), 600 - (20 / np.log(2)) * np.log(50)
    coef = dict(zip(Xtr.columns, logit.coef_[0]))
    base_pts = (offset - factor * logit.intercept_[0]) / Xtr.shape[1]
    rows = []
    for f in FEATURES:
        for _, r in woe_tables[f].iterrows():
            rows.append({
                "feature": f, "bin": r["edges"], "woe": round(float(r["woe"]), 4),
                "points": round(base_pts - factor * coef[f] * float(r["woe"]), 1),
            })
    pd.DataFrame(rows).to_csv(OUT / "scorecard_points.csv", index=False)

    # --- Challenger: XGBoost on raw features ---------------------------------
    xgb = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
        random_state=42,
    )
    xgb.fit(train[ALL_FEATURES], y_tr)
    pd_tr_xgb = xgb.predict_proba(train[ALL_FEATURES])[:, 1]
    pd_te_xgb = xgb.predict_proba(test[ALL_FEATURES])[:, 1]

    # --- Save predictions & metrics ------------------------------------------
    pd.DataFrame({"ID": train["ID"], "y": y_tr,
                  "pd_logit": pd_tr_logit, "pd_xgb": pd_tr_xgb}
                 ).to_csv(OUT / "predictions_train.csv", index=False)
    pd.DataFrame({"ID": test["ID"], "y": y_te,
                  "pd_logit": pd_te_logit, "pd_xgb": pd_te_xgb}
                 ).to_csv(OUT / "predictions_test.csv", index=False)

    result = {
        "champion_logistic": {"train": metrics(y_tr, pd_tr_logit),
                              "test": metrics(y_te, pd_te_logit)},
        "challenger_xgboost": {"train": metrics(y_tr, pd_tr_xgb),
                               "test": metrics(y_te, pd_te_xgb)},
    }
    (OUT / "pd_metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

    # ROC plot
    plt.figure(figsize=(6, 5))
    for name, p in [("Logistic scorecard", pd_te_logit), ("XGBoost", pd_te_xgb)]:
        fpr, tpr, _ = roc_curve(y_te, p)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_te, p):.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC - test (out-of-sample)"); plt.legend(); plt.tight_layout()
    plt.savefig(OUT / "roc_curve.png", dpi=150)


if __name__ == "__main__":
    main()
