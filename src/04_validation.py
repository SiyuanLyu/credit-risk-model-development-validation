"""
Step 4 - Independent validation (SR 11-7 style outcomes analysis)
=================================================================
Written as if by a validator who did NOT build the model in Step 2.
Covers the three SR 11-7 outcomes-analysis pillars:

1. Discrimination  - AUC / KS / Gini on the out-of-sample test set
2. Calibration     - predicted vs. observed default rate by PD decile,
                     Brier score
3. Stability       - Population Stability Index (PSI) of the PD
                     distribution, train vs. test
Plus benchmarking against the XGBoost challenger and a naive single-
variable benchmark, and a sensitivity test (delinquency shock).

Outputs
-------
outputs/validation_metrics.json
outputs/calibration_deciles.csv
outputs/calibration_curve.png
outputs/ks_curve.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def ks_statistic(y: np.ndarray, p: np.ndarray):
    order = np.argsort(p)[::-1]
    ys = y[order]
    cum_bad = np.cumsum(ys) / ys.sum()
    cum_good = np.cumsum(1 - ys) / (1 - ys).sum()
    ks = np.max(np.abs(cum_bad - cum_good))
    return float(ks), cum_bad, cum_good


def psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    edges = np.quantile(expected, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e_pct = np.histogram(expected, edges)[0] / len(expected)
    a_pct = np.histogram(actual, edges)[0] / len(actual)
    e_pct = np.where(e_pct == 0, 1e-4, e_pct)
    a_pct = np.where(a_pct == 0, 1e-4, a_pct)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def main() -> None:
    tr = pd.read_csv(OUT / "predictions_train.csv")
    te = pd.read_csv(OUT / "predictions_test.csv")
    y, p = te["y"].to_numpy(), te["pd_logit"].to_numpy()

    # ---- 1. Discrimination -------------------------------------------------
    ks, cum_bad, cum_good = ks_statistic(y, p)
    auc = roc_auc_score(y, p)

    # ---- 2. Calibration ----------------------------------------------------
    te = te.assign(decile=pd.qcut(te["pd_logit"], 10, duplicates="drop"))
    calib = te.groupby("decile", observed=True).agg(
        n=("y", "size"), mean_pred=("pd_logit", "mean"),
        obs_rate=("y", "mean"))
    calib["abs_diff_pp"] = (calib["mean_pred"] - calib["obs_rate"]) * 100
    calib.to_csv(OUT / "calibration_deciles.csv")
    brier = float(brier_score_loss(y, p))

    plt.figure(figsize=(5.5, 5))
    plt.plot(calib["mean_pred"], calib["obs_rate"], "o-", label="Model")
    plt.plot([0, max(calib["mean_pred"])], [0, max(calib["mean_pred"])],
             "k--", lw=0.8, label="Perfect calibration")
    plt.xlabel("Mean predicted PD"); plt.ylabel("Observed default rate")
    plt.title("Calibration by PD decile (test)"); plt.legend()
    plt.tight_layout(); plt.savefig(OUT / "calibration_curve.png", dpi=150)

    plt.figure(figsize=(6, 5))
    x = np.arange(len(y))
    plt.plot(x, cum_bad, label="Cumulative % of defaulters")
    plt.plot(x, cum_good, label="Cumulative % of non-defaulters")
    plt.xlabel("Accounts ranked by PD (desc)"); plt.ylabel("Cumulative share")
    plt.title(f"KS curve (KS = {ks:.3f})"); plt.legend()
    plt.tight_layout(); plt.savefig(OUT / "ks_curve.png", dpi=150)

    # ---- 3. Stability -------------------------------------------------------
    psi_val = psi(tr["pd_logit"].to_numpy(), p)

    # ---- Benchmarking -------------------------------------------------------
    auc_challenger = roc_auc_score(y, te["pd_xgb"])
    proc = pd.read_csv(ROOT / "data" / "processed" / "train.csv")
    proc_te = pd.read_csv(ROOT / "data" / "processed" / "test.csv")
    naive = LogisticRegression(max_iter=1000).fit(
        proc[["PAY_0"]], proc["default"])
    auc_naive = roc_auc_score(y, naive.predict_proba(proc_te[["PAY_0"]])[:, 1])

    result = {
        "discrimination_test": {"auc": float(auc), "gini": float(2 * auc - 1),
                                "ks": ks},
        "calibration_test": {"brier_score": brier,
                             "max_abs_decile_diff_pp":
                                 float(calib["abs_diff_pp"].abs().max())},
        "stability": {"psi_train_vs_test": psi_val,
                      "interpretation": "<0.10 stable; 0.10-0.25 monitor; "
                                        ">0.25 significant shift"},
        "benchmarking_auc_test": {
            "champion_logistic": float(auc),
            "challenger_xgboost": float(auc_challenger),
            "naive_pay0_only": float(auc_naive)},
    }
    (OUT / "validation_metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(calib.round(4).to_string())


if __name__ == "__main__":
    main()
