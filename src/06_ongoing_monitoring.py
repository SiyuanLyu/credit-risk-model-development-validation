"""
Step 6 - Ongoing monitoring (SR 11-7: models must be monitored after deployment)
================================================================================
Simulates a new "production" batch by resampling the hold-out portfolio —
optionally with a drift shock (higher delinquency mix) — scores it with the
frozen development predictions, and reports:

- PSI of the PD distribution vs. the development sample
  (<0.10 stable / 0.10-0.25 watch / >0.25 breach)
- Observed vs. predicted default rate for the batch (calibration tracking)

Exit code is 1 when PSI breaches 0.25, so the weekly GitHub Actions job can
fail loudly and open an issue — the same escalation a bank monitoring team
would trigger.

Usage: python src/06_ongoing_monitoring.py [--drift 0.0] [--seed 7]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    edges = np.quantile(expected, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e = np.where(e == 0, 1e-4, e)
    a = np.where(a == 0, 1e-4, a)
    return float(np.sum((a - e) * np.log(a / e)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drift", type=float, default=0.0,
                    help="share of batch swapped for high-delinquency accounts")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    dev = pd.read_csv(OUT / "predictions_train.csv")
    prod_pool = pd.read_csv(OUT / "predictions_test.csv")
    rng = np.random.default_rng(args.seed)

    batch = prod_pool.sample(frac=0.5, random_state=args.seed)
    if args.drift > 0:
        n_swap = int(len(batch) * args.drift)
        risky = prod_pool.nlargest(int(len(prod_pool) * 0.2), "pd_logit")
        batch = pd.concat([batch.iloc[n_swap:],
                           risky.sample(n_swap, random_state=args.seed)])
    batch = batch.reset_index(drop=True)

    psi_val = psi(dev["pd_logit"].to_numpy(), batch["pd_logit"].to_numpy())
    status = ("stable" if psi_val < 0.10 else
              "watch" if psi_val <= 0.25 else "BREACH")
    report = {
        "batch_n": int(len(batch)), "drift": args.drift,
        "psi": round(psi_val, 4), "status": status,
        "pred_default_rate": round(float(batch["pd_logit"].mean()), 4),
        "obs_default_rate": round(float(batch["y"].mean()), 4),
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "monitoring_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if status == "BREACH":
        raise SystemExit("PSI breach (>0.25): escalate per monitoring plan.")


if __name__ == "__main__":
    main()
