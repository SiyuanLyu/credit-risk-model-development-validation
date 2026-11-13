"""
Step 3 - LGD / EAD and portfolio Expected Loss (EL)
==================================================
EAD (Exposure at Default) for a revolving credit-card account:
    EAD = drawn balance + CCF x undrawn commitment
with a base-case Credit Conversion Factor CCF = 0.50 (documented assumption).

LGD (Loss Given Default) has no recovery data in this public dataset, so it
is treated as a scenario parameter for unsecured revolving credit:
base 0.85, sensitivity range 0.70 - 0.95.

Expected Loss: EL = PD x LGD x EAD   (account level, summed to portfolio)

Outputs
-------
outputs/el_summary.csv   Portfolio EL by LGD scenario + EL by limit segment
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CCF = 0.50
LGD_SCENARIOS = [0.70, 0.85, 0.95]


def main() -> None:
    test = pd.read_csv(ROOT / "data" / "processed" / "test.csv")
    pred = pd.read_csv(ROOT / "outputs" / "predictions_test.csv")
    df = test.merge(pred, on="ID", how="inner")

    drawn = df["BILL_AMT1"].clip(lower=0)
    undrawn = (df["LIMIT_BAL"] - drawn).clip(lower=0)
    df["ead"] = drawn + CCF * undrawn

    rows = []
    for lgd in LGD_SCENARIOS:
        df["el"] = df["pd_logit"] * lgd * df["ead"]
        rows.append({
            "lgd": lgd, "ccf": CCF,
            "portfolio_ead": float(df["ead"].sum()),
            "portfolio_el": float(df["el"].sum()),
            "el_rate_on_ead_pct": float(100 * df["el"].sum() / df["ead"].sum()),
        })
    summary = pd.DataFrame(rows)

    # EL concentration by credit-limit segment (base LGD)
    df["el"] = df["pd_logit"] * 0.85 * df["ead"]
    df["limit_band"] = pd.cut(
        df["LIMIT_BAL"], [0, 50_000, 100_000, 200_000, 300_000, np.inf],
        labels=["<=50k", "50-100k", "100-200k", "200-300k", ">300k"])
    seg = df.groupby("limit_band", observed=True).agg(
        n=("ID", "count"), ead=("ead", "sum"), el=("el", "sum"),
        avg_pd=("pd_logit", "mean"), obs_default_rate=("y", "mean"))
    seg.to_csv(ROOT / "outputs" / "el_by_limit_band.csv")
    summary.to_csv(ROOT / "outputs" / "el_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(seg.round(3).to_string())


if __name__ == "__main__":
    main()
