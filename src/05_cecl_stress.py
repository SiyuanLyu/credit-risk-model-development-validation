"""
Step 5 - CECL-style loss forecasting & CCAR-style stress scenarios
==================================================================
Satellite-model overlay: portfolio PDs are shifted through the logit
link as a function of the unemployment-rate scenario path:

    logit(PD_scenario) = logit(PD_base) + beta * (UR_scenario - UR_base)

beta = 0.25 per +1pp of unemployment is an ILLUSTRATIVE assumption in
this starter version (flagged as TODO in the README: replace it with an
estimated coefficient from a time-series / panel regression, which is
exactly the kind of satellite-model work banks do).

Scenario deltas are styled after the Federal Reserve's CCAR supervisory
scenarios (baseline / adverse / severely adverse). Macro data: FRED.

Lifetime expected loss uses a simplified CECL flavour: the 1-period PD
is annualised over an assumed 2-year remaining life with a 0.9 survival
decay (documented simplification, to be refined).

Outputs
-------
outputs/stress_scenarios.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BETA_UR = 0.25          # TODO: estimate, do not hard-code, in the full version
LGD = 0.85
CCF = 0.50
LIFE_YEARS, DECAY = 2, 0.9


def latest_unrate() -> float:
    ur = pd.read_csv(ROOT / "data" / "raw" / "fred_UNRATE.csv")
    ur.columns = ["date", "unrate"]
    return float(ur["unrate"].dropna().iloc[-1])


def main() -> None:
    test = pd.read_csv(ROOT / "data" / "processed" / "test.csv")
    pred = pd.read_csv(ROOT / "outputs" / "predictions_test.csv")
    df = test.merge(pred, on="ID", how="inner")
    drawn = df["BILL_AMT1"].clip(lower=0)
    df["ead"] = drawn + CCF * (df["LIMIT_BAL"] - drawn).clip(lower=0)

    ur0 = latest_unrate()
    scenarios = {
        "baseline": ur0,
        "adverse": ur0 + 3.0,
        "severely_adverse": ur0 + 5.3,
    }
    p = df["pd_logit"].to_numpy()
    logit_p = np.log(p / (1 - p))
    rows = []
    for name, ur in scenarios.items():
        p_adj = 1 / (1 + np.exp(-(logit_p + BETA_UR * (ur - ur0))))
        # simplified lifetime PD over LIFE_YEARS with survival decay
        p_life = 1 - np.prod(
            [1 - p_adj * (DECAY ** t) for t in range(LIFE_YEARS)], axis=0)
        el = float(np.sum(p_life * LGD * df["ead"]))
        rows.append({
            "scenario": name, "unemployment_rate": round(ur, 2),
            "avg_pd_pct": float(100 * p_adj.mean()),
            "lifetime_pd_pct": float(100 * p_life.mean()),
            "expected_loss": el,
            "el_rate_on_ead_pct": float(100 * el / df["ead"].sum()),
        })
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "outputs" / "stress_scenarios.csv", index=False)
    print(f"Latest FRED unemployment rate used as baseline: {ur0:.1f}%")
    print(out.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
