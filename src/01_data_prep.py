"""
Step 1 - Data preparation
=========================
Loads the UCI "Default of Credit Card Clients" dataset (30,000 accounts),
cleans known coding quirks, engineers credit-risk features, and creates a
stratified train / test split.

Outputs
-------
data/processed/train.csv
data/processed/test.csv
data/processed/data_profile.json   (row counts, default rates)

Data source: Yeh & Lien (2009), UCI Machine Learning Repository #350.
Target: `default` = 1 if the account defaulted in the following month.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "default_of_credit_card_clients.xls"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYAMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]


def load_raw() -> pd.DataFrame:
    df = pd.read_excel(RAW, header=1, engine="calamine")
    df = df.rename(columns={"default payment next month": "default"})
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Known coding quirks in this dataset (documented in the literature):
    # EDUCATION 0/5/6 are undocumented codes -> group with 4 ("others")
    df.loc[df["EDUCATION"].isin([0, 5, 6]), "EDUCATION"] = 4
    # MARRIAGE 0 is undocumented -> group with 3 ("others")
    df.loc[df["MARRIAGE"] == 0, "MARRIAGE"] = 3
    # Repayment status -2/-1 means "no consumption / paid in full"; floor at -1
    for c in PAY_COLS:
        df[c] = df[c].clip(lower=-1)
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eps = 1.0
    # Utilisation of the credit limit (current month)
    df["utilisation"] = df["BILL_AMT1"] / (df["LIMIT_BAL"] + eps)
    # Payment ratio: amount paid vs. previous bill (behavioural risk signal)
    df["pay_ratio"] = df["PAY_AMT1"] / (df["BILL_AMT2"].abs() + eps)
    df["pay_ratio"] = df["pay_ratio"].clip(-1, 5)
    # Delinquency behaviour over the 6 observed months
    pay = df[PAY_COLS]
    df["max_delinq"] = pay.max(axis=1)              # worst months-past-due
    df["n_delinq"] = (pay >= 1).sum(axis=1)         # months with any delay
    df["n_severe"] = (pay >= 2).sum(axis=1)         # months 2+ months delayed
    # Bill trend: most recent bill vs. 6-month average bill
    df["bill_trend"] = df["BILL_AMT1"] / (df[BILL_COLS].mean(axis=1).abs() + eps)
    # Average utilisation over 6 months
    df["avg_utilisation"] = df[BILL_COLS].mean(axis=1) / (df["LIMIT_BAL"] + eps)
    return df


def main() -> None:
    df = engineer(clean(load_raw()))
    train, test = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df["default"]
    )
    train.to_csv(OUT / "train.csv", index=False)
    test.to_csv(OUT / "test.csv", index=False)
    profile = {
        "n_total": int(len(df)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "default_rate_total": float(df["default"].mean()),
        "default_rate_train": float(train["default"].mean()),
        "default_rate_test": float(test["default"].mean()),
        "n_features_total": int(df.shape[1]),
    }
    (OUT / "data_profile.json").write_text(json.dumps(profile, indent=2))
    print(json.dumps(profile, indent=2))


if __name__ == "__main__":
    main()
