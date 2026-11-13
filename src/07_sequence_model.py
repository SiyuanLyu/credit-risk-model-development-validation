"""
Step 7 - ML extension: sequence deep-learning PD model (challenger #2)
======================================================================
The scorecard and XGBoost models use hand-engineered summary features.
This model instead consumes the RAW 6-month behavioural sequence of each
account (repayment status, bill, payment per month, oldest -> newest) and
learns its own features with a GRU encoder — the behavioural-scoring
upgrade path banks are actively exploring.

Architecture: GRU(4 -> 32) over monthly features
              [pay_status, utilisation, payment/limit, delinquency flag]
              + static features (log limit, age) -> MLP head -> PD.

Governance note: this is a CHALLENGER. It must beat the champion on
discrimination AND survive the explainability review in Step 8 before
anyone should argue for promotion.

Outputs
-------
outputs/sequence_metrics.json     AUC / KS / Gini (test) + training info
outputs/predictions_test_ml.csv   ID, y, pd_gru  (merged by validation)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

MONTHS_OLD_TO_NEW = [6, 5, 4, 3, 2, 1]   # BILL_AMT6 is oldest, 1 is newest


def build_sequences(df: pd.DataFrame):
    """Return (seq [N,6,4], static [N,2]) arrays from a processed frame."""
    limit = df["LIMIT_BAL"].to_numpy(dtype=np.float64) + 1.0
    seqs = []
    for m in MONTHS_OLD_TO_NEW:
        # PAY columns are PAY_0 (newest) and PAY_2..PAY_6; month index m maps:
        # newest month (m=1) -> PAY_0, older months m>=2 -> PAY_m
        pay_col = "PAY_0" if m == 1 else f"PAY_{m}"
        pay = df[pay_col].to_numpy(dtype=np.float64)
        bill = df[f"BILL_AMT{m}"].to_numpy(dtype=np.float64)
        pamt = df[f"PAY_AMT{m}"].to_numpy(dtype=np.float64)
        seqs.append(np.stack([
            np.clip(pay, -1, 8) / 8.0,          # repayment status, scaled
            bill / limit,                        # utilisation that month
            pamt / limit,                        # payment relative to limit
            (pay >= 1).astype(np.float64),       # delinquency flag
        ], axis=1))
    seq = np.stack(seqs, axis=1).astype(np.float32)          # [N, 6, 4]
    static = np.stack([
        np.log(limit),
        df["AGE"].to_numpy(dtype=np.float64),
    ], axis=1)
    return seq, static


class GRUPD(nn.Module):
    def __init__(self, seq_dim=4, static_dim=2, hidden=32):
        super().__init__()
        self.gru = nn.GRU(seq_dim, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden + static_dim, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 1))

    def forward(self, seq, static):
        _, h = self.gru(seq)
        z = torch.cat([h[-1], static], dim=1)
        return self.head(z).squeeze(-1)


def ks_statistic(y: np.ndarray, p: np.ndarray) -> float:
    order = np.argsort(p)[::-1]
    ys = y[order]
    return float(np.max(np.abs(
        np.cumsum(ys) / ys.sum() - np.cumsum(1 - ys) / (1 - ys).sum())))


def main() -> None:
    train = pd.read_csv(PROC / "train.csv")
    test = pd.read_csv(PROC / "test.csv")
    X_tr, S_tr = build_sequences(train)
    X_te, S_te = build_sequences(test)
    y_tr = train["default"].to_numpy(dtype=np.float32)
    y_te = test["default"].to_numpy()

    # standardise static features on train only
    mu, sd = S_tr.mean(axis=0), S_tr.std(axis=0) + 1e-9
    S_tr = ((S_tr - mu) / sd).astype(np.float32)
    S_te = ((S_te - mu) / sd).astype(np.float32)

    idx_tr, idx_val = train_test_split(
        np.arange(len(y_tr)), test_size=0.15, random_state=SEED,
        stratify=y_tr)
    ds_tr = TensorDataset(torch.tensor(X_tr[idx_tr]),
                          torch.tensor(S_tr[idx_tr]),
                          torch.tensor(y_tr[idx_tr]))
    dl = DataLoader(ds_tr, batch_size=256, shuffle=True)

    model = GRUPD()
    pos_weight = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    def predict(idxs, X, S):
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(X[idxs]), torch.tensor(S[idxs]))
        return torch.sigmoid(logits).numpy()

    best_auc, best_state, patience = 0.0, None, 0
    for epoch in range(40):
        model.train()
        for xb, sb, yb in dl:
            opt.zero_grad()
            loss = loss_fn(model(xb, sb), yb)
            loss.backward()
            opt.step()
        val_p = predict(idx_val, X_tr, S_tr)
        val_auc = roc_auc_score(y_tr[idx_val], val_p)
        if val_auc > best_auc:
            best_auc, patience = val_auc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 6:
                print(f"early stop at epoch {epoch + 1}")
                break
    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pd_te = torch.sigmoid(model(torch.tensor(X_te),
                                    torch.tensor(S_te))).numpy()
    auc = float(roc_auc_score(y_te, pd_te))
    result = {
        "model": "GRU sequence challenger (raw 6-month behaviour)",
        "test": {"auc": auc, "gini": float(2 * auc - 1),
                 "ks": ks_statistic(y_te, pd_te)},
        "val_auc_best": float(best_auc),
        "reference": {"champion_logistic_auc": 0.759,
                      "challenger_xgboost_auc": 0.780},
    }
    (OUT / "sequence_metrics.json").write_text(json.dumps(result, indent=2))
    pd.DataFrame({"ID": test["ID"], "y": y_te, "pd_gru": pd_te}).to_csv(
        OUT / "predictions_test_ml.csv", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
