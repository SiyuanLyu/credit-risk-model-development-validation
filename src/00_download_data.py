"""
Step 0 - Download public data (run once; data files are NOT committed to git)
=============================================================================
- UCI "Default of Credit Card Clients" (.xls, ~5.5 MB)
- FRED macro series via the public fredgraph CSV endpoint (no API key):
  UNRATE (unemployment), GDPC1 (real GDP), CSUSHPISA (house prices),
  PAYEMS (nonfarm payrolls)
"""
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

UCI_URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/00350/"
           "default%20of%20credit%20card%20clients.xls")
FRED_IDS = ["UNRATE", "GDPC1", "CSUSHPISA", "PAYEMS"]


def fetch(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"skip (exists): {dest.name}")
        return
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    dest.write_bytes(r.content)
    print(f"downloaded: {dest.name} ({len(r.content):,} bytes)")


def main() -> None:
    fetch(UCI_URL, RAW / "default_of_credit_card_clients.xls")
    for sid in FRED_IDS:
        fetch(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}",
              RAW / f"fred_{sid}.csv")


if __name__ == "__main__":
    main()
