"""Download and load English Premier League results + bookmaker odds.

Source: https://www.football-data.co.uk (one CSV per season).
Run:  python src/data.py        -> downloads CSVs into data/raw/ and writes data/matches.csv
"""
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "matches.csv"

# season start years; 2014 -> 2014/15 ... 2024 -> 2024/25
SEASONS = list(range(2014, 2025))

# columns we keep. Odds columns are used ONLY as a benchmark, never as features.
KEEP = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
        "HS", "AS", "HST", "AST", "HC", "AC",
        "B365>2.5", "B365<2.5", "Avg>2.5", "Avg<2.5"]


def season_code(year: int) -> str:
    return f"{year % 100:02d}{(year + 1) % 100:02d}"


def download(year: int) -> Path:
    path = RAW / f"E0_{season_code(year)}.csv"
    if path.exists() and path.stat().st_size > 1000:
        return path
    url = f"https://www.football-data.co.uk/mmz4281/{season_code(year)}/E0.csv"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    path.write_bytes(r.content)
    return path


def load_season(year: int) -> pd.DataFrame:
    df = pd.read_csv(download(year), encoding="latin-1")
    df = df.dropna(subset=["HomeTeam", "FTHG"])
    # older seasons name the market-average over/under odds BbAv; unify to Avg
    for new, old in (("Avg>2.5", "BbAv>2.5"), ("Avg<2.5", "BbAv<2.5")):
        if new not in df.columns and old in df.columns:
            df[new] = df[old]
    for c in KEEP:  # older seasons may lack some odds columns
        if c not in df.columns:
            df[c] = float("nan")
    df = df[KEEP].copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")
    df["Season"] = year
    return df


def build() -> pd.DataFrame:
    RAW.mkdir(parents=True, exist_ok=True)
    df = pd.concat([load_season(y) for y in SEASONS], ignore_index=True)
    df = df.sort_values("Date", kind="stable").reset_index(drop=True)
    df["TotalGoals"] = df["FTHG"] + df["FTAG"]
    df.to_csv(OUT, index=False)
    return df


def load() -> pd.DataFrame:
    """Return the cleaned match table (builds it on first use)."""
    if not OUT.exists():
        return build()
    return pd.read_csv(OUT, parse_dates=["Date"])


if __name__ == "__main__":
    d = build()
    print(d.shape, d["Season"].value_counts().sort_index().to_dict())
    print(d[["B365>2.5", "Avg>2.5"]].notna().mean())
