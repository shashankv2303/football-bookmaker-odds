"""Leakage-safe feature building.

For every match we only look at matches that were played BEFORE it:
  * each team's last 3 and last 5 matches (goals for/against, shots, shots on target, corners)
  * season-to-date averages of both teams (the paper's "opponent strength" idea)
  * days since each team's previous match (the paper's fatigue idea)
  * for the LSTM/GRU: the last SEQ_LEN matches of each team as an ordered sequence

Bookmaker odds are NOT used as features. They are the benchmark only.
"""
import numpy as np
import pandas as pd

from data import load

SEQ_LEN = 5
STATS = ["gf", "ga", "sh", "sot", "cor"]          # per-team-per-match stats
SEQ_COLS = STATS + ["home", "rest"]               # per-timestep inputs for the sequence model
TRAIN_END, VAL_SEASON, TEST_SEASON = 2022, 2023, 2024   # time-based split


def long_format(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, match) with that team's own stats."""
    h = pd.DataFrame({"mid": df.index, "Date": df["Date"], "team": df["HomeTeam"], "Season": df["Season"],
                      "gf": df["FTHG"], "ga": df["FTAG"], "sh": df["HS"], "sot": df["HST"],
                      "cor": df["HC"], "home": 1})
    a = pd.DataFrame({"mid": df.index, "Date": df["Date"], "team": df["AwayTeam"], "Season": df["Season"],
                      "gf": df["FTAG"], "ga": df["FTHG"], "sh": df["AS"], "sot": df["AST"],
                      "cor": df["AC"], "home": 0})
    return pd.concat([h, a]).sort_values(["team", "Date", "mid"], kind="stable").reset_index(drop=True)


def team_history(lf: pd.DataFrame) -> pd.DataFrame:
    """Rolling / expanding stats per team, shifted by one match so a row never sees its own result."""
    g = lf.groupby("team", sort=False)
    out = lf[["mid", "team", "home"]].copy()
    for s in STATS:
        prev = g[s].shift(1)
        grp = prev.groupby(lf["team"], sort=False)
        out[f"{s}_l3"] = grp.transform(lambda x: x.rolling(3, min_periods=3).mean())
        out[f"{s}_l5"] = grp.transform(lambda x: x.rolling(5, min_periods=5).mean())
        # season-to-date mean of previous matches only
        out[f"{s}_std"] = prev.groupby([lf["team"], lf["Season"]], sort=False).transform(
            lambda x: x.expanding(min_periods=1).mean())
    out["rest"] = g["Date"].diff().dt.days.clip(upper=30)
    return out


def sequences(lf: pd.DataFrame) -> dict:
    """For each (team, match): array of the previous SEQ_LEN matches, oldest first."""
    seqs = {}
    lf = lf.copy()
    lf["rest"] = lf.groupby("team", sort=False)["Date"].diff().dt.days.clip(upper=30)
    for team, g in lf.groupby("team", sort=False):
        arr = g[SEQ_COLS].to_numpy(dtype=float)
        mids = g["mid"].to_numpy()
        for i in range(SEQ_LEN, len(g)):
            seqs[(team, mids[i])] = arr[i - SEQ_LEN:i]
    return seqs


def build_dataset(df: pd.DataFrame | None = None):
    """Return (table, X_seq). `table` has features, targets, odds benchmark and a `split` column."""
    df = (load() if df is None else df).reset_index(drop=True)
    lf = long_format(df)
    hist = team_history(lf)
    feat_cols = [c for c in hist.columns if c not in ("mid", "team", "home")]

    home = hist[hist["home"] == 1].set_index("mid")[feat_cols].add_prefix("h_")
    away = hist[hist["home"] == 0].set_index("mid")[feat_cols].add_prefix("a_")
    table = df.join(home).join(away)
    table["split"] = np.where(table["Season"] <= TRAIN_END, "train",
                      np.where(table["Season"] == VAL_SEASON, "val", "test"))

    seqs = sequences(lf)
    X_seq = np.full((len(table), 2, SEQ_LEN, len(SEQ_COLS)), np.nan)
    for i, row in table[["HomeTeam", "AwayTeam"]].iterrows():
        hs, as_ = seqs.get((row.HomeTeam, i)), seqs.get((row.AwayTeam, i))
        if hs is not None and as_ is not None:
            X_seq[i, 0], X_seq[i, 1] = hs, as_

    # keep only matches where both teams have >= SEQ_LEN prior matches (drops first games of the dataset
    # and the first 5 games of newly promoted teams)
    keep = ~np.isnan(X_seq).any(axis=(1, 2, 3)) & table.filter(regex="^[ha]_").notna().all(axis=1).to_numpy()
    table, X_seq = table[keep].reset_index(drop=True), X_seq[keep]
    return table, X_seq


def feature_columns(table: pd.DataFrame) -> list[str]:
    return [c for c in table.columns if c[:2] in ("h_", "a_")]


if __name__ == "__main__":
    t, xs = build_dataset()
    print(t.shape, xs.shape, t["split"].value_counts().to_dict())
    print(len(feature_columns(t)), "tabular features")
