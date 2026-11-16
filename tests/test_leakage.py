"""Features of a match must not depend on that match's (or any later match's) result.
Run: python tests/test_leakage.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import load
from features import build_dataset, feature_columns

df = load()
base, base_seq = build_dataset(df)
k = 2000                                    # corrupt the result (and stats) of match k and everything after it
mod = df.copy()
for c in ["FTHG", "FTAG", "HS", "AS", "HST", "AST", "HC", "AC"]:
    mod.loc[k:, c] = mod.loc[k:, c] + 7
new, new_seq = build_dataset(mod)

cols = feature_columns(base)
date_k = df.loc[k, "Date"]
before = base["Date"] < date_k              # strictly earlier matches
m = base.merge(new, on=["Date", "HomeTeam", "AwayTeam"], suffixes=("", "_n"))
b = m[m["Date"] < date_k]
for c in cols:
    assert np.allclose(b[c], b[c + "_n"]), f"leak in {c}"
# the match itself: its own features should also be unchanged by its own result
own = m[(m["Date"] == date_k)]
for c in cols:
    assert np.allclose(own[c], own[c + "_n"]), f"own-result leak in {c}"
print("OK: features of matches on/before the corrupted date are unaffected by their own or future results")
