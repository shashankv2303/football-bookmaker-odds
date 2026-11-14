"""Train every model and save predictions for all matches.

Run:  python src/train.py            (all models)
      python src/train.py rf nn      (only some)
Outputs: results/predictions.csv, results/curves.json, results/params.json
"""
import json
import sys
import time

import numpy as np
import pandas as pd

from features import build_dataset
from models import nn, rf, svd
from models.common import TARGET, split_idx
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"

MODELS = {
    "rf": ("Random Forest", rf.run),
    "svd": ("SVD (collab. filtering)", svd.run),
    "nn": ("Neural Network", nn.run_nn),
    "lstm": ("LSTM", nn.run_lstm),
}


def main(names):
    RESULTS.mkdir(exist_ok=True)
    table, X_seq = build_dataset()
    pred_path = RESULTS / "predictions.csv"
    out = pd.read_csv(pred_path, parse_dates=["Date"]) if pred_path.exists() and len(names) < len(MODELS) \
        else table[["Date", "Season", "HomeTeam", "AwayTeam", TARGET, "split", "Avg>2.5", "Avg<2.5",
                    "B365>2.5", "B365<2.5"]].copy()
    curves, params = {}, {}
    for f, d in (("curves.json", curves), ("params.json", params)):
        if (RESULTS / f).exists():
            d.update(json.loads((RESULTS / f).read_text()))

    train_mean = table.loc[split_idx(table)["train"], TARGET].mean()
    out["pred_mean"] = train_mean
    for key in names:
        label, fn = MODELS[key]
        t0 = time.time()
        pred, extra = fn(table, X_seq)
        out[f"pred_{key}"] = pred
        params[key] = extra["params"]
        if "curve" in extra:
            curves[key] = extra["curve"]
        print(f"{label:30s} done in {time.time() - t0:5.1f}s  params={extra['params']}", flush=True)
    out.to_csv(pred_path, index=False)
    (RESULTS / "curves.json").write_text(json.dumps(curves))
    (RESULTS / "params.json").write_text(json.dumps(params, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:] or list(MODELS))
