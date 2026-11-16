"""Shared helpers for all models."""
import numpy as np

TARGET = "TotalGoals"


def mse(y, p):
    return float(np.mean((np.asarray(y) - np.asarray(p)) ** 2))


def split_idx(table):
    s = table["split"].to_numpy()
    return {k: np.where(s == k)[0] for k in ("train", "val", "test")}


def pick_best(candidates, table, fit_predict):
    """Tiny grid search. `fit_predict(params)` returns predictions for ALL rows;
    choose the params with the lowest validation MSE (never look at test)."""
    idx = split_idx(table)
    y = table[TARGET].to_numpy()
    best = None
    for params in candidates:
        pred = fit_predict(params)
        v = mse(y[idx["val"]], pred[idx["val"]])
        if best is None or v < best[0]:
            best = (v, params, pred)
    return best[1], best[2]
