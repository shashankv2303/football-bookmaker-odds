"""Evaluate all models against the bookmaker on the held-out test season.

Run:  python src/evaluate.py     -> results/metrics.csv and results/*.png

Bookmaker benchmark: the market-average Over/Under 2.5 odds are turned into a probability of
"over" (margin removed), then into an implied expected-goals value lambda by solving
P(Poisson(lambda) >= 3) = p_over. That lets us compare models and bookmaker on the same MSE scale.
Our models' predictions are converted to P(over 2.5) the same way (Poisson assumption).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq

RESULTS = Path(__file__).resolve().parents[1] / "results"
LABELS = {"mean": "Train mean", "rf": "Random Forest", "svd": "SVD (collab. filt.)", "nn": "Neural Net",
          "lstm": "LSTM", "book": "Bookmaker"}


def p_over(lam):
    lam = np.asarray(lam, dtype=float)
    return 1 - np.exp(-lam) * (1 + lam + lam ** 2 / 2)


def implied_lambda(p):
    return brentq(lambda l: p_over(l) - p, 0.05, 12)


def evaluate(df, models, odds_over="Avg>2.5", odds_under="Avg<2.5", edge=0.05):
    d = df.dropna(subset=[odds_over, odds_under]).copy()
    pb = (1 / d[odds_over]) / (1 / d[odds_over] + 1 / d[odds_under])        # margin-free book probability
    d["pred_book"] = [implied_lambda(p) for p in pb]
    y, over = d["TotalGoals"].to_numpy(), (d["TotalGoals"] > 2.5).to_numpy()
    rows = []
    for m in models + ["book"]:
        pred = d[f"pred_{m}"].to_numpy()
        p = p_over(pred)
        # value betting: stake 1 on Over / Under when our probability x odds exceeds 1 + edge
        bet_o, bet_u = p * d[odds_over] > 1 + edge, (1 - p) * d[odds_under] > 1 + edge
        profit = np.where(bet_o, np.where(over, d[odds_over] - 1, -1), 0) + \
                 np.where(bet_u, np.where(~over, d[odds_under] - 1, -1), 0)
        n_bets = int(bet_o.sum() + bet_u.sum())
        rows.append({"model": LABELS[m], "MSE": np.mean((y - pred) ** 2), "MAE": np.mean(np.abs(y - pred)),
                     "O/U acc": np.mean((p > 0.5) == over), "Brier(P_over)": np.mean((p - over) ** 2),
                     "bets": n_bets, "ROI": profit.sum() / n_bets if n_bets else np.nan})
    return pd.DataFrame(rows), d


def main():
    df = pd.read_csv(RESULTS / "predictions.csv", parse_dates=["Date"])
    models = [c[5:] for c in df.columns if c.startswith("pred_") and c != "pred_book"]
    out = {}
    for split in ("val", "test"):
        res, d = evaluate(df[df["split"] == split], models)
        out[split] = res
        print(f"\n=== {split.upper()} season ({len(d)} matches with odds) ===")
        print(res.round(3).to_string(index=False))
    # bootstrap CI of accuracy difference vs bookmaker is overkill; report the binomial std error instead
    n = len(d)
    print(f"\nstd. error of an accuracy estimate on n={n}: +/-{np.sqrt(0.25 / n):.3f} (so ~1-2 pts differences are noise)")
    pd.concat(out, names=["split"]).to_csv(RESULTS / "metrics.csv")

    # --- figures ---
    res = out["test"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    order = res.sort_values("MSE")
    ax[0].barh(order["model"], order["MSE"], color=["#d55e00" if m == "Bookmaker" else "#4c72b0" for m in order["model"]])
    ax[0].set_xlabel("Test MSE (total goals)"); ax[0].set_xlim(order["MSE"].min() * 0.95, order["MSE"].max() * 1.02)
    ax[0].invert_yaxis()
    ax[1].barh(order["model"], order["O/U acc"], color=["#d55e00" if m == "Bookmaker" else "#4c72b0" for m in order["model"]])
    ax[1].axvline(0.5, color="k", ls=":"); ax[1].set_xlabel("Over/Under 2.5 accuracy (test)")
    ax[1].set_xlim(0.4, 0.7); ax[1].invert_yaxis()
    plt.tight_layout(); plt.savefig(RESULTS / "model_comparison.png", dpi=150); plt.close()

    curves = json.loads((RESULTS / "curves.json").read_text())
    fig, ax = plt.subplots(1, len(curves), figsize=(5 * len(curves), 3.4))
    for a, (k, c) in zip(np.atleast_1d(ax), curves.items()):
        c = np.array(c); a.plot(c[:, 0], label="train"); a.plot(c[:, 1], label="validation")
        a.set_title(f"{LABELS[k]} training curve"); a.set_xlabel("epoch"); a.set_ylabel("MSE"); a.legend()
    plt.tight_layout(); plt.savefig(RESULTS / "training_curves.png", dpi=150); plt.close()

    # calibration of P(over 2.5) on test: model (LSTM) vs bookmaker
    t = df[df["split"] == "test"].dropna(subset=["Avg>2.5"])
    pb = (1 / t["Avg>2.5"]) / (1 / t["Avg>2.5"] + 1 / t["Avg<2.5"])
    over = (t["TotalGoals"] > 2.5).to_numpy()
    fig, a = plt.subplots(figsize=(4.2, 4))
    for name, p in (("Bookmaker", pb.to_numpy()), ("LSTM", p_over(t["pred_lstm"].to_numpy()))):
        bins = pd.qcut(p, 5, duplicates="drop")
        g = pd.DataFrame({"p": p, "o": over, "b": bins}).groupby("b", observed=True).mean()
        a.plot(g["p"], g["o"], "o-", label=name)
    a.plot([0.2, 0.8], [0.2, 0.8], "k:"); a.set_xlabel("predicted P(over 2.5)"); a.set_ylabel("observed frequency")
    a.legend(); plt.tight_layout(); plt.savefig(RESULTS / "calibration.png", dpi=150); plt.close()


if __name__ == "__main__":
    main()
