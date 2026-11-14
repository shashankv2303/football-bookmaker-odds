"""Collaborative-filtering-style model (paper section 4.1).

Matrix A: rows = home team, columns = away team, entries = total goals in past meetings.
Most cells are empty, so we fill the matrix with a low-rank SVD by iterative imputation
(hard-impute) and predict  mu + u_i S v_j^T.  To avoid leakage the matrix for a match is built
only from the WINDOW matches played before it.
"""
import numpy as np

from .common import TARGET, pick_best


def hard_impute(A, mask, rank, iters=30):
    mu = A[mask].mean()
    Z = np.where(mask, A - mu, 0.0)
    for _ in range(iters):
        U, S, Vt = np.linalg.svd(Z, full_matrices=False)
        L = (U[:, :rank] * S[:rank]) @ Vt[:rank]
        Z = np.where(mask, A - mu, L)
    return mu + L


def run(table, X_seq):
    teams = sorted(set(table["HomeTeam"]) | set(table["AwayTeam"]))
    ti = {t: i for i, t in enumerate(teams)}
    h = table["HomeTeam"].map(ti).to_numpy()
    a = table["AwayTeam"].map(ti).to_numpy()
    y = table[TARGET].to_numpy()
    n, T = len(table), len(teams)

    def fp(p):
        pred = np.full(n, y.mean())
        for i in range(n):
            lo = max(0, i - p["window"])
            if i - lo < 50:
                continue
            A, cnt = np.zeros((T, T)), np.zeros((T, T))
            np.add.at(A, (h[lo:i], a[lo:i]), y[lo:i])
            np.add.at(cnt, (h[lo:i], a[lo:i]), 1)
            mask = cnt > 0
            A[mask] /= cnt[mask]
            pred[i] = hard_impute(A, mask, p["rank"])[h[i], a[i]]
        return np.clip(pred, 0.3, 6.0)

    grid = [dict(window=w, rank=r) for w in (380, 760) for r in (1, 2, 3)]
    params, pred = pick_best(grid, table, fp)
    return pred, {"params": params}
