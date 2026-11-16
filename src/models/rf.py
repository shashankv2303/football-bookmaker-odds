"""Random Forest baseline (the paper's baseline)."""
from sklearn.ensemble import RandomForestRegressor

from features import feature_columns
from .common import TARGET, pick_best, split_idx


def run(table, X_seq):
    X, y, idx = table[feature_columns(table)].to_numpy(), table[TARGET].to_numpy(), split_idx(table)
    grid = [dict(n_estimators=300, max_depth=d, min_samples_leaf=l) for d in (4, 8, None) for l in (5, 20)]

    def fp(p):
        m = RandomForestRegressor(random_state=0, n_jobs=-1, **p).fit(X[idx["train"]], y[idx["train"]])
        return m.predict(X)

    params, pred = pick_best(grid, table, fp)
    return pred, {"params": params}
