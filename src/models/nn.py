"""Fully-connected neural net (paper section 4.2) and LSTM (paper section 4.3), both in PyTorch."""
import numpy as np
import torch
import torch.nn as nn

from features import feature_columns
from .common import TARGET, mse, pick_best, split_idx

torch.set_num_threads(2)


class MLP(nn.Module):
    def __init__(self, d, hidden=(64, 32), drop=0.2):
        super().__init__()
        layers, prev = [], d
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(drop)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class TeamLSTM(nn.Module):
    """One LSTM reads a team's last N matches; home and away teams use separate LSTMs;
    the two final hidden states go through a dense layer to predict total goals."""
    def __init__(self, d_in, hidden=16, drop=0.2):
        super().__init__()
        self.home, self.away = nn.LSTM(d_in, hidden, batch_first=True), nn.LSTM(d_in, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(2 * hidden, 1))

    def forward(self, x):                       # x: (B, 2, N, d)
        _, (hh, _) = self.home(x[:, 0])
        _, (ha, _) = self.away(x[:, 1])
        return self.head(torch.cat([hh[-1], ha[-1]], dim=1)).squeeze(-1)


def train_net(make_model, X, y, idx, lr, wd, epochs=200, patience=25, seed=0, bs=64):
    """MSE training with Adam; keeps the weights of the best validation epoch (early stopping).
    Returns predictions for all rows and the (train, val) loss curves."""
    torch.manual_seed(seed)
    model = make_model()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    Xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    tr, va = torch.tensor(idx["train"]), torch.tensor(idx["val"])
    best, best_state, bad, curve = 1e9, None, 0, []
    for ep in range(epochs):
        model.train()
        perm = tr[torch.randperm(len(tr))]
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            nn.functional.mse_loss(model(Xt[b]), yt[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            tl = nn.functional.mse_loss(model(Xt[tr]), yt[tr]).item()
            vl = nn.functional.mse_loss(model(Xt[va]), yt[va]).item()
        curve.append((tl, vl))
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(Xt).numpy(), curve


def _run(table, X, make_model, grid, seeds=3):
    y, idx = table[TARGET].to_numpy(), split_idx(table)
    curves = {}

    def fp(p):
        preds, cs = zip(*[train_net(lambda: make_model(X.shape[-1], p), X, y, idx, p["lr"], p["wd"], seed=s)
                          for s in range(seeds)])
        curves[str(p)] = cs[0]
        return np.mean(preds, axis=0)

    params, pred = pick_best(grid, table, fp)
    return pred, {"params": params, "curve": curves[str(params)]}


def run_nn(table, X_seq):
    X = table[feature_columns(table)].to_numpy(dtype=float)
    tr = split_idx(table)["train"]
    X = (X - X[tr].mean(0)) / (X[tr].std(0) + 1e-9)             # standardise with TRAIN stats only
    grid = [dict(hidden=h, drop=d, lr=1e-3, wd=w) for h in ((64, 32), (32, 16)) for d in (0.2, 0.4) for w in (1e-4, 1e-2)]
    return _run(table, X, lambda d, p: MLP(d, p["hidden"], p["drop"]), grid)


def run_lstm(table, X_seq):
    tr = split_idx(table)["train"]
    flat = X_seq[tr].reshape(-1, X_seq.shape[-1])
    X = (X_seq - flat.mean(0)) / (flat.std(0) + 1e-9)
    grid = [dict(hidden=h, drop=d, lr=3e-3, wd=w) for h in (8, 16) for d in (0.2, 0.4) for w in (1e-4, 1e-2)]
    return _run(table, X, lambda d, p: TeamLSTM(d, p["hidden"], p["drop"]), grid)
