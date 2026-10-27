"""Spatio-temporal graph neural network (GAT + GRU), the spec's "Stage 4" model, kept as a
comparison model. It sees the same inputs as the other models except the hand-made neighbour
features: instead it LEARNS how to combine neighbouring wells through graph attention.

For every well i and round k:
    x_{i,k-L+1..k} --GRU--> h_{i,k}
    z_{i,k} = Σ_j α_ij W h_{j,k}      over j ∈ neighbours(i) ∪ {i}   (attention α, GAT-style)
    dh_hat  = MLP([h_{i,k}, z_{i,k}, x_{i,k}])
Attention weights α are exported for the explainability panel ("which neighbours mattered").
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .core import FORCING, SEASON, STATIC

NODE_FEATS = ["anom_s", "obs_mask", "dh_prev_s"] + FORCING + SEASON + STATIC
WINDOW = 8


def _torch():
    import torch

    torch.set_num_threads(2)
    return torch


def build_tensors(panel: pd.DataFrame, edges: pd.DataFrame, stats=None):
    """Return dense tensors X (K, N, F), target Y (K, N), masks, plus index maps."""
    p = panel.copy()
    for m, c in zip((1, 5, 8, 11), SEASON):
        p[c] = (p.month == m).astype(float)
    p["obs_mask"] = p.depth_m.notna().astype(float)
    p["anom_s"] = p.anom.fillna(0.0)
    p["dh_prev_s"] = p.dh_prev.fillna(0.0)
    wells = sorted(p.well_id.unique())
    ks = np.arange(p.k.min(), p.k.max() + 1)
    wi = {w: i for i, w in enumerate(wells)}
    ki = {k: i for i, k in enumerate(ks)}
    if stats is None:
        tr = p[p.split == "train"]
        stats = {c: (float(tr[c].mean()), float(tr[c].std() or 1.0)) for c in NODE_FEATS if c not in SEASON + ["obs_mask"]}
    X = np.zeros((len(ks), len(wells), len(NODE_FEATS)), np.float32)
    Y = np.full((len(ks), len(wells)), np.nan, np.float32)
    S = np.empty((len(ks), len(wells)), dtype=object)
    r = p.k.map(ki).to_numpy()
    c = p.well_id.map(wi).to_numpy()
    for j, f in enumerate(NODE_FEATS):
        v = p[f].to_numpy(float)
        if f in stats:
            mu, sd = stats[f]
            v = (v - mu) / sd
        X[r, c, j] = np.nan_to_num(v)
    valid = p.depth_m.notna() & p.dh.notna() & p.rain_int.notna()
    Y[r[valid], c[valid]] = p.dh[valid].clip(-60, 60)
    S[r, c] = p.split.to_numpy()
    e = edges[edges.src.isin(wi) & edges.dst.isin(wi)]
    src = np.array([wi[s] for s in e.src])
    dst = np.array([wi[d] for d in e.dst])
    # add self loops
    n = len(wells)
    src = np.concatenate([src, np.arange(n)])
    dst = np.concatenate([dst, np.arange(n)])
    return dict(X=X, Y=Y, S=S, wells=wells, ks=ks, src=src, dst=dst, stats=stats)


class STGNN:
    name = "ST-GNN (GAT+GRU)"

    def __init__(self, hidden=32, epochs=250, lr=3e-3, seed=0):
        self.hidden, self.epochs, self.lr, self.seed = hidden, epochs, lr, seed

    def _net(self, f):
        torch = _torch()
        nn = torch.nn
        H = self.hidden

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.gru = nn.GRU(f, H, batch_first=True)
                self.W = nn.Linear(H, H, bias=False)
                self.a_src = nn.Linear(H, 1, bias=False)
                self.a_dst = nn.Linear(H, 1, bias=False)
                self.out = nn.Sequential(nn.Linear(2 * H + f, H), nn.ReLU(), nn.Linear(H, 1))

            def forward(self, xw, x_now, src, dst):
                # xw: (B*N, L, F) ; returns (B*N,) and attention
                _, h = self.gru(xw)
                h = h[-1]
                Wh = self.W(h)
                e = torch.nn.functional.leaky_relu(self.a_src(Wh)[dst] + self.a_dst(Wh)[src], 0.2).squeeze(-1)
                e = e - e.max()
                ex = torch.exp(e)
                denom = torch.zeros(h.shape[0]).index_add_(0, src, ex) + 1e-9
                alpha = ex / denom[src]
                z = torch.zeros_like(Wh).index_add_(0, src, alpha[:, None] * Wh[dst])
                return self.out(torch.cat([h, z, x_now], -1)).squeeze(-1), alpha

        return Net()

    def _windows(self, X, k_idx):
        """Stack windows of length WINDOW ending at each round in k_idx -> (len(k_idx)*N, L, F)."""
        L = WINDOW
        pad = np.zeros((L - 1,) + X.shape[1:], np.float32)
        Xp = np.concatenate([pad, X], 0)
        w = np.stack([Xp[k:k + L] for k in k_idx], 0)          # (B, L, N, F)
        return np.transpose(w, (0, 2, 1, 3)).reshape(-1, L, X.shape[2])

    def _batch_edges(self, n_rounds, n):
        src = np.concatenate([self.src + b * n for b in range(n_rounds)])
        dst = np.concatenate([self.dst + b * n for b in range(n_rounds)])
        return src, dst

    def fit(self, T):
        torch = _torch()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        X, Y, S = T["X"], T["Y"], T["S"]
        self.src, self.dst, self.stats = T["src"], T["dst"], T["stats"]
        n = X.shape[1]
        train_mask = (S == "train") & np.isfinite(Y)
        k_train = np.where(train_mask.any(1))[0]
        self.net = self._net(X.shape[2])
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        loss_fn = torch.nn.HuberLoss(delta=2.0)
        bs = 16
        for ep in range(self.epochs):
            np.random.shuffle(k_train)
            for b0 in range(0, len(k_train), bs):
                kb = k_train[b0:b0 + bs]
                xw = torch.from_numpy(self._windows(X, kb))
                xn = torch.from_numpy(X[kb].reshape(-1, X.shape[2]))
                src, dst = self._batch_edges(len(kb), n)
                pred, _ = self.net(xw, xn, torch.from_numpy(src), torch.from_numpy(dst))
                y = torch.from_numpy(np.nan_to_num(Y[kb].reshape(-1)))
                m = torch.from_numpy(train_mask[kb].reshape(-1))
                if m.sum() == 0:
                    continue
                loss = loss_fn(pred[m], y[m])
                opt.zero_grad()
                loss.backward()
                opt.step()
        return self

    def predict_grid(self, X):
        torch = _torch()
        n = X.shape[1]
        out = np.zeros(X.shape[:2], np.float32)
        att = None
        with torch.no_grad():
            for b0 in range(0, X.shape[0], 16):
                kb = np.arange(b0, min(b0 + 16, X.shape[0]))
                xw = torch.from_numpy(self._windows(X, kb))
                xn = torch.from_numpy(X[kb].reshape(-1, X.shape[2]))
                src, dst = self._batch_edges(len(kb), n)
                pred, alpha = self.net(xw, xn, torch.from_numpy(src), torch.from_numpy(dst))
                out[kb] = pred.numpy().reshape(len(kb), n)
                att = alpha.numpy()[-len(self.src):]   # attention for the last round in the batch
        self.last_attention = att
        return out

    def predict_rows(self, T, rows: pd.DataFrame):
        grid = self.predict_grid(T["X"])
        wi = {w: i for i, w in enumerate(T["wells"])}
        ki = {k: i for i, k in enumerate(T["ks"])}
        return np.array([grid[ki[k], wi[w]] for w, k in zip(rows.well_id, rows.k)])
