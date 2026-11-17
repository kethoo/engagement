"""Regression heads on frozen features. All predict both labels at once ([n, 2]).

- ``RidgeHead``      closed-form linear probe; the reference a trained head must beat
- ``TorchHead``      the trainable head (linear or 1-hidden-layer MLP) in PyTorch, MSE,
                     full-batch AdamW. This is the module the steering stage will
                     backpropagate through, so it is kept as a plain ``nn.Module``.
- ``TfidfRidgeHead`` no-LLM text baseline on the article text itself
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn


class RidgeHead:
    def __init__(self, alpha: float):
        self.m = make_pipeline(StandardScaler(), Ridge(alpha=alpha))

    def fit(self, X, Y):
        self.m.fit(X, Y)
        return self

    def predict(self, X):
        return self.m.predict(X)


class TfidfRidgeHead(RidgeHead):
    def __init__(self, alpha: float):
        self.m = make_pipeline(TfidfVectorizer(stop_words='english', sublinear_tf=True, min_df=2), Ridge(alpha=alpha))


class RegressionHead(nn.Module):
    """Standardised hidden state -> [empathy, distress]."""

    def __init__(self, d_in: int, d_out: int = 2, hidden: int = 0, dropout: float = 0.1):
        super().__init__()
        if hidden:
            self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, d_out))
        else:
            self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_in, d_out))
        self.register_buffer('x_mu', torch.zeros(d_in))
        self.register_buffer('x_sd', torch.ones(d_in))
        self.register_buffer('y_mu', torch.zeros(d_out))
        self.register_buffer('y_sd', torch.ones(d_out))

    def forward(self, h):
        return self.net((h - self.x_mu) / self.x_sd) * self.y_sd + self.y_mu


class TorchHead:
    def __init__(self, weight_decay: float, hidden: int = 0, dropout: float = 0.1, lr: float = 1e-3, epochs: int = 300, seed: int = 0):
        self.kw = dict(hidden=hidden, dropout=dropout)
        self.wd, self.lr, self.epochs, self.seed = weight_decay, lr, epochs, seed

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def fit(self, X, Y):
        torch.manual_seed(self.seed)
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        Y = torch.as_tensor(Y, dtype=torch.float32, device=self.device)
        self.net = RegressionHead(X.shape[1], Y.shape[1], **self.kw).to(self.device)
        self.net.x_mu.copy_(X.mean(0)); self.net.x_sd.copy_(X.std(0) + 1e-6)
        self.net.y_mu.copy_(Y.mean(0)); self.net.y_sd.copy_(Y.std(0) + 1e-6)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        Yz = (Y - self.net.y_mu) / self.net.y_sd
        self.net.train()
        for _ in range(self.epochs):  # full batch: ~100 articles
            opt.zero_grad()
            z = self.net.net((X - self.net.x_mu) / self.net.x_sd)
            nn.functional.mse_loss(z, Yz).backward()
            opt.step()
        self.net.eval()
        return self

    @torch.no_grad()
    def predict(self, X):
        return self.net(torch.as_tensor(X, dtype=torch.float32, device=self.device)).cpu().numpy()


def candidates(cfg: dict):
    """Hyper-parameter grid as ``(name, view, make_head)``; ``view`` = (layer, pooling) or 'text'."""
    h = cfg['head']
    t = h['type']
    if t == 'tfidf_ridge':
        return [(f'tfidf a={a:g}', 'text', lambda a=a: TfidfRidgeHead(a)) for a in h['alphas']]
    out = []
    # a torch fit costs ~100x a ridge fit, so torch heads search a narrower layer grid
    layers = h['layers'] if t == 'ridge' else h.get('torch_layers', h['layers'])
    for layer in layers:
        for pool in h['pooling']:
            if t == 'ridge':
                out += [(f'L{layer} {pool} a={a:g}', (layer, pool), lambda a=a: RidgeHead(a)) for a in h['alphas']]
            elif t in ('linear', 'mlp'):
                hid = h['hidden'] if t == 'mlp' else 0
                out += [
                    (f'L{layer} {pool} wd={wd:g}', (layer, pool),
                     lambda wd=wd, hid=hid: TorchHead(wd, hid, h['dropout'], h['lr'], h['epochs'], cfg['seed']))
                    for wd in h['weight_decay']
                ]
            else:
                raise ValueError(f'unknown head type {t!r}')
    return out
