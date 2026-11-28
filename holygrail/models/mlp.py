"""Shallow regularized MLP. Not Quantra's Keras net.

Quantra's course net is a 5-block Dense+sigmoid tower trained with accuracy
on daily BAC. We use a small 180→64→32→2 net, ReLU, dropout, weight decay,
cross-entropy, early stopping on time-ordered val logloss.

If this loses to logistic on the frozen test set, report that. Do not add
layers until it "beats" the linear model — that was how earlier NN bots
overfit.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from holygrail.config import N_CLASSES
from holygrail.policy import encode_y


class _Net(nn.Module):
    def __init__(self, d_in: int = 180, hidden: tuple[int, ...] = (64, 32), dropout: float = 0.2):
        super().__init__()
        layers: list[nn.Module] = []
        prev = d_in
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, N_CLASSES))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLPPolicy:
    name = "mlp"

    def __init__(
        self,
        hidden: tuple[int, ...] = (64, 32),
        dropout: float = 0.2,
        weight_decay: float = 1e-4,
        lr: float = 1e-3,
        batch_size: int = 4096,
        max_epochs: int = 40,
        patience: int = 6,
        seed: int = 7,
    ):
        self.hidden = hidden
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed
        self.net: _Net | None = None
        self.device = torch.device("cpu")
        self.best_state = None

    def fit(self, X, y, sample_weight=None, X_val=None, y_val=None):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        Xn = np.asarray(X, dtype=np.float32)
        yn = encode_y(np.asarray(y)).astype(np.int64)
        d_in = Xn.shape[1]
        self.net = _Net(d_in, self.hidden, self.dropout).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        weighted = sample_weight is not None
        if weighted:
            sw = np.asarray(sample_weight, dtype=np.float32)
            if sw.shape[0] != len(yn):
                raise ValueError("sample_weight length must match y")
            loss_fn = nn.CrossEntropyLoss(reduction="none")
            ds = TensorDataset(
                torch.from_numpy(np.array(Xn, copy=True)),
                torch.from_numpy(np.array(yn, copy=True)),
                torch.from_numpy(np.array(sw, copy=True)),
            )
        else:
            loss_fn = nn.CrossEntropyLoss()
            ds = TensorDataset(torch.from_numpy(np.array(Xn, copy=True)), torch.from_numpy(np.array(yn, copy=True)))
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=False)
        best = float("inf")
        wait = 0
        self.best_state = None
        for _epoch in range(self.max_epochs):
            self.net.train()
            for batch in loader:
                if weighted:
                    xb, yb, wb = batch
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    wb = wb.to(self.device)
                    opt.zero_grad()
                    logits = self.net(xb)
                    loss = (loss_fn(logits, yb) * wb).mean()
                else:
                    xb, yb = batch
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    opt.zero_grad()
                    logits = self.net(xb)
                    loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
            if X_val is not None and y_val is not None:
                val_loss = self._logloss(X_val, y_val)
                if val_loss + 1e-6 < best:
                    best = val_loss
                    wait = 0
                    self.best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
                else:
                    wait += 1
                    if wait >= self.patience:
                        break
        if self.best_state is not None:
            self.net.load_state_dict(self.best_state)
        self.net.eval()
        return self

    def _logloss(self, X, y) -> float:
        proba = self.predict_proba(X)
        yn = encode_y(np.asarray(y))
        p = np.clip(proba[np.arange(len(yn)), yn], 1e-7, 1.0)
        return float(-np.mean(np.log(p)))

    def predict_proba(self, X) -> np.ndarray:
        assert self.net is not None
        Xn = np.asarray(X, dtype=np.float32)
        self.net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(Xn), self.batch_size):
                xb = torch.from_numpy(Xn[i : i + self.batch_size]).to(self.device)
                logits = self.net(xb)
                out.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(out, axis=0)

    def explain(self, X) -> dict:
        return {
            "kind": "mlp_permutation_external",
            "note": "permutation importance is computed in holygrail.models.explain",
        }
