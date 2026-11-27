"""Test metrics: log loss, ECE, per-class P/R, macro F1. No PnL."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)

from holygrail.policy import CLASS_ORDER, encode_y


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    """Multiclass ECE: confidence vs accuracy, equal-width bins on max-prob."""
    yn = encode_y(np.asarray(y_true))
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(yn)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            m = (conf >= lo) & (conf <= hi)
        else:
            m = (conf >= lo) & (conf < hi)
        if not np.any(m):
            continue
        acc = np.mean(pred[m] == yn[m])
        avg_conf = np.mean(conf[m])
        ece += (m.sum() / n) * abs(acc - avg_conf)
    return float(ece)


def evaluate(y_true: np.ndarray, proba: np.ndarray) -> dict:
    yn = encode_y(np.asarray(y_true))
    labels = [0, 1, 2]
    ll = float(log_loss(yn, proba, labels=labels))
    pred = proba.argmax(axis=1)
    p, r, f, s = precision_recall_fscore_support(
        yn, pred, labels=labels, zero_division=0
    )
    macro_f1 = float(f1_score(yn, pred, labels=labels, average="macro", zero_division=0))
    per_class = {}
    for i, name in enumerate(["-1", "0", "+1"]):
        per_class[name] = {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f[i]),
            "support": int(s[i]),
        }
    return {
        "log_loss": ll,
        "ece": expected_calibration_error(y_true, proba),
        "macro_f1": macro_f1,
        "per_class": per_class,
        "class_order": CLASS_ORDER.tolist(),
    }
