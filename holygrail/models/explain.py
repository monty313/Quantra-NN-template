"""SHAP (trees), permutation (MLP), per-column and per-block importance, val ablation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from holygrail.columns import BLOCKS, FEATURE_COLUMNS
from holygrail.models.metrics import evaluate


def permutation_importance(policy, X: pd.DataFrame, y: np.ndarray, n_repeats: int = 3, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    Xn = X.to_numpy(dtype=np.float32) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=np.float32)
    cols = list(X.columns) if isinstance(X, pd.DataFrame) else FEATURE_COLUMNS
    base = evaluate(y, policy.predict_proba(Xn))["log_loss"]
    drops = []
    for j in range(Xn.shape[1]):
        scores = []
        for _ in range(n_repeats):
            Xp = Xn.copy()
            rng.shuffle(Xp[:, j])
            scores.append(evaluate(y, policy.predict_proba(Xp))["log_loss"])
        drops.append(float(np.mean(scores) - base))
    return {"base_log_loss": base, "columns": cols, "log_loss_increase": drops}


def shap_tree(policy, X: pd.DataFrame, max_rows: int = 2000, seed: int = 0) -> dict:
    import shap

    Xn = X.to_numpy(dtype=np.float32) if isinstance(X, pd.DataFrame) else np.asarray(X)
    cols = list(X.columns) if isinstance(X, pd.DataFrame) else FEATURE_COLUMNS
    if len(Xn) > max_rows:
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(Xn), size=max_rows, replace=False)
        Xn = Xn[pick]
    explainer = shap.TreeExplainer(policy.model)
    sv = explainer.shap_values(Xn)
    # shap_values may be list (per class) or array (n, f, c)
    if isinstance(sv, list):
        mag = np.mean([np.abs(s).mean(axis=0) for s in sv], axis=0)
    else:
        arr = np.asarray(sv)
        if arr.ndim == 3:
            mag = np.abs(arr).mean(axis=(0, 2))
        else:
            mag = np.abs(arr).mean(axis=0)
    return {"columns": cols, "mean_abs_shap": mag.astype(float).tolist()}


def block_aggregate(column_scores: dict[str, float]) -> dict[str, float]:
    out = {}
    for block, cols in BLOCKS.items():
        out[block] = float(sum(column_scores.get(c, 0.0) for c in cols))
    return out


def ablation_drop_block(make_policy, X_train, y_train, X_val, y_val, X_val_full=None, columns=None) -> dict:
    """Retrain without each block on train; score log-loss on val. No test peek."""
    columns = list(columns) if columns is not None else list(FEATURE_COLUMNS)
    present = set(X_train.columns)
    base_policy = make_policy()
    base_policy.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    base = evaluate(y_val, base_policy.predict_proba(X_val))["log_loss"]
    results = {"full": base}
    delta_keys = []
    for block, cols in BLOCKS.items():
        drop = [c for c in cols if c in present]
        if not drop:
            continue
        keep = [c for c in columns if c in present and c not in drop]
        if len(keep) == len([c for c in columns if c in present]):
            continue
        p = make_policy()
        p.fit(X_train[keep], y_train, X_val=X_val[keep], y_val=y_val)
        results[block] = evaluate(y_val, p.predict_proba(X_val[keep]))["log_loss"]
        delta_keys.append(block)
    results["delta_vs_full"] = {k: results[k] - base for k in delta_keys}
    return results
