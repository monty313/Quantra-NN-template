#!/usr/bin/env python3
"""Phase C: val-fold only. Never score the sealed test. Do not run Phase D.

Rebuild is scripts/build_dataset.py (scaler fit on train 67 gaps).
This script loads train + val only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from holygrail.columns import BLOCKS, FEATURE_COLUMNS, FEATURE_COLUMNS_V1_0
from holygrail.models.explain import ablation_drop_block, block_aggregate, shap_tree
from holygrail.models.lgbm import LightGBMPolicy
from holygrail.models.logistic import LogisticPolicy
from holygrail.models.majority import MajorityPolicy
from holygrail.models.metrics import evaluate
from holygrail.models.mlp import MLPPolicy
from holygrail.models.persistence import PersistencePolicy

HEAD = "y_5m_dir"
LGBM_KW = dict(
    n_estimators=400,
    max_depth=4,
    num_leaves=16,
    learning_rate=0.03,
    min_child_samples=200,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=2.0,
)


def load_train_val(root: Path):
    """Train and val only. Test parquet is not opened for metrics."""
    Xtr = pd.read_parquet(root / "X_train.parquet")
    ytr = pd.read_parquet(root / "y_train.parquet")
    Xva = pd.read_parquet(root / "X_val.parquet")
    yva = pd.read_parquet(root / "y_val.parquet")
    missing = [c for c in FEATURE_COLUMNS if c not in Xtr.columns]
    if missing:
        raise RuntimeError(f"processed X is not 180-col; missing {missing[:5]}")
    Xtr = Xtr[FEATURE_COLUMNS]
    Xva = Xva[FEATURE_COLUMNS]
    if HEAD not in ytr.columns:
        ytr[HEAD] = ytr["y_5m"]
        yva[HEAD] = yva["y_5m"]
    return Xtr, ytr, Xva, yva


def last5(X, y):
    m = y["last_1m_of_5m"].to_numpy().astype(bool)
    return X.loc[m].reset_index(drop=True), y.loc[m].reset_index(drop=True)


def bar_weights(open_time: pd.Series) -> np.ndarray:
    bar = pd.to_datetime(open_time).dt.floor("5min")
    n = bar.groupby(bar).transform("size").to_numpy(dtype=np.float64)
    return 1.0 / np.maximum(n, 1.0)


def kish_n(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=np.float64)
    s = w.sum()
    q = np.square(w).sum()
    return float((s * s) / q) if q > 0 else 0.0


def pack_eval(y, proba) -> dict:
    return evaluate(np.asarray(y), proba)


def trial(tid, model, features, sampling, n_train, n_val, n_eff_train, metrics, extra=None):
    row = {
        "id": tid,
        "model": model,
        "features": features,
        "sampling": sampling,
        "n_train": int(n_train),
        "n_val": int(n_val),
        "n_eff_train": float(n_eff_train),
        "val_log_loss": metrics["log_loss"],
        "val_accuracy": metrics["accuracy"],
        "val_accuracy_minus_50": metrics["accuracy_minus_50"],
        "val_ece": metrics["ece"],
        "val_per_class": metrics["per_class"],
    }
    if extra:
        row.update(extra)
    print(
        f"{tid} {model} {features} {sampling} "
        f"ll={metrics['log_loss']:.4f} acc={metrics['accuracy']:.4f} n_eff={n_eff_train:.1f}",
        flush=True,
    )
    return row


def fit_eval(make, Xtr, ytr, Xva, yva, sample_weight=None):
    y_tr = ytr[HEAD].to_numpy()
    y_va = yva[HEAD].to_numpy()
    pol = make()
    pol.fit(Xtr, y_tr, sample_weight=sample_weight, X_val=Xva, y_val=y_va)
    proba = pol.predict_proba(Xva)
    return pol, pack_eval(y_va, proba)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed")
    p.add_argument("--out", default="artifacts/phase_c_val.json")
    p.add_argument("--shap-rows", type=int, default=2000)
    args = p.parse_args()
    root = Path(args.data)

    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    Xtr_all, ytr_all, Xva_all, yva_all = load_train_val(root)

    Xtr, ytr = last5(Xtr_all, ytr_all)
    Xva, yva = last5(Xva_all, yva_all)
    cols128 = FEATURE_COLUMNS_V1_0
    cols180 = FEATURE_COLUMNS

    w_all_tr = bar_weights(ytr_all["open_time"])
    w_all_va = bar_weights(yva_all["open_time"])
    w_thin = np.ones(len(ytr), dtype=np.float64)

    n_eff_thin = kish_n(w_thin)
    n_eff_w = kish_n(w_all_tr)

    trials = []
    models = {}

    # C1 majority (thin last-1m)
    pol, met = fit_eval(MajorityPolicy, Xtr, ytr, Xva, yva)
    trials.append(trial("C1", "majority", "none", "thin_last1m", len(Xtr), len(Xva), n_eff_thin, met))
    models["majority"] = met

    # C2 persistence
    pol, met = fit_eval(PersistencePolicy, Xtr, ytr, Xva, yva)
    extra = {"p_continue_train": float(pol.p_continue_)}
    trials.append(
        trial("C2", "persistence", "prev5m_dir", "thin_last1m", len(Xtr), len(Xva), n_eff_thin, met, extra)
    )
    models["persistence"] = met

    # C3 logistic 128 thin
    pol, met = fit_eval(LogisticPolicy, Xtr[cols128], ytr, Xva[cols128], yva)
    trials.append(trial("C3", "logistic", "128", "thin_last1m", len(Xtr), len(Xva), n_eff_thin, met))
    models["logistic_128"] = met
    logit128 = pol

    # C4 logistic 180 thin
    pol, met = fit_eval(LogisticPolicy, Xtr[cols180], ytr, Xva[cols180], yva)
    trials.append(trial("C4", "logistic", "180", "thin_last1m", len(Xtr), len(Xva), n_eff_thin, met))
    models["logistic_180"] = met
    logit180 = pol

    def lgbm():
        return LightGBMPolicy(**LGBM_KW)

    # C5 LightGBM 128 thin
    pol, met = fit_eval(lgbm, Xtr[cols128], ytr, Xva[cols128], yva)
    best_iter = getattr(getattr(pol, "model", None), "best_iteration_", None)
    trials.append(
        trial(
            "C5",
            "lightgbm",
            "128",
            "thin_last1m",
            len(Xtr),
            len(Xva),
            n_eff_thin,
            met,
            {"best_iteration": best_iter, "params": LGBM_KW},
        )
    )
    models["lgbm_128"] = met
    lgb128 = pol

    # C6 LightGBM 180 thin
    pol, met = fit_eval(lgbm, Xtr[cols180], ytr, Xva[cols180], yva)
    best_iter = getattr(getattr(pol, "model", None), "best_iteration_", None)
    trials.append(
        trial(
            "C6",
            "lightgbm",
            "180",
            "thin_last1m",
            len(Xtr),
            len(Xva),
            n_eff_thin,
            met,
            {"best_iteration": best_iter, "params": LGBM_KW},
        )
    )
    models["lgbm_180"] = met
    lgb180 = pol

    # Thin vs weights on logistic 180 and the better LGBM (few trials)
    pol, met = fit_eval(
        LogisticPolicy, Xtr_all[cols180], ytr_all, Xva[cols180], yva, sample_weight=w_all_tr
    )
    trials.append(
        trial("C7", "logistic", "180", "bar_weights_all1m", len(Xtr_all), len(Xva), n_eff_w, met)
    )
    models["logistic_180_weights"] = met

    lgb_winner_name = "180" if models["lgbm_180"]["log_loss"] <= models["lgbm_128"]["log_loss"] else "128"
    lgb_cols = cols180 if lgb_winner_name == "180" else cols128
    pol, met = fit_eval(
        lgbm, Xtr_all[lgb_cols], ytr_all, Xva[lgb_cols], yva, sample_weight=w_all_tr
    )
    best_iter = getattr(getattr(pol, "model", None), "best_iteration_", None)
    trials.append(
        trial(
            "C8",
            "lightgbm",
            lgb_winner_name,
            "bar_weights_all1m",
            len(Xtr_all),
            len(Xva),
            n_eff_w,
            met,
            {"best_iteration": best_iter, "params": LGBM_KW},
        )
    )
    models["lgbm_weights"] = met

    # Winning feature set: lowest val logloss among C3–C6 (thin; the protocol fold)
    named = {
        "logistic_128": (models["logistic_128"]["log_loss"], cols128, "128"),
        "logistic_180": (models["logistic_180"]["log_loss"], cols180, "180"),
        "lgbm_128": (models["lgbm_128"]["log_loss"], cols128, "128"),
        "lgbm_180": (models["lgbm_180"]["log_loss"], cols180, "180"),
    }
    win_key = min(named, key=lambda k: named[k][0])
    win_ll, win_cols, win_feat = named[win_key]
    print("winning_feature_set", win_key, win_feat, win_ll, flush=True)

    # C9 MLP on winning feature set, thin (same fold)
    pol, met = fit_eval(MLPPolicy, Xtr[win_cols], ytr, Xva[win_cols], yva)
    trials.append(trial("C9", "mlp", win_feat, "thin_last1m", len(Xtr), len(Xva), n_eff_thin, met))
    models["mlp_win"] = met
    mlp = pol

    # Ablation + SHAP on val, no test
    print("ablation", flush=True)
    ablate_maker = LogisticPolicy if "logistic" in win_key else lgbm
    ablation = ablation_drop_block(
        ablate_maker,
        Xtr[win_cols],
        ytr[HEAD].to_numpy(),
        Xva[win_cols],
        yva[HEAD].to_numpy(),
        columns=win_cols,
    )

    print("shap", flush=True)
    shap_src = lgb180 if lgb_winner_name == "180" else lgb128
    shap_cols = lgb_cols
    shap15 = shap_tree(shap_src, Xva[shap_cols], max_rows=args.shap_rows)
    col_shap = dict(zip(shap15["columns"], shap15["mean_abs_shap"]))
    shap_blocks = block_aggregate(col_shap)
    top_cols = sorted(col_shap.items(), key=lambda kv: -kv[1])[:20]

    coef180 = np.asarray(logit180.explain(Xva)["coef"])
    # binary: coef shape (1, n) or (n_classes, n)
    if coef180.ndim == 2:
        mag = np.abs(coef180).mean(axis=0)
    else:
        mag = np.abs(coef180)
    coef_map = {c: float(mag[i]) for i, c in enumerate(cols180)}
    coef_blocks = block_aggregate(coef_map)

    report = {
        "phase": "C",
        "note": "VAL ONLY. Sealed test was not scored. No Phase D.",
        "head": HEAD,
        "feature_version": meta.get("feature_version"),
        "cutoff": meta.get("cutoff"),
        "inner_cutoff": meta.get("inner_cutoff"),
        "n_train_1m": int(len(Xtr_all)),
        "n_val_1m": int(len(Xva_all)),
        "n_train_last1m": int(len(Xtr)),
        "n_val_last1m": int(len(Xva)),
        "n_eff_thin": n_eff_thin,
        "n_eff_bar_weights": n_eff_w,
        "n_features": {"128": 128, "180": 180, "scaler_gaps": 67},
        "lgbm_params": LGBM_KW,
        "winning_feature_set": {"trial_key": win_key, "features": win_feat, "val_log_loss": win_ll},
        "trials": trials,
        "models": models,
        "explain": {
            "ablation_drop_block_val": ablation,
            "lgbm_shap_columns_top20": top_cols,
            "lgbm_shap_blocks": shap_blocks,
            "logistic180_abs_coef_blocks": coef_blocks,
            "logistic180_abs_coef_top20": sorted(coef_map.items(), key=lambda kv: -kv[1])[:20],
        },
        "sealed_test": "not evaluated",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
