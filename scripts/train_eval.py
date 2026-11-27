#!/usr/bin/env python3
"""Train majority / logistic / LightGBM / MLP. Report logloss, ECE, F1, regimes, explain."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from holygrail.columns import BLOCKS, FEATURE_COLUMNS
from holygrail.models.explain import ablation_drop_block, block_aggregate, permutation_importance, shap_tree
from holygrail.models.lgbm import LightGBMPolicy
from holygrail.models.logistic import LogisticPolicy
from holygrail.models.majority import MajorityPolicy
from holygrail.models.metrics import evaluate
from holygrail.models.mlp import MLPPolicy
from holygrail.policy import DualHead


def load_part(root: Path, name: str):
    X = pd.read_parquet(root / f"X_{name}.parquet")
    y = pd.read_parquet(root / f"y_{name}.parquet")
    return X[FEATURE_COLUMNS], y


def thin(X, y, step: int):
    sl = slice(None, None, step)
    return X.iloc[sl], y.iloc[sl]


def overlap_weights(index: pd.Series, horizon_min: int) -> np.ndarray:
    """Down-weight overlapping 1m rows: each window of `horizon` shares mass 1."""
    # simple: weight = 1/horizon so effective N ≈ n / horizon at 1m stride
    return np.full(len(index), 1.0 / horizon_min, dtype=np.float64)


def regime_masks(yframe: pd.DataFrame, X: pd.DataFrame) -> dict[str, np.ndarray]:
    atr = yframe["atr_5m"].to_numpy()
    med = np.nanmedian(atr)
    high = atr >= med
    low = ~high
    # trending: |s714_label| == 1 (all TFs same SMA7 vs 14 lean)
    trend = np.abs(X["s714_label"].to_numpy()) == 1
    chop = ~trend
    return {"high_atr": high, "low_atr": low, "trending": trend, "not_trending": chop}


def eval_head(name, y_true, proba) -> dict:
    return {"model": name, **evaluate(y_true, proba)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed")
    p.add_argument("--out", default="artifacts/train_results.json")
    p.add_argument("--thin-step", type=int, default=5)
    p.add_argument("--shap-rows", type=int, default=1500)
    args = p.parse_args()
    root = Path(args.data)
    Xtr, ytr = load_part(root, "train")
    Xva, yva = load_part(root, "val")
    Xte, yte = load_part(root, "test")

    Xtr_t, ytr_t = thin(Xtr, ytr, args.thin_step)
    Xva_t, yva_t = thin(Xva, yva, args.thin_step)
    Xte_t, yte_t = thin(Xte, yte, args.thin_step)

    n_full = len(Xtr)
    n_thin = len(Xtr_t)
    report = {
        "overlap": {
            "n_train_1m": n_full,
            "n_train_thinned_every_5": n_thin,
            "effective_n_15m_thinned": n_thin * (args.thin_step / 15.0),
            "effective_n_1h_thinned": n_thin * (args.thin_step / 60.0),
            "note": "consecutive 1m rows share 14/15 of the 15m window. Thinning every 5m still overlaps. Effective N ≈ n * step / horizon.",
        },
        "class_counts": {
            "train_15m": ytr_t["y_15m"].value_counts().sort_index().to_dict(),
            "train_1h": ytr_t["y_1h"].value_counts().sort_index().to_dict(),
            "val_15m": yva_t["y_15m"].value_counts().sort_index().to_dict(),
            "val_1h": yva_t["y_1h"].value_counts().sort_index().to_dict(),
            "test_15m": yte_t["y_15m"].value_counts().sort_index().to_dict(),
            "test_1h": yte_t["y_1h"].value_counts().sort_index().to_dict(),
        },
        "models": {},
        "regimes": {},
        "explain": {},
    }

    makers = {
        "majority": MajorityPolicy,
        "logistic": LogisticPolicy,
        "lightgbm": LightGBMPolicy,
        "mlp": MLPPolicy,
    }

    fitted = {}
    for key, maker in makers.items():
        print("fit", key, flush=True)
        dual = DualHead(maker)
        dual.fit(
            Xtr_t,
            ytr_t["y_15m"].to_numpy(),
            ytr_t["y_1h"].to_numpy(),
            X_val=Xva_t,
            y_15m_val=yva_t["y_15m"].to_numpy(),
            y_1h_val=yva_t["y_1h"].to_numpy(),
        )
        fitted[key] = dual
        proba = dual.predict_proba(Xte_t)
        report["models"][key] = {
            "test_15m": eval_head(key, yte_t["y_15m"].to_numpy(), proba["y_15m"]),
            "test_1h": eval_head(key, yte_t["y_1h"].to_numpy(), proba["y_1h"]),
            "val_15m": eval_head(key, yva_t["y_15m"].to_numpy(), dual.predict_proba(Xva_t)["y_15m"]),
            "val_1h": eval_head(key, yva_t["y_1h"].to_numpy(), dual.predict_proba(Xva_t)["y_1h"]),
        }

    # Weighted LGBM on unthinned train vs thinned (15m head only, for the overlap comparison)
    print("fit lightgbm weighted (unthinned, 15m)", flush=True)
    w = overlap_weights(ytr["decision_time"], 15)
    lgb_w = LightGBMPolicy()
    lgb_w.fit(
        Xtr,
        ytr["y_15m"].to_numpy(),
        sample_weight=w,
        X_val=Xva_t,
        y_val=yva_t["y_15m"].to_numpy(),
    )
    report["models"]["lightgbm_weighted_15m"] = {
        "test_15m": eval_head(
            "lightgbm_weighted",
            yte_t["y_15m"].to_numpy(),
            lgb_w.predict_proba(Xte_t),
        )
    }

    mlp_ll = report["models"]["mlp"]["test_15m"]["log_loss"]
    log_ll = report["models"]["logistic"]["test_15m"]["log_loss"]
    report["mlp_beats_logistic_15m"] = bool(mlp_ll < log_ll)
    report["mlp_beats_logistic_1h"] = bool(
        report["models"]["mlp"]["test_1h"]["log_loss"] < report["models"]["logistic"]["test_1h"]["log_loss"]
    )

    # Regime slices on test, using LightGBM (not tuned on test)
    lgb = fitted["lightgbm"]
    masks = regime_masks(yte_t, Xte_t)
    proba = lgb.predict_proba(Xte_t)
    for horizon, ycol, pkey in (("15m", "y_15m", "y_15m"), ("1h", "y_1h", "y_1h")):
        report["regimes"][horizon] = {}
        for rname, mask in masks.items():
            if mask.sum() < 50:
                continue
            report["regimes"][horizon][rname] = evaluate(
                yte_t[ycol].to_numpy()[mask], proba[pkey][mask]
            )

    # Explain: SHAP on LGBM val, permutation on MLP val, block ablation on LGBM val
    print("explain", flush=True)
    shap15 = shap_tree(lgb.head_15m, Xva_t, max_rows=args.shap_rows)
    col_shap = dict(zip(shap15["columns"], shap15["mean_abs_shap"]))
    report["explain"]["lgbm_shap_15m_columns"] = col_shap
    report["explain"]["lgbm_shap_15m_blocks"] = block_aggregate(col_shap)

    perm = permutation_importance(
        fitted["mlp"].head_15m, Xva_t.iloc[: min(2000, len(Xva_t))], yva_t["y_15m"].to_numpy()[: min(2000, len(Xva_t))], n_repeats=1
    )
    col_perm = dict(zip(perm["columns"], perm["log_loss_increase"]))
    report["explain"]["mlp_perm_15m_columns"] = col_perm
    report["explain"]["mlp_perm_15m_blocks"] = block_aggregate(col_perm)

    report["explain"]["lgbm_ablation_val_15m"] = ablation_drop_block(
        LightGBMPolicy,
        Xtr_t,
        ytr_t["y_15m"].to_numpy(),
        Xva_t,
        yva_t["y_15m"].to_numpy(),
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("wrote", args.out)
    print("mlp_beats_logistic_15m", report["mlp_beats_logistic_15m"])
    print("mlp_beats_logistic_1h", report["mlp_beats_logistic_1h"])


if __name__ == "__main__":
    main()
