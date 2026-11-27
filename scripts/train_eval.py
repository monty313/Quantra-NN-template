#!/usr/bin/env python3
"""Train majority / logistic / LightGBM / MLP on structure heads y_5m / y_30m."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from holygrail.columns import FEATURE_COLUMNS
from holygrail.models.explain import ablation_drop_block, block_aggregate, permutation_importance, shap_tree
from holygrail.models.lgbm import LightGBMPolicy
from holygrail.models.logistic import LogisticPolicy
from holygrail.models.majority import MajorityPolicy
from holygrail.models.metrics import evaluate
from holygrail.models.mlp import MLPPolicy
from holygrail.policy import DualHead

NEAR_MAJORITY_LOGLOSS = 0.01  # extra lever if best-majority < this


def load_part(root: Path, name: str):
    X = pd.read_parquet(root / f"X_{name}.parquet")
    y = pd.read_parquet(root / f"y_{name}.parquet")
    return X[FEATURE_COLUMNS], y


def last5(X, y):
    m = y["last_1m_of_5m"].to_numpy().astype(bool)
    return X.loc[m], y.loc[m]


def eval_head(name, y_true, proba) -> dict:
    return {"model": name, **evaluate(y_true, proba)}


def near_majority(report_models: dict, head: str) -> bool:
    maj = report_models["majority"][head]["log_loss"]
    best = min(
        report_models[k][head]["log_loss"]
        for k in ("logistic", "lightgbm", "mlp")
        if k in report_models and head in report_models[k]
    )
    return (maj - best) < NEAR_MAJORITY_LOGLOSS


def fit_suite(Xtr, ytr_ltf, ytr_htf, Xva, yva_ltf, yva_htf, Xte, yte_ltf, yte_htf, ltf_key, htf_key):
    makers = {
        "majority": MajorityPolicy,
        "logistic": LogisticPolicy,
        "lightgbm": LightGBMPolicy,
        "mlp": MLPPolicy,
    }
    fitted = {}
    models = {}
    for key, maker in makers.items():
        print("fit", key, ltf_key, htf_key, flush=True)
        dual = DualHead(maker, ltf_key=ltf_key, htf_key=htf_key)
        dual.fit(
            Xtr,
            ytr_ltf,
            ytr_htf,
            X_val=Xva,
            y_ltf_val=yva_ltf,
            y_htf_val=yva_htf,
        )
        fitted[key] = dual
        proba = dual.predict_proba(Xte)
        pva = dual.predict_proba(Xva)
        models[key] = {
            f"test_{ltf_key}": eval_head(key, yte_ltf, proba[ltf_key]),
            f"test_{htf_key}": eval_head(key, yte_htf, proba[htf_key]),
            f"val_{ltf_key}": eval_head(key, yva_ltf, pva[ltf_key]),
            f"val_{htf_key}": eval_head(key, yva_htf, pva[htf_key]),
        }
    return fitted, models


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed")
    p.add_argument("--out", default="artifacts/train_results.json")
    p.add_argument("--old-results", default="artifacts/train_results.json")
    p.add_argument("--shap-rows", type=int, default=1500)
    args = p.parse_args()
    root = Path(args.data)
    Xtr, ytr = load_part(root, "train")
    Xva, yva = load_part(root, "val")
    Xte, yte = load_part(root, "test")

    # One snapshot per 5m bar (last 1m). Shared labels; live_1m X still used.
    Xtr_t, ytr_t = last5(Xtr, ytr)
    Xva_t, yva_t = last5(Xva, yva)
    Xte_t, yte_t = last5(Xte, yte)

    old = {}
    old_path = Path(args.old_results)
    # Read previous tick-head metrics if this file still has them (first run of this script overwrites).
    prev_copy = Path("artifacts/train_results_old_15m_1h.json")
    if prev_copy.exists():
        old = json.loads(prev_copy.read_text(encoding="utf-8"))
    elif old_path.exists():
        prev = json.loads(old_path.read_text(encoding="utf-8"))
        if prev.get("models", {}).get("majority", {}).get("test_15m"):
            old = prev
            prev_copy.write_text(json.dumps(prev, indent=2, default=str), encoding="utf-8")

    report = {
        "primary_heads": ["y_5m", "y_30m"],
        "overlap": {
            "n_train_1m": len(Xtr),
            "n_train_last_1m_of_5m": len(Xtr_t),
            "n_val_last_1m_of_5m": len(Xva_t),
            "n_test_last_1m_of_5m": len(Xte_t),
            "note": "y_5m/y_30m are shared across 1m rows of the same F-bar. Train on last 1m of each 5m.",
        },
        "class_counts": {
            "train_5m": ytr_t["y_5m"].value_counts().sort_index().to_dict(),
            "train_30m": ytr_t["y_30m"].value_counts().sort_index().to_dict(),
            "val_5m": yva_t["y_5m"].value_counts().sort_index().to_dict(),
            "val_30m": yva_t["y_30m"].value_counts().sort_index().to_dict(),
            "test_5m": yte_t["y_5m"].value_counts().sort_index().to_dict(),
            "test_30m": yte_t["y_30m"].value_counts().sort_index().to_dict(),
            "train_15m_comparison": ytr_t["y_15m"].value_counts().sort_index().to_dict(),
            "train_1h_comparison": ytr_t["y_1h"].value_counts().sort_index().to_dict(),
        },
        "old_labels_15m_1h": {
            "note": "previous Stage-1 tick heads, not retrained. Copied for comparison.",
            "models": old.get("models", {}),
        },
        "models": {},
        "regimes": {},
        "explain": {},
        "extra_experiment": None,
    }

    fitted, models = fit_suite(
        Xtr_t,
        ytr_t["y_5m"].to_numpy(),
        ytr_t["y_30m"].to_numpy(),
        Xva_t,
        yva_t["y_5m"].to_numpy(),
        yva_t["y_30m"].to_numpy(),
        Xte_t,
        yte_t["y_5m"].to_numpy(),
        yte_t["y_30m"].to_numpy(),
        "y_5m",
        "y_30m",
    )
    report["models"] = models
    report["mlp_beats_logistic_5m"] = bool(
        models["mlp"]["test_y_5m"]["log_loss"] < models["logistic"]["test_y_5m"]["log_loss"]
    )
    report["mlp_beats_logistic_30m"] = bool(
        models["mlp"]["test_y_30m"]["log_loss"] < models["logistic"]["test_y_30m"]["log_loss"]
    )
    report["near_majority_5m"] = near_majority(models, "test_y_5m")
    report["near_majority_30m"] = near_majority(models, "test_y_30m")

    # Extra lever if still ≈ majority: 1.0× ATR (wider, less class-0 mush), same split/X.
    if report["near_majority_5m"] and report["near_majority_30m"]:
        print("near majority on both heads — extra lever: 1.0× same-HTF ATR", flush=True)
        wtr = ytr_t["label_valid_5m_wide"].to_numpy() & ytr_t["label_valid_30m_wide"].to_numpy()
        wva = yva_t["label_valid_5m_wide"].to_numpy() & yva_t["label_valid_30m_wide"].to_numpy()
        wte = yte_t["label_valid_5m_wide"].to_numpy() & yte_t["label_valid_30m_wide"].to_numpy()
        _, wide_models = fit_suite(
            Xtr_t.loc[wtr],
            ytr_t.loc[wtr, "y_5m_wide"].to_numpy(),
            ytr_t.loc[wtr, "y_30m_wide"].to_numpy(),
            Xva_t.loc[wva],
            yva_t.loc[wva, "y_5m_wide"].to_numpy(),
            yva_t.loc[wva, "y_30m_wide"].to_numpy(),
            Xte_t.loc[wte],
            yte_t.loc[wte, "y_5m_wide"].to_numpy(),
            yte_t.loc[wte, "y_30m_wide"].to_numpy(),
            "y_5m_wide",
            "y_30m_wide",
        )
        report["extra_experiment"] = {
            "lever": "STRUCTURE_WIDE_ATR_FRACTION=1.0 (same HTF ATR, still 3-class)",
            "class_counts_train_5m_wide": ytr_t.loc[wtr, "y_5m_wide"].value_counts().sort_index().to_dict(),
            "class_counts_train_30m_wide": ytr_t.loc[wtr, "y_30m_wide"].value_counts().sort_index().to_dict(),
            "models": wide_models,
            "near_majority_5m_wide": near_majority(wide_models, "test_y_5m_wide"),
            "near_majority_30m_wide": near_majority(wide_models, "test_y_30m_wide"),
        }
    else:
        report["extra_experiment"] = {
            "skipped": True,
            "reason": "primary 0.5× ATR heads beat majority by more than 0.01 logloss on at least one head",
        }

    lgb = fitted["lightgbm"]
    atr = yte_t["atr_5m"].to_numpy()
    med = np.nanmedian(atr)
    masks = {
        "high_atr": atr >= med,
        "low_atr": atr < med,
        "trending": np.abs(Xte_t["s714_label"].to_numpy()) == 1,
        "not_trending": np.abs(Xte_t["s714_label"].to_numpy()) != 1,
    }
    proba = lgb.predict_proba(Xte_t)
    for horizon, ycol, pkey in (("5m", "y_5m", "y_5m"), ("30m", "y_30m", "y_30m")):
        report["regimes"][horizon] = {}
        for rname, mask in masks.items():
            if mask.sum() < 50:
                continue
            report["regimes"][horizon][rname] = evaluate(
                yte_t[ycol].to_numpy()[mask], proba[pkey][mask]
            )

    print("explain", flush=True)
    shap15 = shap_tree(lgb.head_ltf, Xva_t, max_rows=args.shap_rows)
    col_shap = dict(zip(shap15["columns"], shap15["mean_abs_shap"]))
    report["explain"]["lgbm_shap_5m_columns"] = col_shap
    report["explain"]["lgbm_shap_5m_blocks"] = block_aggregate(col_shap)

    nperm = min(2000, len(Xva_t))
    perm = permutation_importance(
        fitted["mlp"].head_ltf, Xva_t.iloc[:nperm], yva_t["y_5m"].to_numpy()[:nperm], n_repeats=1
    )
    col_perm = dict(zip(perm["columns"], perm["log_loss_increase"]))
    report["explain"]["mlp_perm_5m_columns"] = col_perm
    report["explain"]["mlp_perm_5m_blocks"] = block_aggregate(col_perm)

    report["explain"]["lgbm_ablation_val_5m"] = ablation_drop_block(
        LightGBMPolicy,
        Xtr_t,
        ytr_t["y_5m"].to_numpy(),
        Xva_t,
        yva_t["y_5m"].to_numpy(),
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("wrote", args.out)
    print("near_majority_5m", report["near_majority_5m"], "near_majority_30m", report["near_majority_30m"])
    print("mlp_beats_logistic_5m", report["mlp_beats_logistic_5m"])
    print("mlp_beats_logistic_30m", report["mlp_beats_logistic_30m"])


if __name__ == "__main__":
    main()
