#!/usr/bin/env python3
"""Train 2-class majority / logistic / LightGBM / MLP on y_5m, y_30m, y_1h_structure."""

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
from holygrail.policy import MultiHead

HEADS = ("y_5m", "y_30m", "y_1h_structure")


def load_part(root: Path, name: str):
    X = pd.read_parquet(root / f"X_{name}.parquet")
    y = pd.read_parquet(root / f"y_{name}.parquet")
    return X[FEATURE_COLUMNS], y


def last5(X, y):
    m = y["last_1m_of_5m"].to_numpy().astype(bool)
    return X.loc[m], y.loc[m]


def eval_head(name, y_true, proba) -> dict:
    return {"model": name, **evaluate(y_true, proba)}


def ys_from(frame: pd.DataFrame) -> dict:
    return {k: frame[k].to_numpy() for k in HEADS}


def fit_suite(Xtr, ytr, Xva, yva, Xte, yte):
    makers = {
        "majority": MajorityPolicy,
        "logistic": LogisticPolicy,
        "lightgbm": LightGBMPolicy,
        "mlp": MLPPolicy,
    }
    fitted = {}
    models = {}
    ytr_d, yva_d, yte_d = ys_from(ytr), ys_from(yva), ys_from(yte)
    for key, maker in makers.items():
        print("fit", key, HEADS, flush=True)
        mh = MultiHead(maker, keys=HEADS)
        mh.fit(Xtr, ytr_d, X_val=Xva, ys_val=yva_d)
        fitted[key] = mh
        proba = mh.predict_proba(Xte)
        pva = mh.predict_proba(Xva)
        models[key] = {}
        for h in HEADS:
            models[key][f"test_{h}"] = eval_head(key, yte_d[h], proba[h])
            models[key][f"val_{h}"] = eval_head(key, yva_d[h], pva[h])
    return fitted, models


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed")
    p.add_argument("--out", default="artifacts/train_results.json")
    p.add_argument("--shap-rows", type=int, default=1500)
    args = p.parse_args()
    root = Path(args.data)
    Xtr, ytr = load_part(root, "train")
    Xva, yva = load_part(root, "val")
    Xte, yte = load_part(root, "test")

    Xtr_t, ytr_t = last5(Xtr, ytr)
    Xva_t, yva_t = last5(Xva, yva)
    Xte_t, yte_t = last5(Xte, yte)

    old_tick = {}
    prev_copy = Path("artifacts/train_results_old_15m_1h.json")
    if prev_copy.exists():
        old_tick = json.loads(prev_copy.read_text(encoding="utf-8"))

    report = {
        "primary_heads": list(HEADS),
        "n_classes": 2,
        "label_kind": "sign_tp_next_minus_this",
        "overlap": {
            "n_train_1m": len(Xtr),
            "n_train_last_1m_of_5m": len(Xtr_t),
            "n_val_last_1m_of_5m": len(Xva_t),
            "n_test_last_1m_of_5m": len(Xte_t),
            "note": "Direction labels (sign TP next−this) shared per F-bar. Train on last 1m of each 5m.",
        },
        "class_counts": {
            f"{split}_{head}": frame[head].value_counts().sort_index().to_dict()
            for split, frame in (("train", ytr_t), ("val", yva_t), ("test", yte_t))
            for head in HEADS
        },
        "old_labels_15m_1h": {
            "note": "previous Stage-1 tick heads, not retrained.",
            "models": old_tick.get("models", {}),
        },
        "models": {},
        "regimes": {},
        "explain": {},
    }

    fitted, models = fit_suite(Xtr_t, ytr_t, Xva_t, yva_t, Xte_t, yte_t)
    report["models"] = models
    for h in HEADS:
        report[f"mlp_beats_logistic_{h}"] = bool(
            models["mlp"][f"test_{h}"]["log_loss"] < models["logistic"][f"test_{h}"]["log_loss"]
        )
        maj = models["majority"][f"test_{h}"]["log_loss"]
        best = min(models[k][f"test_{h}"]["log_loss"] for k in ("logistic", "lightgbm", "mlp"))
        report[f"best_minus_majority_{h}"] = float(maj - best)

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
    for h in HEADS:
        report["regimes"][h] = {}
        for rname, mask in masks.items():
            if mask.sum() < 50:
                continue
            report["regimes"][h][rname] = evaluate(yte_t[h].to_numpy()[mask], proba[h][mask])

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
    for h in HEADS:
        print(h, "majority", models["majority"][f"test_{h}"]["log_loss"],
              "best_gap", report[f"best_minus_majority_{h}"])


if __name__ == "__main__":
    main()
