#!/usr/bin/env python3
"""Round 2 Step 5: parquet + logistic 310 vs 180 on the same rows.

Newest 20% sealed — never scored. Rolling 60/20 on the rest.
Train-only RobustScaler. Logistic only. No LightGBM/MLP. No PR #1.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from holygrail.columns import FEATURE_COLUMNS, SCALER_COLUMNS
from holygrail.config import EMBARGO_MINUTES, SCALER_QUANTILE_RANGE
from holygrail.features import build_features
from holygrail.io import load_mt5_m1
from holygrail.models.logistic import LogisticPolicy
from holygrail.models.metrics import evaluate
from holygrail.round2_columns import ROUND2_FEATURE_COLUMNS, ROUND2_SCALER_COLUMNS
from holygrail.round2_matrix import build_round2_matrix, count_report, persistence_acc
from holygrail.timing import last_row_of_bar

HEAD = "y_5m_dir"
CC = "y_5m_close"
PURGE_MIN = 5  # y_5m_dir horizon is one 5m bar


def fit_scaler(X: pd.DataFrame, cols: list[str]) -> RobustScaler:
    scaler = RobustScaler(quantile_range=SCALER_QUANTILE_RANGE)
    scaler.fit(X[cols].to_numpy(dtype=np.float64))
    scale = scaler.scale_.copy()
    scale[~np.isfinite(scale) | (scale == 0)] = 1.0
    scaler.scale_ = scale
    return scaler


def apply_scaler(X: pd.DataFrame, scaler: RobustScaler, cols: list[str]) -> pd.DataFrame:
    out = X.copy()
    arr = scaler.transform(out[cols].to_numpy(dtype=np.float64)).astype(np.float32, copy=False)
    for i, c in enumerate(cols):
        out[c] = arr[:, i]
    return out


def top20(y: np.ndarray, proba: np.ndarray) -> dict:
    conf = np.asarray(proba).max(axis=1)
    q = float(np.quantile(conf, 0.80))
    m = conf >= q
    n = int(m.sum())
    if n == 0:
        return {"n": 0, "coverage": 0.0, "threshold": q, "accuracy": None, "log_loss": None}
    met = evaluate(y[m], proba[m])
    return {
        "n": n,
        "coverage": float(m.mean()),
        "threshold": q,
        "accuracy": met["accuracy"],
        "log_loss": met["log_loss"],
    }


def purge_train(times: pd.DatetimeIndex, train_idx: np.ndarray, val_idx: np.ndarray) -> np.ndarray:
    val_start = times[val_idx[0]]
    cutoff = val_start - pd.Timedelta(minutes=PURGE_MIN + EMBARGO_MINUTES)
    return train_idx[times[train_idx] <= cutoff]


def fold_slices(n: int) -> list[dict]:
    """Newest 20% sealed. Two chronological 60/20 folds that never enter sealed."""
    sealed = int(n * 0.80)
    unsealed = sealed
    return [
        {
            "name": "fold_60_20",
            "train": (0, int(n * 0.60)),
            "val": (int(n * 0.60), sealed),
            "note": "60% train / 20% val of all kept; newest 20% sealed",
        },
        {
            "name": "fold_unsealed_60_20",
            "train": (0, int(unsealed * 0.60)),
            "val": (int(unsealed * 0.60), int(unsealed * 0.80)),
            "note": "rolling: 60/20 of the unsealed 80%",
        },
    ]


def persist_fold(y: np.ndarray, valid: np.ndarray, times: pd.DatetimeIndex) -> dict:
    return persistence_acc(y, valid, times)


def run_logistic(Xtr, ytr, Xva, yva, cols, scaler_cols) -> tuple[dict, np.ndarray]:
    scaler = fit_scaler(Xtr, scaler_cols)
    Xtr_s = apply_scaler(Xtr[cols], scaler, scaler_cols)
    Xva_s = apply_scaler(Xva[cols], scaler, scaler_cols)
    pol = LogisticPolicy(C=1.0, max_iter=400)
    pol.fit(Xtr_s, ytr)
    proba = pol.predict_proba(Xva_s)
    met = evaluate(yva, proba)
    t20 = top20(yva, proba)
    return {**met, "top20": t20, "n_train": int(len(Xtr)), "n_val": int(len(Xva))}, proba


def pack_row(change, fold, model, n, acc, ll, persist, t20, ties, extra=None):
    row = {
        "date": date.today().isoformat(),
        "change": change,
        "fold": fold,
        "model": model,
        "n": int(n),
        "accuracy": acc,
        "log_loss": ll,
        "persist_n": None if persist is None else persist.get("n"),
        "persist_acc": None if persist is None else persist.get("acc"),
        "top20_n": None if t20 is None else t20.get("n"),
        "top20_coverage": None if t20 is None else t20.get("coverage"),
        "top20_acc": None if t20 is None else t20.get("accuracy"),
        "ties_val": ties,
    }
    if extra:
        row.update(extra)
    return row


def append_trials(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    old_fields = ["date", "change", "fold", "model", "n", "accuracy", "log_loss", "ties_val"]
    full_fields = old_fields + [
        "persist_n",
        "persist_acc",
        "top20_n",
        "top20_coverage",
        "top20_acc",
    ]

    def _write(dest: Path, fields: list[str], *, append: bool) -> None:
        exists = dest.exists() and append
        mode = "a" if append else "w"
        with dest.open(mode, newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if not exists:
                w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in fields})
        print(f"wrote {len(rows)} trials -> {dest}", flush=True)

    _write(path, old_fields, append=True)
    _write(path.with_name("round2-results-trials.csv"), full_fields, append=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--csv",
        default="data/mt5_raw/EURUSD_M1_202101131130_202605270000_2020_2026.csv",
    )
    p.add_argument("--out", default="data/processed_round2")
    p.add_argument("--store", default="/cursor/stores/self/docs")
    args = p.parse_args()
    out = Path(args.out)
    store = Path(args.store)
    out.mkdir(parents=True, exist_ok=True)

    m1 = load_mt5_m1(args.csv)
    print(f"loaded {len(m1)} 1m", flush=True)

    print("build 310", flush=True)
    table = build_round2_matrix(m1)
    gap_counts = count_report(table, m1)
    print(
        f"gap kept={gap_counts['n_rows_after_gap_drop']} "
        f"dropped={gap_counts['n_dropped_lookback_gap']} "
        f"short={gap_counts['n_dropped_short_lookback']} "
        f"long={gap_counts['n_dropped_long_hole_tail']}",
        flush=True,
    )

    print("build 180", flush=True)
    ft = build_features(m1)
    last = last_row_of_bar(ft.timestamps, "5min")
    assert int(last.sum()) == len(table.X)
    X180_all = ft.X.loc[last].reset_index(drop=True)
    valid180 = ft.valid[last]
    X180_all = X180_all[FEATURE_COLUMNS]

    keep = (
        (~table.drop_lookback_gap)
        & table.valid_x
        & valid180
        & table.y_5m_dir_valid
    )
    idx = np.flatnonzero(keep)
    print(f"same-row keep {int(keep.sum())} / {len(table.X)} 5m", flush=True)

    X310 = table.X.iloc[idx].reset_index(drop=True)
    X180 = X180_all.iloc[idx].reset_index(drop=True)
    y = pd.DataFrame(
        {
            "open_time": table.timestamps[idx],
            "last_1m": table.last_1m[idx],
            "decision_time": table.decision_time[idx],
            HEAD: table.y_5m_dir[idx],
            CC: table.y_5m_close[idx],
            "y_5m_close_valid": table.y_5m_close_valid[idx],
            "prev5m_dir": X310["prev5m_dir"].to_numpy(),
        }
    )
    for tf, flag in table.row_incomplete.items():
        y[f"incomplete_{tf}"] = flag[idx]
    n = len(y)
    sealed_i = int(n * 0.80)
    y["split"] = np.where(np.arange(n) >= sealed_i, "sealed", "unsealed")
    fold_lab = np.full(n, "sealed", dtype=object)
    fold_lab[: int(n * 0.60)] = "train"
    fold_lab[int(n * 0.60) : sealed_i] = "val"
    y["fold_60_20"] = fold_lab

    print("write parquet", flush=True)
    X310.to_parquet(out / "X_310.parquet", index=False)
    X180.to_parquet(out / "X_180.parquet", index=False)
    y.to_parquet(out / "y.parquet", index=False)
    meta = {
        "n_1m": int(len(m1)),
        "n_5m_before_gap": int(len(table.X)),
        "gap": gap_counts,
        "n_same_rows": int(n),
        "n_sealed": int((y["split"] == "sealed").sum()),
        "n_unsealed": int((y["split"] == "unsealed").sum()),
        "n_columns_310": 310,
        "n_columns_180": 180,
        "n_scaler_310": len(ROUND2_SCALER_COLUMNS),
        "n_scaler_180": len(SCALER_COLUMNS),
        "sealed_rule": "newest 20% of same-row kept 5m, never scored",
        "label": HEAD,
        "close_to_close_is_reference": True,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    times = pd.DatetimeIndex(y["last_1m"])
    y_dir = y[HEAD].to_numpy()
    y_cc = y[CC].to_numpy()
    cc_ok = y["y_5m_close_valid"].to_numpy().astype(bool)
    prev = y["prev5m_dir"].to_numpy()

    folds = fold_slices(n)
    trials = []
    fold_reports = []
    for spec in folds:
        tr0, tr1 = spec["train"]
        va0, va1 = spec["val"]
        train_idx = np.arange(tr0, tr1)
        val_idx = np.arange(va0, va1)
        train_idx = purge_train(times, train_idx, val_idx)
        print(
            f"{spec['name']} train {len(train_idx)} val {len(val_idx)} "
            f"({spec['note']})",
            flush=True,
        )

        persist_dir = persist_fold(
            y_dir[val_idx], np.ones(len(val_idx), dtype=bool), times[val_idx]
        )
        # also prev5m_dir vs y on val
        prev_ok = np.isfinite(prev[val_idx]) & np.isfinite(y_dir[val_idx])
        persist_prev = {
            "n": int(prev_ok.sum()),
            "acc": float(np.mean(prev[val_idx][prev_ok] == y_dir[val_idx][prev_ok]))
            if prev_ok.any()
            else None,
        }
        cc_val = cc_ok[val_idx]
        persist_cc = persist_fold(y_cc[val_idx], cc_val, times[val_idx])

        print("  logistic 180 y_dir", flush=True)
        m180, _ = run_logistic(
            X180.iloc[train_idx],
            y_dir[train_idx],
            X180.iloc[val_idx],
            y_dir[val_idx],
            FEATURE_COLUMNS,
            SCALER_COLUMNS,
        )
        print("  logistic 310 y_dir", flush=True)
        m310, _ = run_logistic(
            X310.iloc[train_idx],
            y_dir[train_idx],
            X310.iloc[val_idx],
            y_dir[val_idx],
            ROUND2_FEATURE_COLUMNS,
            ROUND2_SCALER_COLUMNS,
        )

        tr_cc = train_idx[cc_ok[train_idx]]
        va_cc = val_idx[cc_ok[val_idx]]
        print("  logistic 180 close-to-close reference", flush=True)
        m180_cc, _ = run_logistic(
            X180.iloc[tr_cc],
            y_cc[tr_cc],
            X180.iloc[va_cc],
            y_cc[va_cc],
            FEATURE_COLUMNS,
            SCALER_COLUMNS,
        )
        print("  logistic 310 close-to-close reference", flush=True)
        m310_cc, _ = run_logistic(
            X310.iloc[tr_cc],
            y_cc[tr_cc],
            X310.iloc[va_cc],
            y_cc[va_cc],
            ROUND2_FEATURE_COLUMNS,
            ROUND2_SCALER_COLUMNS,
        )

        n_ties_cc = int((~cc_ok[val_idx]).sum())
        fold_reports.append(
            {
                "fold": spec["name"],
                "note": spec["note"],
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "n_val_cc": int(len(va_cc)),
                "n_cc_ties_dropped_val": n_ties_cc,
                "persist_y_dir_consecutive": persist_dir,
                "persist_prev5m_dir": persist_prev,
                "persist_close_to_close": persist_cc,
                "logistic_180_y_dir": {
                    k: m180[k] for k in ("n_train", "n_val", "accuracy", "log_loss", "top20")
                },
                "logistic_310_y_dir": {
                    k: m310[k] for k in ("n_train", "n_val", "accuracy", "log_loss", "top20")
                },
                "logistic_180_close_to_close_reference": {
                    k: m180_cc[k] for k in ("n_train", "n_val", "accuracy", "log_loss", "top20")
                },
                "logistic_310_close_to_close_reference": {
                    k: m310_cc[k] for k in ("n_train", "n_val", "accuracy", "log_loss", "top20")
                },
            }
        )
        for model, met, persist, ties in (
            ("logistic_180", m180, persist_prev, 0),
            ("logistic_310", m310, persist_prev, 0),
            ("persistence_y_dir", {"accuracy": persist_prev["acc"], "log_loss": None, "top20": None, "n_val": persist_prev["n"]}, persist_prev, 0),
            ("logistic_180_close_to_close_ref", m180_cc, persist_cc, n_ties_cc),
            ("logistic_310_close_to_close_ref", m310_cc, persist_cc, n_ties_cc),
            ("persistence_close_to_close_ref", {"accuracy": persist_cc["acc"], "log_loss": None, "top20": None, "n_val": persist_cc["n"]}, persist_cc, n_ties_cc),
        ):
            trials.append(
                pack_row(
                    f"round2-step5 {model}",
                    spec["name"],
                    model,
                    met["n_val"],
                    met["accuracy"],
                    met["log_loss"],
                    persist,
                    met.get("top20"),
                    ties,
                )
            )
            print(
                f"  {model} acc={met['accuracy']:.4f} "
                f"ll={met['log_loss'] if met['log_loss'] is None else f'{met['log_loss']:.4f}'} "
                f"n={met['n_val']}",
                flush=True,
            )

    report = {
        "feature_version": "2.1-step5",
        "did_not_score_sealed": True,
        "did_not_train_lgbm_mlp": True,
        "did_not_touch_pr1": True,
        "gap": gap_counts,
        "n_same_rows": int(n),
        "n_sealed_unscored": int(n - sealed_i),
        "open_first": str(y["open_time"].iloc[0]),
        "open_last_unsealed": str(y["open_time"].iloc[sealed_i - 1]),
        "open_first_sealed_unscored": str(y["open_time"].iloc[sealed_i]),
        "open_last": str(y["open_time"].iloc[-1]),
        "folds": fold_reports,
        "parquet": str(out),
    }
    store.mkdir(parents=True, exist_ok=True)
    (store / "round2-results.json").write_text(json.dumps(report, indent=2, default=str))
    append_trials(store / "round2-label-study-trials.csv", trials)
    print(json.dumps({k: report[k] for k in ("n_same_rows", "n_sealed_unscored", "gap") if k in report}, indent=2, default=str), flush=True)
    print("STOP: sealed unused. no lgbm/mlp. no PR#1.", flush=True)


if __name__ == "__main__":
    main()
