#!/usr/bin/env python3
"""Build X (180), direction y_5m / y_30m / y_1h_structure, scaler.

Phase B: columns 1–128 unchanged; blocks 8–11 appended. RobustScaler on 67
gap columns, train-fit only. y_5m_dir is unchanged (alias of y_5m).
y_15m / y_1h are written as unused comparison columns.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from holygrail.columns import FEATURE_COLUMNS, FEATURE_VERSION
from holygrail.config import STRUCTURE_PURGE_MINUTES, TF_FREQ
from holygrail.features import build_features
from holygrail.io import load_mt5_m1
from holygrail.labels import build_labels
from holygrail.scaler import fit_gap_scaler, save_scaler, transform_gaps
from holygrail.splits import class_counts, time_split
from holygrail.timing import last_row_of_bar


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/mt5_raw/EURUSD_M1_202101131130_202605270000_2020_2026.csv")
    p.add_argument("--out", default="data/processed")
    p.add_argument("--max-rows", type=int, default=0, help="optional cap for smoke runs")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    m1 = load_mt5_m1(args.csv)
    if args.max_rows:
        m1 = m1.iloc[: args.max_rows]
    print(f"loaded {len(m1)} 1m bars {m1.index[0]} -> {m1.index[-1]}", flush=True)

    ft = build_features(m1)
    lab = build_labels(m1, ft)
    print(f"features valid {int(ft.valid.sum())}/{len(ft.valid)}", flush=True)

    all_struct = lab.label_valid_5m & lab.label_valid_30m & lab.label_valid_1h_structure
    split = time_split(
        ft.decision_time,
        ft.valid,
        all_struct,
        label_horizon_minutes=STRUCTURE_PURGE_MINUTES,
    )
    train_idx = split.train_idx
    val_idx = split.val_idx
    test_idx = split.test_idx

    X = ft.X
    scaler = fit_gap_scaler(X.iloc[train_idx])
    X_scaled = transform_gaps(X, scaler)
    save_scaler(
        scaler,
        Path("artifacts/scaler.joblib"),
        cutoff=str(split.cutoff),
        inner_cutoff=str(split.inner_cutoff),
        extra={"symbol": Path(args.csv).name, "n_train": int(len(train_idx))},
    )

    last5 = last_row_of_bar(ft.timestamps, TF_FREQ["5m"])

    def pack(idx):
        return pd.DataFrame(
            {
                "open_time": ft.timestamps[idx],
                "decision_time": ft.decision_time[idx],
                "y_5m": lab.y_5m[idx],
                "y_5m_dir": lab.y_5m_dir[idx],  # alias of y_5m
                "y_30m": lab.y_30m[idx],
                "y_1h_structure": lab.y_1h_structure[idx],
                "y_15m": lab.y_15m[idx],
                "y_1h": lab.y_1h[idx],
                "atr_5m": ft.atr_5m[idx],
                "last_1m_of_5m": last5[idx],
                "label_valid_1h_structure": lab.label_valid_1h_structure[idx],
            }
        )

    meta = {
        "feature_version": FEATURE_VERSION,
        "n_rows": int(len(m1)),
        "n_feature_valid": int(ft.valid.sum()),
        "cutoff": str(split.cutoff),
        "inner_cutoff": str(split.inner_cutoff),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "primary_heads": ["y_5m", "y_30m", "y_1h_structure"],
        "label_kind": "sign_tp_next_minus_this",
        "n_classes": 2,
        "comparison_heads": ["y_15m", "y_1h"],
        "purge_minutes": STRUCTURE_PURGE_MINUTES,
        "class_counts": {
            "train_5m": class_counts(lab.y_5m[train_idx]),
            "train_30m": class_counts(lab.y_30m[train_idx]),
            "train_1h_structure": class_counts(lab.y_1h_structure[train_idx]),
            "train_15m": class_counts(lab.y_15m[train_idx]),
            "train_1h": class_counts(lab.y_1h[train_idx]),
            "val_5m": class_counts(lab.y_5m[val_idx]),
            "val_30m": class_counts(lab.y_30m[val_idx]),
            "val_1h_structure": class_counts(lab.y_1h_structure[val_idx]),
            "test_5m": class_counts(lab.y_5m[test_idx]),
            "test_30m": class_counts(lab.y_30m[test_idx]),
            "test_1h_structure": class_counts(lab.y_1h_structure[test_idx]),
        },
        "columns": FEATURE_COLUMNS,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("class_counts", json.dumps(meta["class_counts"], indent=2), flush=True)

    for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        part = X_scaled.iloc[idx].copy().astype(np.float32)
        y = pack(idx)
        part.to_parquet(out / f"X_{name}.parquet")
        y.to_parquet(out / f"y_{name}.parquet")
        print(name, len(idx), flush=True)

    np.save(out / "train_idx.npy", train_idx)
    np.save(out / "val_idx.npy", val_idx)
    np.save(out / "test_idx.npy", test_idx)
    print("wrote", out)


if __name__ == "__main__":
    main()
