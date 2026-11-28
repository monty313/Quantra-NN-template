"""RobustScaler on the 67 gap columns, training partition only (spec §13).

Why not StandardScaler on all 180 columns (svm-ftmo): binaries and levels are
already -1/0/+1. Scaling them mixes sign with magnitude and can flip a
"below-median" scaled gap while the raw binary is still +1. T8 checks that
binaries stay raw. prev5m_dir is raw ±1/0, not a scaler column.

Fit only on train. Held-out extremes must not move center_/scale_ (T5).
Phase B added 21 gap columns (67 total); binaries, %b, labels, and prev5m_dir stay raw.
IQR 0 → scale 1 so a constant column does not blow up.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from holygrail.columns import FEATURE_COLUMNS, FEATURE_VERSION, SCALER_COLUMNS
from holygrail.config import SCALER_QUANTILE_RANGE


def fit_gap_scaler(X_train: pd.DataFrame) -> RobustScaler:
    if list(X_train.columns) != FEATURE_COLUMNS:
        raise ValueError("X_train columns must match FEATURE_COLUMNS order")
    scaler = RobustScaler(quantile_range=SCALER_QUANTILE_RANGE)
    scaler.fit(X_train[SCALER_COLUMNS].to_numpy(dtype=np.float64))
    # IQR 0 → scale 1
    scale = scaler.scale_.copy()
    scale[~np.isfinite(scale) | (scale == 0)] = 1.0
    scaler.scale_ = scale
    return scaler


def transform_gaps(X: pd.DataFrame, scaler: RobustScaler) -> pd.DataFrame:
    out = X.copy()
    arr = out[SCALER_COLUMNS].to_numpy(dtype=np.float64)
    scaled = scaler.transform(arr).astype(np.float32, copy=False)
    for i, col in enumerate(SCALER_COLUMNS):
        out[col] = scaled[:, i]
    return out


def save_scaler(
    scaler: RobustScaler,
    path: str | Path,
    *,
    cutoff: str,
    inner_cutoff: str | None = None,
    extra: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)
    meta = {
        "feature_version": FEATURE_VERSION,
        "scaler": "RobustScaler",
        "quantile_range": list(SCALER_QUANTILE_RANGE),
        "columns": SCALER_COLUMNS,
        "column_order": FEATURE_COLUMNS,
        "center_": scaler.center_.tolist(),
        "scale_": scaler.scale_.tolist(),
        "cutoff": str(cutoff),
        "inner_cutoff": None if inner_cutoff is None else str(inner_cutoff),
    }
    if extra:
        meta.update(extra)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_scaler(path: str | Path) -> tuple[RobustScaler, dict]:
    path = Path(path)
    scaler = joblib.load(path)
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if meta["columns"] != SCALER_COLUMNS:
        raise ValueError("saved scaler columns do not match SCALER_COLUMNS")
    return scaler, meta
