# HolyGrail Stage 1 — market-state model

The Quantra notebooks under `Neural-Networks-in-Trading_new/` are a starting template only (course code, accuracy metric, daily BAC). Stage 1 is a new, pluggable market-state pipeline: **features → labels → policy**. Why each choice was made: comments in `holygrail/` plus `docs/stage1-plan.md`, `docs/stage1-why-notes.md`, `docs/stage1-results.md`.

Outputs are calibrated 2-class probabilities (`-1 / +1`) for the **next closed 5m, 30m, and 1h direction**. Not ATR size fences, not 15-minute ticks, not trade signals. No entries, exits, sizing, or PnL.

## Layout

```
holygrail/          feature pipeline, labels, scaler, policy interface
holygrail/models/   majority, logistic, LightGBM, shallow MLP
scripts/            download, inspect, build dataset, train/eval
tests/              T1–T44 from spec v1.0
artifacts/          fitted RobustScaler + metadata
data/mt5_raw/       MT5 1m CSVs (not in git; download)
```

Policy contract (swap LightGBM / MLP / later RL without touching features):

```python
policy.fit(X, {"y_5m": ..., "y_30m": ..., "y_1h_structure": ...})
policy.predict_proba(X)  # dict of (n, 2) columns [-1, +1]
policy.explain(X)
```

## Rebuild

```bash
pip install -r requirements.txt
python scripts/download_mt5.py
python scripts/inspect_data.py
python -m pytest tests/test_t1_t44.py tests/test_structure_labels.py -q
python scripts/build_dataset.py --csv data/mt5_raw/EURUSD_M1_202101131130_202605270000_2020_2026.csv
python scripts/train_eval.py
```

`build_dataset.py` writes `data/processed/X_{train,val,test}.parquet` (scaled 128-col `X`), matching `y_*.parquet` (y_5m / y_30m / y_1h_structure for training; y_15m / y_1h unused comparison), and freezes `artifacts/scaler.joblib` + `artifacts/scaler.json`. Purge is 120 minutes so the 1h structure window does not cross the cutoff.

## Spec decisions

See the results report. Short version: timestamps are naive broker-server **open** times (not UTC); `std`/`iBands` ddof=0; ATR and RSI are Wilder; SMA shift is an MT5 bar-index offset; `live_1m` snapshots with developing 5m/30m/1h bars.

Authoritative spec: HolyGrail Feature Specification v1.0 (PDF).
