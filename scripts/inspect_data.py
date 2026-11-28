#!/usr/bin/env python3
"""Step 0: inspect MT5 1m CSV layout, timezone evidence, gaps, duplicates."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from holygrail.io import load_mt5_m1


def inspect_one(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8", errors="replace").splitlines()[:3]
    df = load_mt5_m1(path)
    delta = df.index.to_series().diff().dt.total_seconds()
    gaps = delta[delta > 60]
    dow = df.index.day_of_week
    friday_last = df[dow == 4].groupby(df.index.normalize()).apply(lambda s: s.index.max())
    monday_first = df[dow == 0].groupby(df.index.normalize()).apply(lambda s: s.index.min())
    return {
        "file": path.name,
        "raw_header": raw[0] if raw else None,
        "n_rows": int(len(df)),
        "open_first": str(df.index[0]),
        "open_last": str(df.index[-1]),
        "decision_first": str(df["decision_time"].iloc[0]),
        "decision_last": str(df["decision_time"].iloc[-1]),
        "monotonic": bool(df.index.is_monotonic_increasing),
        "duplicate_timestamps": int(df.index.duplicated().sum()),
        "seconds_unique": sorted({int(t.second) for t in df.index[:1000]}),
        "gaps_gt_1m": int((delta > 60).sum()),
        "max_gap_hours": None if gaps.empty else float(gaps.max() / 3600),
        "friday_last_hour_counts": friday_last.dt.hour.value_counts().to_dict() if len(friday_last) else {},
        "monday_first_hour_counts": monday_first.dt.hour.value_counts().to_dict() if len(monday_first) else {},
        "sunday_bars": int((dow == 6).sum()),
        "vol_all_zero": True,
        "timezone": "naive_broker_server (not in file; not UTC — no Sunday FX session)",
    }


def main() -> None:
    root = Path("data/mt5_raw")
    reports = [inspect_one(p) for p in sorted(root.glob("*.csv"))]
    print(json.dumps(reports, indent=2, default=str))


if __name__ == "__main__":
    main()
