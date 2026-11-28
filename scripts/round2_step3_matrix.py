#!/usr/bin/env python3
"""Round 2 Step 3: combined matrix counts only. Do not train. Do not score test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from holygrail.io import load_mt5_m1
from holygrail.round2_columns import ROUND2_FEATURE_COLUMNS, ROUND2_FEATURE_VERSION
from holygrail.round2_matrix import build_round2_matrix, count_report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--csv",
        default="data/mt5_raw/EURUSD_M1_202101131130_202605270000_2020_2026.csv",
    )
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--out-json", default="")
    args = p.parse_args()

    m1 = load_mt5_m1(args.csv)
    if args.max_rows:
        m1 = m1.iloc[: args.max_rows]
    print(
        f"loaded {len(m1)} 1m bars {m1.index[0]} -> {m1.index[-1]}",
        flush=True,
    )
    table = build_round2_matrix(m1)
    report = count_report(table, m1)
    report["n_1m_loaded"] = int(len(m1))
    report["open_first"] = str(m1.index[0])
    report["open_last"] = str(m1.index[-1])
    report["n_named_columns"] = len(ROUND2_FEATURE_COLUMNS)
    report["feature_version"] = ROUND2_FEATURE_VERSION
    report["column_names"] = ROUND2_FEATURE_COLUMNS
    print(json.dumps({k: v for k, v in report.items() if k != "column_names"}, indent=2), flush=True)
    print(
        f"STOP: columns={report['n_columns']} rows={report['n_rows_after_gap_drop']} "
        f"dropped_gap={report['n_dropped_lookback_gap']}",
        flush=True,
    )
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
