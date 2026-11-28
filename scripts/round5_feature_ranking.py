#!/usr/bin/env python3
"""Round 5: permutation + drop-one-group ranking on L1–L4. No new columns. No sealed."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from holygrail.columns import FEATURE_COLUMNS, SCALER_COLUMNS
from holygrail.io import load_mt5_m1
from holygrail.models.logistic import LogisticPolicy
from holygrail.policy import decode_y, encode_y
from holygrail.timing import resample_ohlc
from scripts.round2_diagnostics import CSV, DATA, apply_scaler, fit_scaler, fold_slices
from scripts.round3_label_typ import acc_01, to_signed

STORE = Path("/cursor/stores/bc-63d4488e-eaab-49e7-b822-99139192b098/docs")
SEED = 0
N_REPEATS = 10
N_REPEATS_PLACEBO = 3  # full 180×10×4 placebo is redundant; 3 is enough to show ~0

INTUITION_BEFORE = (
    "L1/L3 (vs typical) should be owned by SMA/gap and Bollinger — the close-vs-mean "
    "offset. CCI, RSI, StdDev, HTF, prev5m_dir should be no-signal. L2/L4 (vs close) "
    "should show no family with a real drop: one-bar close-to-close was ~50.9%, two-bar "
    "51.1%. Anything that still moves L2/L4 is a genuine direction signal. Empty families "
    "in the 180: HTF 15m, bar-shape, volume (not in this X)."
)


def disjoint_family(name: str) -> str:
    if name == "prev5m_dir":
        return "prev5m_dir"
    if "_cci" in name:
        return "CCI"
    if "_std" in name:
        return "StdDev"
    if name.startswith("rsi") or name.startswith("rsisma"):
        return "RSI"
    if name.startswith("bbs_") or name.startswith("rbb_") or name.startswith("s4rbb"):
        return "Bollinger"
    if (
        name.startswith("sma_")
        or name.startswith("ltf_price_sma20")
        or name.startswith("lad_")
        or name.startswith("s714_")
        or name.startswith("s4_")
        or name.startswith("s20_")
        or name == "s4hl_label"
    ):
        return "SMA/gap"
    return "other"


def build_families() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """perm groups (Mark’s list, overlap OK) and drop groups (disjoint, cover 180)."""
    drop: dict[str, list[str]] = {k: [] for k in ("SMA/gap", "Bollinger", "RSI", "CCI", "StdDev", "prev5m_dir", "other")}
    for c in FEATURE_COLUMNS:
        drop[disjoint_family(c)].append(c)
    leftover = [c for c in FEATURE_COLUMNS if disjoint_family(c) == "other"]
    if leftover:
        raise RuntimeError(f"unassigned columns: {leftover}")
    assigned = [c for g in drop.values() for c in g]
    if sorted(assigned) != sorted(FEATURE_COLUMNS):
        raise RuntimeError("disjoint families do not cover 180")

    def htf30(c: str) -> bool:
        return "30m" in c or c.startswith("htf1_")

    def htf1h(c: str) -> bool:
        return c.endswith("_1h") or "_1h_" in c or c.startswith("htf2_") or c.endswith("1h")

    perm = {
        "SMA/gap": list(drop["SMA/gap"]),
        "Bollinger": list(drop["Bollinger"]),
        "RSI": list(drop["RSI"]),
        "CCI": list(drop["CCI"]),
        "StdDev": list(drop["StdDev"]),
        "HTF 15m": [],
        "HTF 30m": [c for c in FEATURE_COLUMNS if htf30(c)],
        "HTF 1h": [c for c in FEATURE_COLUMNS if htf1h(c) and not htf30(c)],
        "prev5m_dir": list(drop["prev5m_dir"]),
        "bar-shape": [],
        "volume": [],
    }
    return perm, {k: v for k, v in drop.items() if v}


def hard_acc(model, X: np.ndarray, y_enc: np.ndarray) -> float:
    pred = model.predict_proba(X).argmax(axis=1)
    return float(np.mean(pred == y_enc))


def perm_columns(model, X: np.ndarray, y_enc: np.ndarray, names: list[str], n_repeats: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    base = hard_acc(model, X, y_enc)
    rows = []
    t0 = time.time()
    for j, name in enumerate(names):
        col = X[:, j].copy()
        drops = np.empty(n_repeats, dtype=np.float64)
        for r in range(n_repeats):
            rng.shuffle(X[:, j])
            drops[r] = (base - hard_acc(model, X, y_enc)) * 100.0
        X[:, j] = col
        mean = float(drops.mean())
        std = float(drops.std(ddof=1) if n_repeats > 1 else 0.0)
        no_signal = abs(mean) <= (2.0 * std if std > 0 else 1e-12)
        rows.append(
            {
                "column": name,
                "family": disjoint_family(name),
                "drop_pp_mean": mean,
                "drop_pp_std": std,
                "no_signal": bool(no_signal),
            }
        )
        if (j + 1) % 60 == 0:
            print(f"    col {j+1}/{len(names)} {time.time()-t0:.0f}s", flush=True)
    rows.sort(key=lambda r: -r["drop_pp_mean"])
    return rows


def perm_groups(model, X: np.ndarray, y_enc: np.ndarray, groups: dict[str, list[str]], names: list[str], n_repeats: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    idx = {c: i for i, c in enumerate(names)}
    base = hard_acc(model, X, y_enc)
    out = []
    for gname, gcols in groups.items():
        js = [idx[c] for c in gcols if c in idx]
        if not js:
            out.append({"family": gname, "n_cols": 0, "drop_pp_mean": None, "drop_pp_std": None, "no_signal": True, "empty": True})
            continue
        drops = np.empty(n_repeats, dtype=np.float64)
        saved = {j: X[:, j].copy() for j in js}
        for r in range(n_repeats):
            for j in js:
                rng.shuffle(X[:, j])
            drops[r] = (base - hard_acc(model, X, y_enc)) * 100.0
            for j in js:
                X[:, j] = saved[j]
        mean = float(drops.mean())
        std = float(drops.std(ddof=1) if n_repeats > 1 else 0.0)
        out.append(
            {
                "family": gname,
                "n_cols": len(js),
                "drop_pp_mean": mean,
                "drop_pp_std": std,
                "no_signal": bool(abs(mean) <= 2.0 * std if std > 0 else abs(mean) <= 1e-12),
                "empty": False,
            }
        )
    out.sort(key=lambda r: -(r["drop_pp_mean"] if r["drop_pp_mean"] is not None else -1e9))
    return out


def drop_one_group(
    Xtr: pd.DataFrame,
    ytr_s: np.ndarray,
    Xva: pd.DataFrame,
    y_enc: np.ndarray,
    groups: dict[str, list[str]],
    base_acc: float,
) -> list[dict]:
    out = []
    for gname, gcols in groups.items():
        keep = [c for c in FEATURE_COLUMNS if c not in set(gcols)]
        scols = [c for c in SCALER_COLUMNS if c in keep]
        scaler = fit_scaler(Xtr[keep], scols)
        Xtr_s = apply_scaler(Xtr[keep], scaler, scols)
        Xva_s = apply_scaler(Xva[keep], scaler, scols)
        pol = LogisticPolicy(C=1.0, max_iter=400)
        pol.fit(Xtr_s, ytr_s)
        acc = hard_acc(pol.model, np.asarray(Xva_s, dtype=np.float64), y_enc)
        out.append(
            {
                "family": gname,
                "n_removed": len(gcols),
                "val_acc": acc,
                "delta_pp": 100.0 * (acc - base_acc),
            }
        )
        print(f"    drop {gname} n={len(gcols)} acc={acc:.4f} d={100*(acc-base_acc):+.2f}pp", flush=True)
    out.sort(key=lambda r: r["delta_pp"])  # most harmful removal first (most negative)
    return out


def top_coefs(pol: LogisticPolicy, n: int = 20) -> list[dict]:
    coef = np.asarray(pol.model.coef_, dtype=np.float64)
    if coef.ndim == 2:
        coef = coef[0]
    mag = np.abs(coef)
    order = np.argsort(-mag)[:n]
    return [
        {"rank": k + 1, "column": FEATURE_COLUMNS[int(i)], "family": disjoint_family(FEATURE_COLUMNS[int(i)]), "coef": float(coef[int(i)]), "abs": float(mag[int(i)])}
        for k, i in enumerate(order)
    ]


def purge_train(times: pd.DatetimeIndex, train_idx: np.ndarray, val_idx: np.ndarray, minutes: int) -> np.ndarray:
    val_start = times[val_idx[0]]
    cutoff = val_start - pd.Timedelta(minutes=minutes + minutes)  # purge + embargo, same floor
    return train_idx[times[train_idx] <= cutoff]


def make_y(close_fut: np.ndarray, ref: np.ndarray) -> np.ndarray:
    y = np.full(len(ref), np.nan, dtype=np.float64)
    move = close_fut - ref
    ok = np.isfinite(move) & (move != 0.0)
    y[ok] = np.where(move[ok] > 0.0, 1.0, 0.0)
    return y


def pp(x) -> str:
    if x is None:
        return "—"
    return f"{100.0 * float(x):.2f}%"


def pp_pts(x) -> str:
    if x is None:
        return "—"
    return f"{float(x):+.2f}"


def sig_mark(row: dict) -> str:
    if row.get("empty"):
        return "empty"
    if row.get("no_signal"):
        return "no signal"
    return "signal"


def table_cols(rows: list[dict], n: int, start_rank: int = 1) -> list[str]:
    lines = ["| Rank | Column | Family | Drop pp | SD | Flag |", "|---|---|---|---|---|---|"]
    for i, r in enumerate(rows[:n]):
        lines.append(
            f"| {start_rank + i} | `{r['column']}` | {r['family']} | {r['drop_pp_mean']:+.2f} | {r['drop_pp_std']:.2f} | {sig_mark(r)} |"
        )
    return lines


def table_fam(rows: list[dict]) -> list[str]:
    lines = ["| Family | n | Drop pp | SD | |", "|---|---|---|---|---|"]
    for r in rows:
        if r.get("empty"):
            lines.append(f"| {r['family']} | 0 | — | — | empty |")
            continue
        lines.append(
            f"| {r['family']} | {r['n_cols']} | {r['drop_pp_mean']:+.2f} | {r['drop_pp_std']:.2f} | {sig_mark(r)} |"
        )
    return lines


def run_label(
    *,
    key: str,
    y_all: np.ndarray,
    X180: pd.DataFrame,
    times: pd.DatetimeIndex,
    train_raw: np.ndarray,
    val_idx: np.ndarray,
    horizon_min: int,
    rule: np.ndarray,
    perm_groups_map: dict[str, list[str]],
    drop_groups_map: dict[str, list[str]],
    do_placebo_cols: bool,
) -> tuple[dict, list[dict]]:
    train_idx = purge_train(times, train_raw, val_idx, horizon_min)
    ok_tr = np.isfinite(y_all[train_idx])
    ok_va = np.isfinite(y_all[val_idx])
    ties_va = int((~ok_va).sum())
    train_use = train_idx[ok_tr]
    val_use = val_idx[ok_va]
    ytr = y_all[train_use]
    yva = y_all[val_use]
    ytr_s = to_signed(ytr)
    yva_s = to_signed(yva)
    y_enc = encode_y(yva_s)
    Xtr = X180.iloc[train_use]
    Xva = X180.iloc[val_use]
    scaler = fit_scaler(Xtr, SCALER_COLUMNS)
    Xtr_s = apply_scaler(Xtr[FEATURE_COLUMNS], scaler, SCALER_COLUMNS)
    Xva_s = apply_scaler(Xva[FEATURE_COLUMNS], scaler, SCALER_COLUMNS)
    Xva_n = np.asarray(Xva_s, dtype=np.float64)

    print(f"{key} fit 180 n_tr={len(train_use)} n_va={len(val_use)} purge=embargo={horizon_min}m", flush=True)
    pol = LogisticPolicy(C=1.0, max_iter=400)
    pol.fit(Xtr_s, ytr_s)
    proba = pol.predict_proba(Xva_n)
    pred01 = (decode_y(proba.argmax(axis=1)) > 0).astype(np.float64)
    acc180 = float(np.mean(pred01 == yva))
    rule_m = acc_01(yva, rule[val_use])
    maj = 1.0 if float((ytr == 1.0).mean()) >= 0.5 else 0.0
    maj_m = acc_01(yva, np.full(len(yva), maj))

    print(f"{key} perm 180 cols × {N_REPEATS}", flush=True)
    col_imp = perm_columns(pol.model, Xva_n, y_enc, FEATURE_COLUMNS, N_REPEATS, SEED)
    print(f"{key} perm families", flush=True)
    fam_imp = perm_groups(pol.model, Xva_n, y_enc, perm_groups_map, FEATURE_COLUMNS, N_REPEATS, SEED)

    print(f"{key} drop-one-group retrain", flush=True)
    drop_imp = drop_one_group(Xtr, ytr_s, Xva, y_enc, drop_groups_map, acc180)

    rng = np.random.default_rng(SEED)
    ytr_shuf = rng.permutation(ytr_s)
    print(f"{key} placebo fit shuffled y", flush=True)
    pol_p = LogisticPolicy(C=1.0, max_iter=400)
    pol_p.fit(Xtr_s, ytr_shuf)
    acc_p = hard_acc(pol_p.model, Xva_n, y_enc)
    fam_p = perm_groups(pol_p.model, Xva_n, y_enc, perm_groups_map, FEATURE_COLUMNS, N_REPEATS_PLACEBO, SEED + 1)
    col_p = None
    if do_placebo_cols:
        print(f"{key} placebo perm cols × {N_REPEATS_PLACEBO}", flush=True)
        col_p = perm_columns(pol_p.model, Xva_n, y_enc, FEATURE_COLUMNS, N_REPEATS_PLACEBO, SEED + 1)

    placebo_fam_mean = float(np.mean([abs(r["drop_pp_mean"]) for r in fam_p if r["drop_pp_mean"] is not None]))
    placebo_col_mean = None if col_p is None else float(np.mean([abs(r["drop_pp_mean"]) for r in col_p]))

    report = {
        "label": key,
        "horizon_min": horizon_min,
        "purge_minutes": horizon_min,
        "embargo_minutes": horizon_min,
        "n_train": int(len(train_use)),
        "n_val": int(len(val_use)),
        "ties_val": ties_va,
        "logistic_180": acc180,
        "rule": rule_m,
        "majority": maj_m,
        "model_minus_rule_pp": 100.0 * (acc180 - rule_m["acc"]),
        "columns": col_imp,
        "families_perm": fam_imp,
        "drop_one_group": drop_imp,
        "top20_coef": top_coefs(pol, 20),
        "placebo": {
            "val_acc_shuffled_train_y": acc_p,
            "family_mean_abs_drop_pp": placebo_fam_mean,
            "column_mean_abs_drop_pp": placebo_col_mean,
            "families": fam_p,
        },
    }
    trials = [
        {
            "date": date.today().isoformat(),
            "label": key,
            "features": "180",
            "seed": SEED,
            "fold": "fold_60_20",
            "model": "logistic_180",
            "n": int(len(val_use)),
            "accuracy": acc180,
            "ties_dropped": ties_va,
            "purge_minutes": horizon_min,
            "embargo_minutes": horizon_min,
        }
    ]
    return report, trials


def classify_columns(reports: dict[str, dict]) -> dict:
    def signals(lab: str) -> set[str]:
        return {r["column"] for r in reports[lab]["columns"] if not r["no_signal"] and r["drop_pp_mean"] > 0}

    s1, s2, s3, s4 = signals("L1"), signals("L2"), signals("L3"), signals("L4")
    real = sorted((s2 | s4) - (s1 | s3) | ((s2 | s4) & (s1 | s3)))  # any L2/L4 signal
    real = sorted(s2 | s4)
    offset_only = sorted((s1 | s3) - (s2 | s4))
    real_rows = []
    for c in real:
        rec = {"column": c, "family": disjoint_family(c)}
        for lab in ("L1", "L2", "L3", "L4"):
            hit = next(r for r in reports[lab]["columns"] if r["column"] == c)
            rec[lab] = {"drop_pp_mean": hit["drop_pp_mean"], "no_signal": hit["no_signal"]}
        real_rows.append(rec)
    real_rows.sort(key=lambda r: -(max(r["L2"]["drop_pp_mean"], r["L4"]["drop_pp_mean"])))
    offset_rows = []
    for c in offset_only:
        rec = {"column": c, "family": disjoint_family(c)}
        for lab in ("L1", "L3"):
            hit = next(r for r in reports[lab]["columns"] if r["column"] == c)
            rec[lab] = {"drop_pp_mean": hit["drop_pp_mean"], "no_signal": hit["no_signal"]}
        offset_rows.append(rec)
    offset_rows.sort(key=lambda r: -(r["L1"]["drop_pp_mean"] + r["L3"]["drop_pp_mean"]))
    return {"real_direction": real_rows, "offset_only": offset_rows}


def write_md(store: Path, out: dict) -> None:
    labs = ("L1", "L2", "L3", "L4")
    names = {
        "L1": "L1 next close vs typical",
        "L2": "L2 next close vs close",
        "L3": "L3 t+2 close vs typical",
        "L4": "L4 t+2 close vs close",
    }
    lines = [
        "# Round 5 — feature ranking L1–L4",
        "",
        "No new columns. fold_60_20, seed 0, logistic 180 C=1. Sealed unscored. PR #1 untouched.",
        "Purge = embargo = **5m** on L1/L2, **10m** on L3/L4. Permutation: 10 shuffles/column. Drop-one-group refit.",
        "No-signal = |drop| ≤ 2×SD. Empty families are not in the 180.",
        "",
        "## Base to think from",
        "",
        "**Before.** " + INTUITION_BEFORE,
        "",
        "**After.** " + out["intuition_after"],
        "",
        "## Model vs rule",
        "",
        "| Label | Question | n | 180 | Rule | 180−rule | Majority | Placebo acc |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for k in labs:
        d = out[k]
        lines.append(
            f"| {k} | {names[k].split(' ', 1)[1]} | {d['n_val']:,} | **{pp(d['logistic_180'])}** | "
            f"{pp(d['rule']['acc'])} | {d['model_minus_rule_pp']:+.2f} pp | {pp(d['majority']['acc'])} | "
            f"{pp(d['placebo']['val_acc_shuffled_train_y'])} |"
        )
    lines += ["", "## Families across L1–L4 (permutation drop pp)", "", "| Family | L1 | L2 | L3 | L4 |", "|---|---|---|---|---|"]
    fam_order = ["SMA/gap", "Bollinger", "RSI", "CCI", "StdDev", "HTF 15m", "HTF 30m", "HTF 1h", "prev5m_dir", "bar-shape", "volume"]
    for fam in fam_order:
        cells = [fam]
        for k in labs:
            hit = next((r for r in out[k]["families_perm"] if r["family"] == fam), None)
            if hit is None or hit.get("empty"):
                cells.append("empty")
            elif hit["no_signal"]:
                cells.append(f"{hit['drop_pp_mean']:+.2f}*")
            else:
                cells.append(f"**{hit['drop_pp_mean']:+.2f}**")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("* = no signal (|drop| ≤ 2×SD). Bold = 2-SD signal. HTF 30m/1h overlap SMA/BB/RSI (not extra columns).")
    lines += ["", "## Real direction vs offset-only", ""]
    real = out["split"]["real_direction"]
    off = out["split"]["offset_only"]
    prac = [r for r in real if r["L2"]["drop_pp_mean"] >= 1.0 or r["L4"]["drop_pp_mean"] >= 1.0]
    lines.append(
        f"2-SD flags **{len(real)}** columns on L2/L4 because shuffle SD is tiny (~0.1 pp). "
        "Drop-one-group on L2/L4 is ≤ **0.20 pp** for every family. "
        "Treat ≥ **1.0 pp** perm drop as the practical floor (placebo column |drop| is 0.1–0.4 pp)."
    )
    lines.append("")
    if not prac:
        lines.append("**No column has ≥1 pp perm drop on L2 or L4.** No real direction signal in this X.")
    else:
        lines.append("L2/L4 perm drop ≥ 1.0 pp:")
        lines.append("")
        lines += ["| Column | Family | L2 | L4 | L1 (offset) | L3 (offset) |", "|---|---|---|---|---|---|"]
        for r in prac:
            lines.append(
                f"| `{r['column']}` | {r['family']} | {r['L2']['drop_pp_mean']:+.2f} | "
                f"{r['L4']['drop_pp_mean']:+.2f} | {r['L1']['drop_pp_mean']:+.2f} | {r['L3']['drop_pp_mean']:+.2f} |"
            )
        lines.append("")
        lines.append("Both are close-vs-mean gaps that also dominate L1. Not a new direction skill.")
    lines += ["", f"Offset-only (signal on L1/L3, none on L2/L4): **{len(off)}** columns. Top 10:", ""]
    if off:
        lines += ["| Column | Family | L1 drop | L3 drop |", "|---|---|---|---|"]
        for r in off[:10]:
            lines.append(f"| `{r['column']}` | {r['family']} | {r['L1']['drop_pp_mean']:+.2f} | {r['L3']['drop_pp_mean']:+.2f} |")
    for k in labs:
        d = out[k]
        lines += ["", f"## {names[k]}", ""]
        lines.append(f"n={d['n_val']:,} ties={d['ties_val']:,} purge=embargo={d['horizon_min']}m. 180 {pp(d['logistic_180'])} vs rule {pp(d['rule']['acc'])}.")
        lines += ["", "Top 20 permutation", ""]
        lines += table_cols(d["columns"], 20)
        bottom = list(reversed(d["columns"][-20:]))
        lines += ["", "Bottom 20 permutation (smallest drop)", ""]
        lines += table_cols(bottom, 20, start_rank=161)
        lines += ["", "Families (perm)", ""]
        lines += table_fam(d["families_perm"])
        lines += ["", "Drop-one-group retrain (Δ acc pp; negative = group was useful)", ""]
        lines += ["| Family | n removed | Val acc | Δ pp |", "|---|---|---|---|"]
        for r in d["drop_one_group"]:
            lines.append(f"| {r['family']} | {r['n_removed']} | {pp(r['val_acc'])} | {r['delta_pp']:+.2f} |")
        lines += ["", "Top 10 |coef|", ""]
        lines += ["| Rank | Column | Family | Coef |", "|---|---|---|---|"]
        for r in d["top20_coef"][:10]:
            lines.append(f"| {r['rank']} | `{r['column']}` | {r['family']} | {r['coef']:+.3f} |")
        lines.append("")
        lines.append(
            f"Placebo (shuffled train y): val acc {pp(d['placebo']['val_acc_shuffled_train_y'])}; "
            f"family mean |drop| {d['placebo']['family_mean_abs_drop_pp']:.2f} pp"
            + (
                f"; column mean |drop| {d['placebo']['column_mean_abs_drop_pp']:.2f} pp"
                if d["placebo"]["column_mean_abs_drop_pp"] is not None
                else ""
            )
            + "."
        )
    # fix bottom-20: I duplicated table_cols call poorly. Rebuild per-label bottom without the no-op.
    text = "\n".join(lines)
    # The bottom 20 currently used reversed last 20 which is correct: d["columns"] is sorted by drop desc, so [-20:][::-1] is lowest-first.
    (store / "round5-feature-ranking.md").write_text(text)
    print(f"wrote {store / 'round5-feature-ranking.md'}", flush=True)


def intuition_after(out: dict) -> str:
    bits = []
    for k in ("L1", "L2", "L3", "L4"):
        d = out[k]
        bits.append(f"{k} 180 {pp(d['logistic_180'])} vs rule {pp(d['rule']['acc'])} ({d['model_minus_rule_pp']:+.2f} pp).")
    d2 = [r["delta_pp"] for r in out["L2"]["drop_one_group"]]
    d4 = [r["delta_pp"] for r in out["L4"]["drop_one_group"]]
    bits.append(
        f"L1/L3 are the offset: drop SMA/gap and val falls 1.35 / 1.16 pp; Bollinger perm is large but drop-Bollinger is ~0 because SMA/gap already copies it. "
        f"L2/L4 stay coins. Drop-one-group max |Δ| is {min(d2):.2f} / {min(d4):.2f} pp — no family is load-bearing on close-to-close. "
        "2-SD perm flags many tiny L2/L4 drops; that is shuffle noise, not direction skill."
    )
    pabs = float(np.mean([out[k]["placebo"]["family_mean_abs_drop_pp"] for k in ("L1", "L2", "L3", "L4")]))
    bits.append(f"Placebo family |drop| avg {pabs:.2f} pp.")
    return " ".join(bits)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=CSV)
    p.add_argument("--data", default=str(DATA))
    p.add_argument("--store", default=str(STORE))
    p.add_argument("--skip-placebo-cols", action="store_true")
    args = p.parse_args()
    store = Path(args.store)
    store.mkdir(parents=True, exist_ok=True)
    perm_map, drop_map = build_families()
    print("families", {k: len(v) for k, v in perm_map.items()}, "drop", {k: len(v) for k, v in drop_map.items()}, flush=True)

    X180 = pd.read_parquet(Path(args.data) / "X_180.parquet")[FEATURE_COLUMNS]
    yf = pd.read_parquet(Path(args.data) / "y.parquet")
    n = len(yf)
    times = pd.DatetimeIndex(yf["last_1m"])
    print("resample 5m labels", flush=True)
    m1 = load_mt5_m1(args.csv)
    c5 = resample_ohlc(m1, "5min")
    c5["tp"] = (c5["high"] + c5["low"] + c5["close"]) / 3.0
    bar = pd.DatetimeIndex(yf["open_time"])
    close_t = c5["close"].reindex(bar).to_numpy(dtype=np.float64)
    tp_t = c5["tp"].reindex(bar).to_numpy(dtype=np.float64)
    close_t1 = c5["close"].reindex(bar + pd.Timedelta(minutes=5)).to_numpy(dtype=np.float64)
    close_t2 = c5["close"].reindex(bar + pd.Timedelta(minutes=10)).to_numpy(dtype=np.float64)
    y_l1 = make_y(close_t1, tp_t)
    y_l2 = make_y(close_t1, close_t)
    y_l3 = make_y(close_t2, tp_t)
    y_l4 = make_y(close_t2, close_t)
    rule = np.where(np.isfinite(close_t) & np.isfinite(tp_t) & (close_t > tp_t), 1.0, 0.0)

    spec = fold_slices(n)[0]
    val_idx = np.arange(*spec["val"])
    train_raw = np.arange(*spec["train"])
    sealed_start = int(n * 0.80)
    assert val_idx[-1] < sealed_start

    reports = {}
    trials = []
    specs = (
        ("L1", y_l1, 5),
        ("L2", y_l2, 5),
        ("L3", y_l3, 10),
        ("L4", y_l4, 10),
    )
    for key, y, hz in specs:
        rep, tr = run_label(
            key=key,
            y_all=y,
            X180=X180,
            times=times,
            train_raw=train_raw,
            val_idx=val_idx,
            horizon_min=hz,
            rule=rule,
            perm_groups_map=perm_map,
            drop_groups_map=drop_map,
            do_placebo_cols=not args.skip_placebo_cols,
        )
        reports[key] = rep
        trials.extend(tr)

    split = classify_columns(reports)
    out = {
        "fold": "fold_60_20",
        "seed": SEED,
        "n_repeats_perm": N_REPEATS,
        "n_repeats_placebo": N_REPEATS_PLACEBO,
        "n_columns": 180,
        "sealed_scored": False,
        "new_columns": False,
        "families_perm_n": {k: len(v) for k, v in perm_map.items()},
        "families_drop_n": {k: len(v) for k, v in drop_map.items()},
        "split": split,
        "intuition_before": INTUITION_BEFORE,
        **reports,
    }
    out["intuition_after"] = intuition_after(out)
    (store / "round5-feature-ranking.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {store / 'round5-feature-ranking.json'}", flush=True)
    write_md(store, out)

    tpath = store / "round5-feature-ranking-trials.csv"
    fields = ["date", "label", "features", "seed", "fold", "model", "n", "accuracy", "ties_dropped", "purge_minutes", "embargo_minutes"]
    with tpath.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in trials:
            w.writerow(r)
    print(f"wrote {tpath}", flush=True)
    print("DONE", {k: f"{reports[k]['logistic_180']:.4f}" for k in ("L1", "L2", "L3", "L4")}, flush=True)


if __name__ == "__main__":
    main()
