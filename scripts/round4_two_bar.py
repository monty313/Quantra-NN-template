#!/usr/bin/env python3
"""Round 4: two-bar labels A/B. Logistic 180/128. fold_60_20. No sealed. No new columns."""

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

from holygrail.columns import FEATURE_COLUMNS, FEATURE_COLUMNS_V1_0, SCALER_COLUMNS
from holygrail.io import load_mt5_m1
from holygrail.models.metrics import evaluate
from holygrail.policy import decode_y
from holygrail.timing import resample_ohlc
from scripts.round2_diagnostics import CSV, DATA, fold_slices
from scripts.round3_label_typ import (
    SCALER_128,
    SEED,
    SHUFFLE_BLOCK,
    acc_01,
    apply_scaler,
    block_shuffle,
    equal_count_calibration,
    fit_logistic,
    fit_scaler,
    group_acc,
    pp,
    to_01_from_signed,
    to_signed,
)

STORE = Path("/cursor/stores/bc-63d4488e-eaab-49e7-b822-99139192b098/docs")
# Label looks 10 minutes ahead (2 × 5m). Both at the required floor.
PURGE_MINUTES = 10
EMBARGO_MINUTES = 10

INTUITION_BEFORE = """\
**Before the numbers.** Two-bar y is the honest next question after Round 3: if 1-bar
next-close-vs-typical is just this close vs this typical (rule 61.90%, model −0.15 pp),
then looking at bar t+2 either (1) the known offset still leaks into y because FX 5m
closes stay near this close, so A is a weaker copy of R (~55–58%), or (2) two bars is
enough that R dies and we finally measure leftover skill. CEO G1 said leftover 5m
direction on this X is a coin; A-residual vs 50% is that test on a longer horizon.
B is two-bar close-to-close. One-bar close-to-close was already ~50.9%. Belief: B is
still a coin, R is the wrong baseline for B, and close-to-close persistence
(sign(close_t − close_{t−1})) is the fair sibling — also near 50%. The 180 will still
agree with R on A because X is a close-vs-mean machine. No new columns can change that.
"""


def purge_train(times: pd.DatetimeIndex, train_idx: np.ndarray, val_idx: np.ndarray) -> np.ndarray:
    val_start = times[val_idx[0]]
    cutoff = val_start - pd.Timedelta(minutes=PURGE_MINUTES + EMBARGO_MINUTES)
    return train_idx[times[train_idx] <= cutoff]


def pack_trial(label, features, model, n, acc, log_loss, ties) -> dict:
    return {
        "date": date.today().isoformat(),
        "label": label,
        "features": features,
        "seed": SEED,
        "fold": "fold_60_20",
        "model": model,
        "n": None if n is None else int(n),
        "accuracy": acc,
        "log_loss": log_loss,
        "ties_dropped": ties,
        "purge_minutes": PURGE_MINUTES,
        "embargo_minutes": EMBARGO_MINUTES,
    }


def append_trials(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "date",
        "label",
        "features",
        "seed",
        "fold",
        "model",
        "n",
        "accuracy",
        "log_loss",
        "ties_dropped",
        "purge_minutes",
        "embargo_minutes",
    ]
    exists = path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    print(f"wrote {len(rows)} trials -> {path}", flush=True)


def run_one(
    *,
    name: str,
    y_all: np.ndarray,
    missing: np.ndarray,
    X180: pd.DataFrame,
    times: pd.DatetimeIndex,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    rule: np.ndarray,
    prev: np.ndarray,
    persist_close: np.ndarray | None,
) -> tuple[dict, list[dict], np.ndarray]:
    finite = np.isfinite(y_all)
    ties = ~finite & ~missing
    tr_ok = finite[train_idx]
    va_ok = finite[val_idx]
    train_use = train_idx[tr_ok]
    val_use = val_idx[va_ok]
    ytr = y_all[train_use]
    yva = y_all[val_use]
    ytr_s = to_signed(ytr)
    yva_s = to_signed(yva)
    Xtr = X180.iloc[train_use]
    Xva = X180.iloc[val_use]

    print(f"{name} logistic 180 train={len(train_use)} val={len(val_use)}", flush=True)
    pol180, proba180 = fit_logistic(Xtr, ytr_s, Xva, FEATURE_COLUMNS, SCALER_COLUMNS)
    met180 = evaluate(yva_s, proba180)
    pred180 = to_01_from_signed(decode_y(proba180.argmax(axis=1)))
    p_up = np.clip(proba180[:, 1], 1e-7, 1.0 - 1e-7)

    print(f"{name} logistic 128", flush=True)
    _pol128, proba128 = fit_logistic(Xtr, ytr_s, Xva, FEATURE_COLUMNS_V1_0, SCALER_128)
    met128 = evaluate(yva_s, proba128)

    rule_va = rule[val_use]
    maj_cls = 1.0 if float((ytr == 1.0).mean()) >= 0.5 else 0.0
    maj_pred = np.full(len(yva), maj_cls)
    prev_va = prev[val_use]
    prev_ok = np.isfinite(prev_va) & (prev_va != 0.0)
    prev_pred = np.where(prev_va > 0, 1.0, 0.0)
    rule_m = acc_01(yva, rule_va)
    maj_m = acc_01(yva, maj_pred)
    prev_m = acc_01(yva[prev_ok], prev_pred[prev_ok])

    persist_m = None
    if persist_close is not None:
        pc = persist_close[val_use]
        pc_ok = np.isfinite(pc)
        persist_m = acc_01(yva[pc_ok], pc[pc_ok])

    agree = pred180 == rule_va
    n_ag = int(agree.sum())
    n_dg = int((~agree).sum())
    acc_ag = float(np.mean(pred180[agree] == yva[agree])) if n_ag else None
    acc_dg = float(np.mean(pred180[~agree] == yva[~agree])) if n_dg else None
    residual = yva != rule_va
    n_res = int(residual.sum())
    res_acc = float(np.mean(pred180[residual] == yva[residual])) if n_res else None

    hours = times[val_use].hour.to_numpy()
    weekdays = times[val_use].day_of_week.to_numpy()
    rng = np.random.default_rng(SEED)
    placebo_label = acc_01(rng.permutation(yva), pred180)
    placebo_time = acc_01(block_shuffle(yva, SHUFFLE_BLOCK, rng), pred180)

    report = {
        "label": name,
        "n_train": int(len(train_use)),
        "n_val": int(len(val_use)),
        "ties_train": int((ties[train_idx]).sum()),
        "ties_val": int((ties[val_idx]).sum()),
        "missing_t2_train": int(missing[train_idx].sum()),
        "missing_t2_val": int(missing[val_idx].sum()),
        "logistic_180": {
            "n": int(len(yva)),
            "accuracy": met180["accuracy"],
            "log_loss": met180["log_loss"],
            "ece": met180["ece"],
        },
        "logistic_128": {
            "n": int(len(yva)),
            "accuracy": met128["accuracy"],
            "log_loss": met128["log_loss"],
            "ece": met128["ece"],
        },
        "baselines": {
            "rule_close_gt_typical": rule_m,
            "majority": {**maj_m, "train_majority": maj_cls},
            "prev5m_dir": {**prev_m, "zeros_dropped": int((~prev_ok).sum())},
            "persist_close_t_vs_tminus1": persist_m,
        },
        "model_minus_rule_pp": 100.0 * (met180["accuracy"] - rule_m["acc"]),
        "disagreement": {
            "n_agree": n_ag,
            "acc_agree": acc_ag,
            "n_disagree": n_dg,
            "acc_disagree": acc_dg,
            "agree_rate": float(n_ag / len(yva)) if len(yva) else None,
        },
        "residual_y_ne_rule": {"n": n_res, "acc": res_acc, "vs_50": None if res_acc is None else res_acc - 0.5},
        "calibration_10bin": equal_count_calibration(yva, p_up, 10),
        "acc_by_hour": group_acc(yva, pred180, hours),
        "acc_by_weekday": group_acc(yva, pred180, weekdays),
        "placebo_shuffled_label": placebo_label,
        "placebo_shuffled_time": placebo_time,
    }
    trials = [
        pack_trial(name, "180", "logistic_180", report["logistic_180"]["n"], met180["accuracy"], met180["log_loss"], report["ties_val"]),
        pack_trial(name, "128", "logistic_128", report["logistic_128"]["n"], met128["accuracy"], met128["log_loss"], report["ties_val"]),
        pack_trial(name, "none", "rule_close_gt_typical", rule_m["n"], rule_m["acc"], None, 0),
        pack_trial(name, "none", "majority", maj_m["n"], maj_m["acc"], None, 0),
        pack_trial(name, "prev5m_dir", "prev5m_dir", prev_m["n"], prev_m["acc"], None, int((~prev_ok).sum())),
        pack_trial(name, "180", "placebo_shuffled_label", placebo_label["n"], placebo_label["acc"], None, 0),
        pack_trial(name, "180", "placebo_shuffled_time_1h", placebo_time["n"], placebo_time["acc"], None, 0),
    ]
    if persist_m is not None:
        trials.append(
            pack_trial(name, "close_t_vs_tminus1", "persist_close", persist_m["n"], persist_m["acc"], None, persist_m["n_dropped"])
        )
    return report, trials, pred180


def cal_table(bins: list[dict]) -> list[str]:
    lines = [
        "| Bin | n | Mean pred | Actual rate |",
        "|---|---|---|---|",
    ]
    for b in bins:
        if b["n"] == 0:
            lines.append(f"| {b['bin']} | 0 | — | — |")
        else:
            lines.append(f"| {b['bin']} | {b['n']:,} | {b['mean_pred']:.4f} | {b['actual_rate']:.4f} |")
    return lines


def hour_table(rows: list[dict]) -> list[str]:
    lines = ["| Hour | n | Acc |", "|---|---|---|"]
    for r in rows:
        lines.append(f"| {int(r['key']):02d} | {r['n']:,} | {pp(r['acc'])} |")
    return lines


def wd_table(rows: list[dict]) -> list[str]:
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    lines = ["| Weekday | n | Acc |", "|---|---|---|"]
    for r in rows:
        name = names[int(r["key"])] if str(r["key"]).isdigit() and 0 <= int(r["key"]) < 7 else r["key"]
        lines.append(f"| {name} | {r['n']:,} | {pp(r['acc'])} |")
    return lines


def label_section(title: str, d: dict, *, residual: bool) -> list[str]:
    m180 = d["logistic_180"]
    m128 = d["logistic_128"]
    rule = d["baselines"]["rule_close_gt_typical"]
    maj = d["baselines"]["majority"]
    prev = d["baselines"]["prev5m_dir"]
    dis = d["disagreement"]
    lines = [
        f"## {title}",
        "",
        f"- train n (purge + finite y): **{d['n_train']:,}**",
        f"- val n finite y: **{d['n_val']:,}**",
        f"- ties dropped train / val: **{d['ties_train']:,}** / **{d['ties_val']:,}**",
        f"- missing t+2 dropped train / val: **{d['missing_t2_train']:,}** / **{d['missing_t2_val']:,}**",
        "",
        "| Model / baseline | n | Acc |",
        "|---|---|---|",
        f"| **Logistic 180** | {m180['n']:,} | **{pp(m180['accuracy'])}** |",
        f"| Logistic 128 | {m128['n']:,} | {pp(m128['accuracy'])} |",
        f"| Rule: this close > this typical else 0 | {rule['n']:,} | **{pp(rule['acc'])}** |",
        f"| Majority (train mode = {int(maj['train_majority'])}) | {maj['n']:,} | {pp(maj['acc'])} |",
        f"| prev5m_dir (zeros dropped) | {prev['n']:,} | {pp(prev['acc'])} |",
    ]
    pc = d["baselines"]["persist_close_t_vs_tminus1"]
    if pc is not None:
        lines.append(f"| persist sign(close_t − close_t−1) | {pc['n']:,} | {pp(pc['acc'])} |")
    lines += [
        "",
        f"**KEY: logistic 180 minus rule = {d['model_minus_rule_pp']:+.2f} pp.**",
        f"Logloss 180 {m180['log_loss']:.4f}. Logloss 128 {m128['log_loss']:.4f}.",
        "",
        "### Agree / disagree with the rule",
        "",
        f"- Agree n={dis['n_agree']:,} acc={pp(dis['acc_agree'])} (rate {pp(dis['agree_rate'])})",
        f"- Disagree n={dis['n_disagree']:,} acc={pp(dis['acc_disagree'])}",
        "",
    ]
    if residual:
        r = d["residual_y_ne_rule"]
        vs = "n/a" if r["vs_50"] is None else f"{100.0 * r['vs_50']:+.2f} pp"
        lines += [
            "### Residual of A (y ≠ rule) vs 50%",
            "",
            f"- n={r['n']:,} model acc={pp(r['acc'])} (vs 50%: {vs})",
            "",
        ]
    lines += ["### 10-bin calibration (equal-count, P(y=1), logistic 180)", ""]
    lines += cal_table(d["calibration_10bin"]["bins"])
    lines += ["", "### Acc by hour", ""]
    lines += hour_table(d["acc_by_hour"])
    lines += ["", "### Acc by weekday", ""]
    lines += wd_table(d["acc_by_weekday"])
    lines += [
        "",
        "### Placebo (frozen 180 preds vs shuffled val y)",
        "",
        f"- Shuffled-label: n={d['placebo_shuffled_label']['n']:,} acc={pp(d['placebo_shuffled_label']['acc'])}",
        f"- Shuffled-time 1h blocks: n={d['placebo_shuffled_time']['n']:,} acc={pp(d['placebo_shuffled_time']['acc'])}",
        "",
    ]
    return lines


def write_md(store: Path, out: dict) -> None:
    a = out["A"]
    b = out["B"]
    after = out["intuition_after"]
    lines = [
        "# Round 4 — two-bar labels A and B",
        "",
        "No new columns. Same 180-col machine. fold_60_20, seed 0, logistic C=1, train-only scaler.",
        "Sealed newest 20% unscored. No LightGBM/MLP. PR #1 untouched. Observation only.",
        "",
        f"JSON: `{store / 'round4-two-bar.json'}`",
        "",
        "## Base to think from",
        "",
        INTUITION_BEFORE,
        "",
        "**What the numbers did to that belief.**",
        "",
        after,
        "",
        "## Machine",
        "",
        f"- Prediction time: right after a 5m bar closes, once every 5 minutes.",
        f"- Purge: **{PURGE_MINUTES} minutes**. Embargo: **{EMBARGO_MINUTES} minutes**. "
        f"Train cutoff = val_start − {PURGE_MINUTES + EMBARGO_MINUTES} m. "
        "Both at the 10-minute (2-bar) floor. Stage 1 1h-head embargo of 60m is not used here.",
        f"- n parquet rows: {out['n_parquet']:,}. Sealed: {out['n_sealed']:,} (not scored).",
        f"- Missing t+2 on all parquet rows: {out['n_missing_t2']:,}.",
        "- Leak tests (pytest): shift y one 5m bar → X unchanged; mutate close of bar t+2 → y flips, X at t unchanged.",
        "- Labels never in X. `y_5m_dir` and Round 3 y kept.",
        "",
        "## Side by side vs Round 3 (one-bar)",
        "",
        "| | 1-bar (Round 3 / Round 2) | 2-bar (this) logistic 180 | 2-bar rule | 180 − rule |",
        "|---|---|---|---|---|",
        f"| A close vs typical | 61.75% / rule 61.90% | {pp(a['logistic_180']['accuracy'])} | {pp(a['baselines']['rule_close_gt_typical']['acc'])} | {a['model_minus_rule_pp']:+.2f} pp |",
        f"| B close vs close | ~50.9% close-to-close 1-bar | {pp(b['logistic_180']['accuracy'])} | {pp(b['baselines']['rule_close_gt_typical']['acc'])} | {b['model_minus_rule_pp']:+.2f} pp |",
        "",
    ]
    lines += label_section("A — `y_5m_close2_vs_typ`", a, residual=True)
    lines += label_section("B — `y_5m_close2_vs_close`", b, residual=False)
    lines += [
        "## Notes",
        "",
        "- 0/1 mapped to −1/+1 only at `LogisticPolicy.fit`.",
        "- Rule baseline is always this bar’s close vs this bar’s typical (known at t), for both labels.",
        "- B also reports sign(close_t − close_{t−1}) persistence; zeros/missing dropped from that n.",
        "- Residual of A is model acc on val rows where y_A ≠ rule, vs 50%. Not a new feature.",
        "",
    ]
    (store / "round4-two-bar.md").write_text("\n".join(lines))
    print(f"wrote {store / 'round4-two-bar.md'}", flush=True)


def intuition_after(a: dict, b: dict) -> str:
    ra = a["baselines"]["rule_close_gt_typical"]["acc"]
    rb = b["baselines"]["rule_close_gt_typical"]["acc"]
    ma = a["logistic_180"]["accuracy"]
    mb = b["logistic_180"]["accuracy"]
    res = a["residual_y_ne_rule"]["acc"]
    agr_a = a["disagreement"]["agree_rate"]
    agr_b = b["disagreement"]["agree_rate"]
    dg_a = a["disagreement"]["acc_disagree"]
    bits = []
    bits.append(
        f"A: 180 is {pp(ma)} vs rule {pp(ra)} ({a['model_minus_rule_pp']:+.2f} pp). "
        f"Agree-with-rule {pp(agr_a)}; disagree acc {pp(dg_a)}. "
        f"Residual (y≠R) n={a['residual_y_ne_rule']['n']:,} acc {pp(res)} vs 50%."
    )
    if ra is not None and ra >= 0.54:
        bits.append("R still owns A at two bars — the known offset survives a 10-minute wait. Belief (1) held.")
    else:
        bits.append("R died on A. Two bars was enough to wash out close-vs-typical. Belief (1) was wrong.")
    if res is not None and res > 0.55:
        bits.append("A-residual beat 50%. First leftover-skill signal on this X. Do not add columns until placebos and 128 vs 180 agree.")
    elif res is not None and res < 0.40:
        bits.append(
            "A-residual is far below 50% because the 180 still copies R, so it is systematically wrong "
            "on R’s misses. A coin would be 50%. CEO G1 stands: no leftover 5m direction skill."
        )
    elif res is not None and abs(res - 0.5) < 0.02:
        bits.append("A-residual is a coin. CEO G1 stands: leftover 5m direction on this X is dead.")
    else:
        bits.append("A-residual is not a clear leftover skill. Do not add columns on it.")
    bits.append(
        f"B: 180 is {pp(mb)} vs rule {pp(rb)} ({b['model_minus_rule_pp']:+.2f} pp), agree-with-rule {pp(agr_b)}. "
        f"One-bar close-to-close was ~50.9%."
    )
    if mb is not None and mb < 0.53:
        bits.append("B is still a coin. Two-bar close-to-close did not create a forecast. Belief on B held.")
    elif mb is not None and mb > 0.58:
        bits.append("B moved off a coin. Check it is not secretly copying R (agree-with-rule).")
    pc = b["baselines"]["persist_close_t_vs_tminus1"]
    if pc and pc.get("acc") is not None:
        bits.append(f"B persist sign(close_t − close_t−1) = {pp(pc['acc'])}.")
    bits.append(
        f"Placebos A {pp(a['placebo_shuffled_label']['acc'])} / {pp(a['placebo_shuffled_time']['acc'])}; "
        f"B {pp(b['placebo_shuffled_label']['acc'])} / {pp(b['placebo_shuffled_time']['acc'])}."
    )
    return " ".join(bits)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=CSV)
    p.add_argument("--data", default=str(DATA))
    p.add_argument("--store", default=str(STORE))
    args = p.parse_args()
    data = Path(args.data)
    store = Path(args.store)
    store.mkdir(parents=True, exist_ok=True)

    print("load parquet", flush=True)
    X180 = pd.read_parquet(data / "X_180.parquet")[FEATURE_COLUMNS]
    yf = pd.read_parquet(data / "y.parquet")
    n = len(yf)
    assert len(X180) == n
    times = pd.DatetimeIndex(yf["last_1m"])
    sealed_n = int((yf["split"] == "sealed").sum()) if "split" in yf.columns else int(n * 0.20)

    print("compute two-bar labels from 1m", flush=True)
    m1 = load_mt5_m1(args.csv)
    c5 = resample_ohlc(m1, "5min")
    c5["tp"] = (c5["high"] + c5["low"] + c5["close"]) / 3.0
    bar = pd.DatetimeIndex(yf["open_time"])
    t2 = bar + pd.Timedelta(minutes=10)
    t1_prev = bar - pd.Timedelta(minutes=5)
    close_t = c5["close"].reindex(bar).to_numpy(dtype=np.float64)
    tp_t = c5["tp"].reindex(bar).to_numpy(dtype=np.float64)
    close_t2 = c5["close"].reindex(t2).to_numpy(dtype=np.float64)
    close_tm1 = c5["close"].reindex(t1_prev).to_numpy(dtype=np.float64)
    missing = ~np.isfinite(close_t2)
    n_missing = int(missing.sum())

    def make_y(ref: np.ndarray) -> np.ndarray:
        y = np.full(n, np.nan, dtype=np.float64)
        move = close_t2 - ref
        ok = np.isfinite(move) & (move != 0.0)
        y[ok] = np.where(move[ok] > 0.0, 1.0, 0.0)
        return y

    y_a = make_y(tp_t)
    y_b = make_y(close_t)
    rule = np.where(np.isfinite(close_t) & np.isfinite(tp_t) & (close_t > tp_t), 1.0, 0.0)
    prev = yf["prev5m_dir"].to_numpy(dtype=np.float64)
    persist_close = np.full(n, np.nan, dtype=np.float64)
    dclose = close_t - close_tm1
    pok = np.isfinite(dclose) & (dclose != 0.0)
    persist_close[pok] = np.where(dclose[pok] > 0.0, 1.0, 0.0)

    spec = fold_slices(n)[0]
    val_idx = np.arange(*spec["val"])
    train_idx = purge_train(times, np.arange(*spec["train"]), val_idx)
    sealed_start = int(n * 0.80)
    assert val_idx[-1] < sealed_start
    print(
        f"fold_60_20 purge={PURGE_MINUTES}m embargo={EMBARGO_MINUTES}m "
        f"train_idx {len(train_idx)} val_idx {len(val_idx)} missing_t2={n_missing}",
        flush=True,
    )

    a_rep, a_trials, _ = run_one(
        name="y_5m_close2_vs_typ",
        y_all=y_a,
        missing=missing,
        X180=X180,
        times=times,
        train_idx=train_idx,
        val_idx=val_idx,
        rule=rule,
        prev=prev,
        persist_close=None,
    )
    b_rep, b_trials, _ = run_one(
        name="y_5m_close2_vs_close",
        y_all=y_b,
        missing=missing,
        X180=X180,
        times=times,
        train_idx=train_idx,
        val_idx=val_idx,
        rule=rule,
        prev=prev,
        persist_close=persist_close,
    )

    after = intuition_after(a_rep, b_rep)
    out = {
        "fold": "fold_60_20",
        "seed": SEED,
        "purge_minutes": PURGE_MINUTES,
        "embargo_minutes": EMBARGO_MINUTES,
        "n_parquet": n,
        "n_sealed": sealed_n,
        "n_missing_t2": n_missing,
        "sealed_scored": False,
        "new_columns": False,
        "A": a_rep,
        "B": b_rep,
        "intuition_before": INTUITION_BEFORE,
        "intuition_after": after,
        "round3_ref": {
            "A_1bar_logistic_180": 0.6175,
            "A_1bar_rule": 0.6190,
            "B_1bar_close_to_close": 0.509,
        },
    }
    (store / "round4-two-bar.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {store / 'round4-two-bar.json'}", flush=True)
    write_md(store, out)
    append_trials(store / "round4-two-bar-trials.csv", a_trials + b_trials)
    print(
        f"DONE A 180={a_rep['logistic_180']['accuracy']:.4f} rule={a_rep['baselines']['rule_close_gt_typical']['acc']:.4f} "
        f"res={a_rep['residual_y_ne_rule']['acc']} | "
        f"B 180={b_rep['logistic_180']['accuracy']:.4f} rule={b_rep['baselines']['rule_close_gt_typical']['acc']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
