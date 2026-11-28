#!/usr/bin/env python3
"""Round 3: logistic 180 / 128 on y_5m_nextclose_vs_typ. fold_60_20. No sealed."""

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
from holygrail.models.logistic import LogisticPolicy
from holygrail.models.metrics import evaluate
from holygrail.policy import decode_y
from holygrail.timing import resample_ohlc
from scripts.round2_diagnostics import (
    CSV,
    DATA,
    apply_scaler,
    fit_scaler,
    fold_slices,
    purge_train,
)

STORE = Path("/cursor/stores/bc-63d4488e-eaab-49e7-b822-99139192b098/docs")
HEAD = "y_5m_nextclose_vs_typ"
SEED = 0
SHUFFLE_BLOCK = 12  # 1h of 5m bars
SCALER_128 = [c for c in SCALER_COLUMNS if c in FEATURE_COLUMNS_V1_0]


def to_signed(y01: np.ndarray) -> np.ndarray:
    y01 = np.asarray(y01, dtype=np.float64)
    out = np.full(y01.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(y01)
    out[ok] = np.where(y01[ok] == 1.0, 1.0, -1.0)
    return out


def to_01_from_signed(pm: np.ndarray) -> np.ndarray:
    pm = np.asarray(pm)
    return np.where(pm > 0, 1, 0).astype(np.int8)


def block_shuffle(y: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    chunks = [y[i : i + block] for i in range(0, len(y), block)]
    order = rng.permutation(len(chunks))
    return np.concatenate([chunks[i] for i in order])


def equal_count_calibration(y01: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict:
    y01 = np.asarray(y01, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    order = np.argsort(p, kind="mergesort")
    y = y01[order]
    ps = p[order]
    n = len(p)
    edges = np.linspace(0, n, n_bins + 1, dtype=int)
    rows = []
    for i in range(n_bins):
        sl = slice(edges[i], edges[i + 1])
        ni = int(edges[i + 1] - edges[i])
        if ni == 0:
            rows.append(
                {
                    "bin": i + 1,
                    "n": 0,
                    "mean_pred": None,
                    "actual_rate": None,
                    "p_lo": None,
                    "p_hi": None,
                }
            )
            continue
        rows.append(
            {
                "bin": i + 1,
                "n": ni,
                "mean_pred": float(ps[sl].mean()),
                "actual_rate": float(y[sl].mean()),
                "p_lo": float(ps[sl].min()),
                "p_hi": float(ps[sl].max()),
            }
        )
    return {"n_bins": n_bins, "n": n, "bins": rows}


def acc_01(y: np.ndarray, pred: np.ndarray) -> dict:
    y = np.asarray(y)
    pred = np.asarray(pred)
    ok = np.isfinite(y) & np.isfinite(pred)
    n = int(ok.sum())
    if n == 0:
        return {"n": 0, "acc": None, "n_dropped": int(len(y) - n)}
    return {
        "n": n,
        "acc": float(np.mean(y[ok] == pred[ok])),
        "n_dropped": int(len(y) - n),
    }


def group_acc(y: np.ndarray, pred: np.ndarray, keys: np.ndarray) -> list[dict]:
    rows = []
    for k in sorted(np.unique(keys)):
        m = keys == k
        n = int(m.sum())
        rows.append(
            {
                "key": int(k) if np.issubdtype(type(k), np.integer) or isinstance(k, (int, np.integer)) else str(k),
                "n": n,
                "acc": float(np.mean(y[m] == pred[m])) if n else None,
            }
        )
    return rows


def pack_trial(*, features, model, n, acc, log_loss, ties, extra=None) -> dict:
    row = {
        "date": date.today().isoformat(),
        "label": HEAD,
        "features": features,
        "seed": SEED,
        "fold": "fold_60_20",
        "model": model,
        "n": None if n is None else int(n),
        "accuracy": acc,
        "log_loss": log_loss,
        "ties_dropped": ties,
    }
    if extra:
        row.update(extra)
    return row


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
    ]
    exists = path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    print(f"wrote {len(rows)} trials -> {path}", flush=True)


def pp(x) -> str:
    if x is None:
        return "n/a"
    return f"{100.0 * float(x):.2f}%"


def pp_delta(x) -> str:
    if x is None:
        return "n/a"
    return f"{100.0 * float(x):+.2f} pp"


def fit_logistic(Xtr, ytr_signed, Xva, cols, scaler_cols) -> tuple[LogisticPolicy, np.ndarray]:
    scaler = fit_scaler(Xtr, scaler_cols)
    pol = LogisticPolicy(C=1.0, max_iter=400)
    pol.fit(apply_scaler(Xtr[cols], scaler, scaler_cols), ytr_signed)
    proba = pol.predict_proba(apply_scaler(Xva[cols], scaler, scaler_cols))
    return pol, proba


def write_md(store: Path, d: dict) -> None:
    m180 = d["logistic_180"]
    m128 = d["logistic_128"]
    rule = d["baselines"]["rule_close_gt_typical"]
    maj = d["baselines"]["majority"]
    prev = d["baselines"]["prev5m_dir"]
    dis = d["disagreement"]
    lines = [
        "# Round 3 — y_5m_nextclose_vs_typ, logistic 180 / 128, fold_60_20",
        "",
        "New label only. 180-col builder unchanged. `y_5m_dir` kept as reference, not deleted.",
        "Sealed newest 20% unscored. No LightGBM/MLP. No new features. PR #1 untouched.",
        "",
        f"JSON: `{store / 'round3-label-typ.json'}`",
        "",
        "## Label",
        "",
        "After this 5m bar closes, every 5 minutes: **1** if close of the **next** 5m > typical of **this** 5m, **0** if lower.",
        "Typical = (H+L+C)/3 of this bar. Direction only. Ties (next close == this typical) dropped, counted.",
        "Labels are never features. X uses data up to this bar’s close only.",
        "",
        f"- fold: `{d['fold']}`",
        f"- train n after purge + finite new label: **{d['n_train']:,}**",
        f"- val n finite new label: **{d['n_val']:,}**",
        f"- ties dropped train / val: **{d['ties_train']:,}** / **{d['ties_val']:,}**",
        f"- sealed rows in this parquet: {d['n_sealed']:,} (not scored)",
        "",
        "## Val accuracy",
        "",
        "| Model / baseline | n | Acc |",
        "|---|---|---|",
        f"| **Logistic 180** (primary) | {m180['n']:,} | **{pp(m180['accuracy'])}** |",
        f"| Logistic 128 | {m128['n']:,} | {pp(m128['accuracy'])} |",
        f"| Rule: this close > this typical else 0 | {rule['n']:,} | **{pp(rule['acc'])}** |",
        f"| Majority (train mode = {int(maj['train_majority'])}) | {maj['n']:,} | {pp(maj['acc'])} |",
        f"| prev5m_dir (+1→1, −1→0; zeros dropped) | {prev['n']:,} | {pp(prev['acc'])} |",
        "",
        f"**KEY: logistic 180 minus rule = {d['model_minus_rule_pp']:+.2f} pp.**",
        "",
        f"Logistic 180 logloss {m180['log_loss']:.4f}. Logistic 128 logloss {m128['log_loss']:.4f}.",
        "",
        "## When the 180-col model disagrees with the rule",
        "",
        f"- Agree n={dis['n_agree']:,} acc={pp(dis['acc_agree'])}",
        f"- Disagree n={dis['n_disagree']:,} acc={pp(dis['acc_disagree'])}",
        f"- Model vs rule agreement rate: {pp(dis['agree_rate'])}",
        "",
        "## 10-bin calibration (equal-count, P(y=1), logistic 180)",
        "",
        "| Bin | n | Mean pred | Actual rate |",
        "|---|---|---|---|",
    ]
    for b in d["calibration_10bin"]["bins"]:
        if b["n"] == 0:
            lines.append(f"| {b['bin']} | 0 | — | — |")
        else:
            lines.append(
                f"| {b['bin']} | {b['n']:,} | {b['mean_pred']:.4f} | {b['actual_rate']:.4f} |"
            )
    lines += [
        "",
        "## Acc by hour (logistic 180, val last_1m hour)",
        "",
        "| Hour | n | Acc |",
        "|---|---|---|",
    ]
    for r in d["acc_by_hour"]:
        lines.append(f"| {r['key']:02d} | {r['n']:,} | {pp(r['acc'])} |")
    lines += [
        "",
        "## Acc by weekday (Mon=0 … Fri=4)",
        "",
        "| Weekday | n | Acc |",
        "|---|---|---|",
    ]
    wd = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for r in d["acc_by_weekday"]:
        name = wd[int(r["key"])] if str(r["key"]).isdigit() and 0 <= int(r["key"]) < 7 else r["key"]
        lines.append(f"| {name} | {r['n']:,} | {pp(r['acc'])} |")
    lines += [
        "",
        "## Top 10 logistic 180 coefficients by |coef| (scaled space)",
        "",
        "| Rank | Feature | Coef | Sign |",
        "|---|---|---|---|",
    ]
    for i, r in enumerate(d["top10_coefs"], 1):
        sign = "+" if r["coef"] > 0 else "−"
        lines.append(f"| {i} | `{r['feature']}` | {r['coef']:+.4f} | {sign} (y=1) |")
    lines += [
        "",
        "## Placebo (frozen logistic 180 preds vs shuffled val y)",
        "",
        f"- Shuffled-label (permute val y, seed {SEED}): n={d['placebo_shuffled_label']['n']:,} acc={pp(d['placebo_shuffled_label']['acc'])}",
        f"- Shuffled-time (1h blocks of {SHUFFLE_BLOCK} 5m bars): n={d['placebo_shuffled_time']['n']:,} acc={pp(d['placebo_shuffled_time']['acc'])}",
        "",
        "## Notes",
        "",
        "- 0/1 label mapped to −1/+1 only at `LogisticPolicy.fit` (`encode_y` rejects class 0).",
        "- P(y=1) = proba column for class +1.",
        "- Rule baseline uses this bar’s close vs this bar’s typical (known at t). Else 0.",
        "- `y_5m_dir` still in parquet. Not the train label.",
        f"- Agreement new-label vs mapped `y_5m_dir` on val (both finite): {pp(d['overlap_with_y_5m_dir']['acc'])} n={d['overlap_with_y_5m_dir']['n']:,}.",
        "",
    ]
    (store / "round3-label-typ.md").write_text("\n".join(lines))
    print(f"wrote {store / 'round3-label-typ.md'}", flush=True)


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

    print("compute y_5m_nextclose_vs_typ from 1m", flush=True)
    m1 = load_mt5_m1(args.csv)
    c5 = resample_ohlc(m1, "5min")
    c5["tp"] = (c5["high"] + c5["low"] + c5["close"]) / 3.0
    bar = pd.DatetimeIndex(yf["open_time"])
    nxt = bar + pd.Timedelta(minutes=5)
    close_this = c5["close"].reindex(bar).to_numpy(dtype=np.float64)
    tp_this = c5["tp"].reindex(bar).to_numpy(dtype=np.float64)
    close_next = c5["close"].reindex(nxt).to_numpy(dtype=np.float64)
    move = close_next - tp_this
    y_new = np.full(n, np.nan, dtype=np.float64)
    ok = np.isfinite(move) & (move != 0.0)
    y_new[ok] = np.where(move[ok] > 0.0, 1.0, 0.0)
    rule = np.where(np.isfinite(close_this) & np.isfinite(tp_this) & (close_this > tp_this), 1.0, 0.0)
    prev = yf["prev5m_dir"].to_numpy(dtype=np.float64)
    y_dir = yf["y_5m_dir"].to_numpy(dtype=np.float64)

    spec = fold_slices(n)[0]
    train_idx = purge_train(times, np.arange(*spec["train"]), np.arange(*spec["val"]))
    val_idx = np.arange(*spec["val"])
    # Never touch sealed.
    sealed_start = int(n * 0.80)
    assert val_idx[-1] < sealed_start

    tr_ok = np.isfinite(y_new[train_idx])
    va_ok = np.isfinite(y_new[val_idx])
    ties_train = int((~tr_ok).sum())
    ties_val = int((~va_ok).sum())
    train_use = train_idx[tr_ok]
    val_use = val_idx[va_ok]
    print(
        f"fold_60_20 train {len(train_use)} (ties drop {ties_train}) "
        f"val {len(val_use)} (ties drop {ties_val})",
        flush=True,
    )

    ytr = y_new[train_use]
    yva = y_new[val_use]
    ytr_s = to_signed(ytr)
    yva_s = to_signed(yva)
    Xtr = X180.iloc[train_use]
    Xva = X180.iloc[val_use]

    print("logistic 180", flush=True)
    pol180, proba180 = fit_logistic(Xtr, ytr_s, Xva, FEATURE_COLUMNS, SCALER_COLUMNS)
    met180 = evaluate(yva_s, proba180)
    pred180 = to_01_from_signed(decode_y(proba180.argmax(axis=1)))
    p_up = np.clip(proba180[:, 1], 1e-7, 1.0 - 1e-7)

    print("logistic 128", flush=True)
    _pol128, proba128 = fit_logistic(Xtr, ytr_s, Xva, FEATURE_COLUMNS_V1_0, SCALER_128)
    met128 = evaluate(yva_s, proba128)
    pred128 = to_01_from_signed(decode_y(proba128.argmax(axis=1)))

    rule_va = rule[val_use]
    maj_cls = 1.0 if float((ytr == 1.0).mean()) >= 0.5 else 0.0
    maj_pred = np.full(len(yva), maj_cls)
    prev_va = prev[val_use]
    prev_ok = np.isfinite(prev_va) & (prev_va != 0.0)
    prev_pred = np.where(prev_va > 0, 1.0, 0.0)

    rule_m = acc_01(yva, rule_va)
    maj_m = acc_01(yva, maj_pred)
    prev_m = acc_01(yva[prev_ok], prev_pred[prev_ok])

    agree = pred180 == rule_va
    n_ag = int(agree.sum())
    n_dg = int((~agree).sum())
    acc_ag = float(np.mean(pred180[agree] == yva[agree])) if n_ag else None
    acc_dg = float(np.mean(pred180[~agree] == yva[~agree])) if n_dg else None

    model_minus_rule = 100.0 * (met180["accuracy"] - rule_m["acc"])

    coef = np.asarray(pol180.model.coef_, dtype=np.float64)
    if coef.ndim == 2:
        coef = coef[0]
    mag = np.abs(coef)
    top_i = np.argsort(-mag)[:10]
    top10 = [
        {"feature": FEATURE_COLUMNS[int(i)], "coef": float(coef[int(i)]), "abs": float(mag[int(i)])}
        for i in top_i
    ]

    hours = times[val_use].hour.to_numpy()
    # pandas 2+/3 dayofweek on DatetimeIndex
    weekdays = times[val_use].day_of_week.to_numpy()

    rng = np.random.default_rng(SEED)
    y_perm = rng.permutation(yva)
    y_block = block_shuffle(yva, SHUFFLE_BLOCK, rng)
    placebo_label = acc_01(y_perm, pred180)
    placebo_time = acc_01(y_block, pred180)

    y_dir_va = y_dir[val_use]
    dir_ok = np.isfinite(y_dir_va)
    y_dir_01 = np.where(y_dir_va > 0, 1.0, 0.0)
    overlap = acc_01(yva[dir_ok], y_dir_01[dir_ok])

    report = {
        "label": HEAD,
        "fold": "fold_60_20",
        "seed": SEED,
        "n_parquet": n,
        "n_sealed": sealed_n,
        "n_train": int(len(train_use)),
        "n_val": int(len(val_use)),
        "ties_train": ties_train,
        "ties_val": ties_val,
        "sealed_scored": False,
        "y_5m_dir_kept": True,
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
        },
        "model_minus_rule_pp": float(model_minus_rule),
        "disagreement": {
            "n_agree": n_ag,
            "acc_agree": acc_ag,
            "n_disagree": n_dg,
            "acc_disagree": acc_dg,
            "agree_rate": float(n_ag / len(yva)) if len(yva) else None,
        },
        "calibration_10bin": equal_count_calibration(yva, p_up, 10),
        "acc_by_hour": group_acc(yva, pred180, hours),
        "acc_by_weekday": group_acc(yva, pred180, weekdays),
        "top10_coefs": top10,
        "placebo_shuffled_label": placebo_label,
        "placebo_shuffled_time": placebo_time,
        "overlap_with_y_5m_dir": overlap,
        "logistic_128_pred_acc": float(np.mean(pred128 == yva)),
    }

    (store / "round3-label-typ.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"wrote {store / 'round3-label-typ.json'}", flush=True)
    write_md(store, report)

    trials = [
        pack_trial(
            features="180",
            model="logistic_180",
            n=report["logistic_180"]["n"],
            acc=report["logistic_180"]["accuracy"],
            log_loss=report["logistic_180"]["log_loss"],
            ties=ties_val,
        ),
        pack_trial(
            features="128",
            model="logistic_128",
            n=report["logistic_128"]["n"],
            acc=report["logistic_128"]["accuracy"],
            log_loss=report["logistic_128"]["log_loss"],
            ties=ties_val,
        ),
        pack_trial(
            features="none",
            model="rule_close_gt_typical",
            n=rule_m["n"],
            acc=rule_m["acc"],
            log_loss=None,
            ties=0,
        ),
        pack_trial(
            features="none",
            model="majority",
            n=maj_m["n"],
            acc=maj_m["acc"],
            log_loss=None,
            ties=0,
        ),
        pack_trial(
            features="prev5m_dir",
            model="prev5m_dir",
            n=prev_m["n"],
            acc=prev_m["acc"],
            log_loss=None,
            ties=int((~prev_ok).sum()),
        ),
        pack_trial(
            features="180",
            model="placebo_shuffled_label",
            n=placebo_label["n"],
            acc=placebo_label["acc"],
            log_loss=None,
            ties=0,
        ),
        pack_trial(
            features="180",
            model="placebo_shuffled_time_1h",
            n=placebo_time["n"],
            acc=placebo_time["acc"],
            log_loss=None,
            ties=0,
        ),
    ]
    append_trials(store / "round3-label-typ-trials.csv", trials)
    print(
        f"DONE logistic180={met180['accuracy']:.4f} rule={rule_m['acc']:.4f} "
        f"delta={model_minus_rule:+.4f}pp",
        flush=True,
    )


if __name__ == "__main__":
    main()
