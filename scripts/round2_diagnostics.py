#!/usr/bin/env python3
"""Round 2 diagnostics. Logistic 180, fold_60_20, same rows. No sealed exam."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler

from holygrail.columns import BLOCKS, FEATURE_COLUMNS, SCALER_COLUMNS
from holygrail.config import ATR_PERIOD, EMBARGO_MINUTES, SCALER_QUANTILE_RANGE
from holygrail.indicators import atr_wilder, sma
from holygrail.io import load_mt5_m1
from holygrail.models.logistic import LogisticPolicy
from holygrail.models.metrics import evaluate
from holygrail.policy import decode_y
from holygrail.timing import resample_ohlc

PURGE_MIN = 5


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


def purge_train(times: pd.DatetimeIndex, train_idx: np.ndarray, val_idx: np.ndarray) -> np.ndarray:
    val_start = times[val_idx[0]]
    cutoff = val_start - pd.Timedelta(minutes=PURGE_MIN + EMBARGO_MINUTES)
    return train_idx[times[train_idx] <= cutoff]


def fold_slices(n: int) -> list[dict]:
    sealed = int(n * 0.80)
    return [
        {
            "name": "fold_60_20",
            "train": (0, int(n * 0.60)),
            "val": (int(n * 0.60), sealed),
        }
    ]

CSV = "data/mt5_raw/EURUSD_M1_202101131130_202605270000_2020_2026.csv"
DATA = Path("data/processed_round2")
STORE = Path("/cursor/stores/self/docs")
N_REPEATS = 5
SEED = 0


def acc_rule(y: np.ndarray, pred: np.ndarray) -> dict:
    ok = np.isfinite(pred) & (pred != 0) & np.isfinite(y)
    n = int(ok.sum())
    if n == 0:
        return {"n": 0, "acc": None, "n_zero_or_nan": int((~ok).sum())}
    return {
        "n": n,
        "acc": float(np.mean(y[ok] == pred[ok])),
        "n_zero_or_nan": int((~ok).sum()),
    }


def pred_from_proba(proba: np.ndarray) -> np.ndarray:
    return decode_y(np.asarray(proba).argmax(axis=1))


def tf_group(name: str) -> str:
    if name.startswith("ltf_") or name.endswith("_5m") or name == "prev5m_dir":
        return "5m"
    if name.startswith("htf1_") or name.endswith("_30m"):
        return "30m"
    if name.startswith("htf2_") or name.endswith("_1h"):
        return "1h"
    return "shared"


def perm_cols(policy, X: pd.DataFrame, y: np.ndarray, n_repeats: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    Xn = X.to_numpy(dtype=np.float64)
    cols = list(X.columns)
    base = evaluate(y, policy.predict_proba(Xn))["accuracy"]
    rows = []
    for j, name in enumerate(cols):
        drops = []
        for _ in range(n_repeats):
            Xp = Xn.copy()
            rng.shuffle(Xp[:, j])
            drops.append(base - evaluate(y, policy.predict_proba(Xp))["accuracy"])
        rows.append(
            {
                "column": name,
                "acc_drop": float(np.mean(drops)),
                "acc_drop_std": float(np.std(drops)),
                "tf": tf_group(name),
            }
        )
    rows.sort(key=lambda r: -r["acc_drop"])
    return rows


def perm_groups(policy, X: pd.DataFrame, y: np.ndarray, groups: dict[str, list[str]], n_repeats: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    cols = list(X.columns)
    idx = {c: i for i, c in enumerate(cols)}
    Xn = X.to_numpy(dtype=np.float64)
    base = evaluate(y, policy.predict_proba(Xn))["accuracy"]
    out = []
    for gname, gcols in groups.items():
        js = [idx[c] for c in gcols if c in idx]
        if not js:
            continue
        drops = []
        for _ in range(n_repeats):
            Xp = Xn.copy()
            for j in js:
                rng.shuffle(Xp[:, j])
            drops.append(base - evaluate(y, policy.predict_proba(Xp))["accuracy"])
        out.append(
            {
                "group": gname,
                "n_cols": len(js),
                "acc_drop": float(np.mean(drops)),
                "acc_drop_std": float(np.std(drops)),
            }
        )
    out.sort(key=lambda r: -r["acc_drop"])
    return out


def main() -> None:
    print("load parquet", flush=True)
    X180 = pd.read_parquet(DATA / "X_180.parquet")[FEATURE_COLUMNS]
    yf = pd.read_parquet(DATA / "y.parquet")
    n = len(yf)
    times = pd.DatetimeIndex(yf["last_1m"])
    y = yf["y_5m_dir"].to_numpy()
    spec = fold_slices(n)[0]
    assert spec["name"] == "fold_60_20"
    train_idx = np.arange(*spec["train"])
    val_idx = np.arange(*spec["val"])
    train_idx = purge_train(times, train_idx, val_idx)
    print(f"fold_60_20 train {len(train_idx)} val {len(val_idx)} (sealed unused)", flush=True)

    print("fit logistic 180", flush=True)
    scaler = fit_scaler(X180.iloc[train_idx], SCALER_COLUMNS)
    Xtr = apply_scaler(X180.iloc[train_idx], scaler, SCALER_COLUMNS)
    Xva = apply_scaler(X180.iloc[val_idx], scaler, SCALER_COLUMNS)
    ytr = y[train_idx]
    yva = y[val_idx]
    pol = LogisticPolicy(C=1.0, max_iter=400)
    pol.fit(Xtr, ytr)
    proba_tr = pol.predict_proba(Xtr)
    proba_va = pol.predict_proba(Xva)
    met_tr = evaluate(ytr, proba_tr)
    met_va = evaluate(yva, proba_va)
    pred_va = pred_from_proba(proba_va)

    print("5m OHLC + ATR for rules / size / decomp / bar-shape", flush=True)
    m1 = load_mt5_m1(CSV)
    c5 = resample_ohlc(m1, "5min")
    tick = m1["tickvol"].resample("5min", label="left", closed="left").sum()
    c5["tickvol"] = tick.reindex(c5.index)
    c5["tp"] = (c5["high"] + c5["low"] + c5["close"]) / 3.0
    atr_s = pd.Series(
        atr_wilder(c5["high"].to_numpy(), c5["low"].to_numpy(), c5["close"].to_numpy(), ATR_PERIOD),
        index=c5.index,
    )
    bar = pd.DatetimeIndex(yf["open_time"])
    nxt = bar + pd.Timedelta(minutes=5)
    open5 = c5["open"].reindex(bar).to_numpy()
    high5 = c5["high"].reindex(bar).to_numpy()
    low5 = c5["low"].reindex(bar).to_numpy()
    close5 = c5["close"].reindex(bar).to_numpy()
    tp = c5["tp"].reindex(bar).to_numpy()
    tp_next = c5["tp"].reindex(nxt).to_numpy()
    atr = atr_s.reindex(bar).to_numpy()
    tickv = c5["tickvol"].reindex(bar).to_numpy()
    tick_sma = pd.Series(sma(c5["tickvol"].to_numpy(dtype=np.float64), 20), index=c5.index)
    tick_sma_v = tick_sma.reindex(bar).to_numpy()

    close_minus_tp = close5 - tp
    rule_a = np.sign(close_minus_tp)
    rule_b = X180["prev5m_dir"].to_numpy()
    rule_c = np.sign(X180["ltf_price_sma20_gap_5m"].to_numpy())
    # bin already exists; use it when gap is 0
    rule_c = X180["ltf_price_sma20_bin_5m"].to_numpy()

    va = val_idx
    rules = {
        "a_sign_close_minus_typical": acc_rule(yva, rule_a[va]),
        "b_prev5m_dir": acc_rule(yva, rule_b[va]),
        "c_sign_5m_sma20_gap": acc_rule(yva, rule_c[va]),
    }
    a_ok = np.isfinite(rule_a[va]) & (rule_a[va] != 0)
    agree = a_ok & (rule_a[va] == pred_va)
    disagree = a_ok & (rule_a[va] != pred_va)
    model_vs_a = {
        "n_agree": int(agree.sum()),
        "acc_where_a_agrees_model": float(np.mean(yva[agree] == pred_va[agree])) if agree.any() else None,
        "n_disagree": int(disagree.sum()),
        "acc_where_a_disagrees_model": float(np.mean(yva[disagree] == pred_va[disagree])) if disagree.any() else None,
        "frac_model_equals_a": float(np.mean(pred_va[a_ok] == rule_a[va][a_ok])) if a_ok.any() else None,
    }

    move = tp_next - tp
    known = close5 - tp  # close_this - TP_this, known at t
    unknown = tp_next - close5  # TP_next - close_this
    atr_ok = np.isfinite(atr[va]) & (atr[va] > 0) & np.isfinite(move[va])
    size = np.abs(move[va]) / atr[va]
    size[~atr_ok] = np.nan
    finite_size = np.isfinite(size)
    qs = np.quantile(size[finite_size], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    size_buckets = []
    for i in range(5):
        if i < 4:
            m = finite_size & (size >= qs[i]) & (size < qs[i + 1])
        else:
            m = finite_size & (size >= qs[i]) & (size <= qs[i + 1])
        size_buckets.append(
            {
                "bucket": i + 1,
                "lo": float(qs[i]),
                "hi": float(qs[i + 1]),
                "n": int(m.sum()),
                "acc": float(np.mean(yva[m] == pred_va[m])) if m.any() else None,
                "mean_abs_move_over_atr": float(np.mean(size[m])) if m.any() else None,
            }
        )

    k_va = known[va]
    u_va = unknown[va]
    m_va = move[va]
    both = np.isfinite(k_va) & np.isfinite(u_va) & np.isfinite(m_va) & (m_va != 0)
    sign_k = np.sign(k_va)
    sign_u = np.sign(u_va)
    decomp = {
        "identity": "TP_next-TP_this = (TP_next-close_this) + (close_this-TP_this)",
        "n": int(both.sum()),
        "frac_sign_known_equals_y": float(np.mean(sign_k[both] == yva[both])),
        "frac_sign_unknown_equals_y": float(np.mean(sign_u[both] == yva[both])),
        "frac_known_zero": float(np.mean(sign_k[both] == 0)),
        "frac_unknown_zero": float(np.mean(sign_u[both] == 0)),
        "mean_abs_known_over_abs_move": float(np.mean(np.abs(k_va[both]) / np.abs(m_va[both]))),
        "mean_abs_unknown_over_abs_move": float(np.mean(np.abs(u_va[both]) / np.abs(m_va[both]))),
        "frac_known_larger_magnitude": float(np.mean(np.abs(k_va[both]) > np.abs(u_va[both]))),
        "frac_same_sign_known_unknown": float(
            np.mean((sign_k[both] != 0) & (sign_u[both] != 0) & (sign_k[both] == sign_u[both]))
        ),
        "corr_move_vs_known": float(np.corrcoef(m_va[both], k_va[both])[0, 1]),
        "corr_move_vs_unknown": float(np.corrcoef(m_va[both], u_va[both])[0, 1]),
        "note": "known = close_this-TP_this is rule (a). If frac_sign_known_equals_y is high, the label is partly this-bar shape.",
    }

    print("learning curve 25/50/75/100%", flush=True)
    curve = []
    ntr = len(train_idx)
    for frac in (0.25, 0.50, 0.75, 1.00):
        k = max(1, int(ntr * frac))
        sub = train_idx[:k]
        sc = fit_scaler(X180.iloc[sub], SCALER_COLUMNS)
        Xtr_s = apply_scaler(X180.iloc[sub], sc, SCALER_COLUMNS)
        Xva_s = apply_scaler(X180.iloc[val_idx], sc, SCALER_COLUMNS)
        p = LogisticPolicy(C=1.0, max_iter=400)
        p.fit(Xtr_s, y[sub])
        mtr = evaluate(y[sub], p.predict_proba(Xtr_s))
        mva = evaluate(yva, p.predict_proba(Xva_s))
        curve.append(
            {
                "frac": frac,
                "n_train": int(k),
                "train_acc": mtr["accuracy"],
                "train_log_loss": mtr["log_loss"],
                "val_acc": mva["accuracy"],
                "val_log_loss": mva["log_loss"],
            }
        )
        print(f"  {frac:.0%} n={k} val_acc={mva['accuracy']:.4f}", flush=True)

    atr_va = atr[va]
    atr_ok2 = np.isfinite(atr_va) & (atr_va > 0)
    tert = np.quantile(atr_va[atr_ok2], [1 / 3, 2 / 3])
    atr_slices = []
    for name, m in (
        ("low", atr_ok2 & (atr_va <= tert[0])),
        ("mid", atr_ok2 & (atr_va > tert[0]) & (atr_va <= tert[1])),
        ("high", atr_ok2 & (atr_va > tert[1])),
    ):
        atr_slices.append(
            {
                "slice": name,
                "n": int(m.sum()),
                "acc": float(np.mean(yva[m] == pred_va[m])) if m.any() else None,
                "mean_atr": float(np.mean(atr_va[m])) if m.any() else None,
            }
        )

    hours = times[va].hour.to_numpy()
    hour_rows = []
    for h_i in range(24):
        m = hours == h_i
        if not m.any():
            continue
        hour_rows.append(
            {
                "hour": h_i,
                "n": int(m.sum()),
                "acc": float(np.mean(yva[m] == pred_va[m])),
            }
        )

    print("permutation importance val", flush=True)
    col_imp = perm_cols(pol, Xva, yva, N_REPEATS, SEED)
    tf_groups = {"5m": [], "30m": [], "1h": [], "shared": []}
    for c in FEATURE_COLUMNS:
        tf_groups[tf_group(c)].append(c)
    block_groups = {k: list(v) for k, v in BLOCKS.items()}
    print("group permutation TF + block", flush=True)
    tf_imp = perm_groups(pol, Xva, yva, tf_groups, N_REPEATS, SEED)
    block_imp = perm_groups(pol, Xva, yva, block_groups, N_REPEATS, SEED)

    print("bar-shape block vs 180", flush=True)
    rng = high5 - low5
    close_loc = np.full(n, np.nan)
    ok_rng = np.isfinite(rng) & (rng > 0)
    close_loc[ok_rng] = (close5[ok_rng] - low5[ok_rng]) / rng[ok_rng]
    upper = high5 - np.maximum(open5, close5)
    lower = np.minimum(open5, close5) - low5
    atr_pos = np.isfinite(atr) & (atr > 0)
    shape = pd.DataFrame(
        {
            "shape_range_atr": np.where(atr_pos, rng / atr, np.nan),
            "shape_close_loc": close_loc,
            "shape_close_minus_tp_atr": np.where(atr_pos, (close5 - tp) / atr, np.nan),
            "shape_upper_wick_atr": np.where(atr_pos, upper / atr, np.nan),
            "shape_lower_wick_atr": np.where(atr_pos, lower / atr, np.nan),
            "shape_rel_tickvol": np.where(
                np.isfinite(tickv) & np.isfinite(tick_sma_v) & (tick_sma_v > 0),
                tickv / tick_sma_v,
                np.nan,
            ),
        }
    )
    shape_cols = list(shape.columns)
    shape_scaler_cols = [c for c in shape_cols if c != "shape_close_loc"]
    shape_ok = np.isfinite(shape.to_numpy(dtype=np.float64)).all(axis=1)
    tr_s = train_idx[shape_ok[train_idx]]
    va_s = val_idx[shape_ok[val_idx]]
    X180s = pd.concat([X180, shape], axis=1)
    extra_scaler = SCALER_COLUMNS + shape_scaler_cols
    sc2 = fit_scaler(X180s.iloc[tr_s], extra_scaler)
    Xtr2 = apply_scaler(X180s.iloc[tr_s], sc2, extra_scaler)
    Xva2 = apply_scaler(X180s.iloc[va_s], sc2, extra_scaler)
    pol2 = LogisticPolicy(C=1.0, max_iter=400)
    pol2.fit(Xtr2, y[tr_s])
    met_shape = evaluate(y[va_s], pol2.predict_proba(Xva2))
    # 180-only on the same finite-shape val rows
    sc3 = fit_scaler(X180.iloc[tr_s], SCALER_COLUMNS)
    pol3 = LogisticPolicy(C=1.0, max_iter=400)
    pol3.fit(apply_scaler(X180.iloc[tr_s], sc3, SCALER_COLUMNS), y[tr_s])
    met_180_same = evaluate(
        y[va_s],
        pol3.predict_proba(apply_scaler(X180.iloc[va_s], sc3, SCALER_COLUMNS)),
    )

    tiny = size_buckets[0]["acc"]
    large = size_buckets[-1]["acc"]
    size_read = (
        "acc higher on large moves"
        if large is not None and tiny is not None and large > tiny + 0.02
        else (
            "acc higher on tiny moves"
            if large is not None and tiny is not None and tiny > large + 0.02
            else "not only-large or only-tiny; similar across size"
        )
    )

    report = {
        "fold": "fold_60_20",
        "features": "logistic_180",
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "sealed_scored": False,
        "train": {"accuracy": met_tr["accuracy"], "log_loss": met_tr["log_loss"]},
        "val": {"accuracy": met_va["accuracy"], "log_loss": met_va["log_loss"]},
        "trivial_rules_val": rules,
        "model_vs_rule_a": model_vs_a,
        "size_buckets_abs_tp_move_over_atr": size_buckets,
        "size_read": size_read,
        "decomp": decomp,
        "learning_curve": curve,
        "val_by_atr_tertile": atr_slices,
        "val_by_hour": hour_rows,
        "perm_top15": col_imp[:15],
        "perm_tf_groups": tf_imp,
        "perm_blocks": block_imp,
        "bar_shape": {
            "columns": shape_cols,
            "n_train": int(len(tr_s)),
            "n_val": int(len(va_s)),
            "logistic_180_same_rows": {
                "accuracy": met_180_same["accuracy"],
                "log_loss": met_180_same["log_loss"],
            },
            "logistic_180_plus_shape": {
                "accuracy": met_shape["accuracy"],
                "log_loss": met_shape["log_loss"],
            },
            "delta_acc": met_shape["accuracy"] - met_180_same["accuracy"],
        },
    }
    STORE.mkdir(parents=True, exist_ok=True)
    (STORE / "round2-diagnostics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("train", "val", "trivial_rules_val", "size_read", "bar_shape")}, indent=2), flush=True)
    print("wrote diagnostics json", flush=True)


if __name__ == "__main__":
    main()
