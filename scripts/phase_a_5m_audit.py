#!/usr/bin/env python3
"""Phase A 5m direction audit. No new features. Does not replace y_5m / y_5m_dir."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from holygrail.columns import FEATURE_COLUMNS
from holygrail.io import load_mt5_m1
from holygrail.models.logistic import LogisticPolicy
from holygrail.timing import resample_ohlc

CSV = "data/mt5_raw/EURUSD_M1_202101131130_202605270000_2020_2026.csv"
OUT = Path("artifacts/phase_a_5m_audit.json")
SEED = 7
BLOCK = 12  # 12 consecutive 5m bars = 1 hour
PLACEBO_STOP = 0.52
SHUFFLE_STOP = 0.52


def last5(X, y):
    m = y["last_1m_of_5m"].to_numpy().astype(bool)
    return X.loc[m].reset_index(drop=True), y.loc[m].reset_index(drop=True)


def acc(y, pred) -> float:
    y = np.asarray(y)
    pred = np.asarray(pred)
    return float((y == pred).mean())


def consecutive_mask(open_time: pd.Series) -> np.ndarray:
    bar = pd.to_datetime(open_time).dt.floor("5min")
    delta = (bar - bar.shift(1)).dt.total_seconds().to_numpy() / 60.0
    return delta == 5.0


def fit_acc(Xtr, ytr, Xte, yte) -> dict:
    pol = LogisticPolicy()
    pol.fit(Xtr, np.asarray(ytr))
    p_up = pol.predict_proba(Xte)[:, 1]
    pred = np.where(p_up >= 0.5, 1.0, -1.0)
    yte = np.asarray(yte)
    return {
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "acc": acc(yte, pred),
        "pred_up_frac": float((pred == 1).mean()),
        "y_up_frac": float((yte == 1).mean()),
    }


def block_shuffle(y: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    chunks = [y[i : i + block] for i in range(0, len(y), block)]
    order = rng.permutation(len(chunks))
    return np.concatenate([chunks[i] for i in order])


def main() -> None:
    root = Path("data/processed")
    Xtr = pd.read_parquet(root / "X_train.parquet")[FEATURE_COLUMNS]
    ytr = pd.read_parquet(root / "y_train.parquet")
    Xte = pd.read_parquet(root / "X_test.parquet")[FEATURE_COLUMNS]
    yte = pd.read_parquet(root / "y_test.parquet")
    if "y_5m_dir" not in yte.columns:
        ytr["y_5m_dir"] = ytr["y_5m"]
        yte["y_5m_dir"] = yte["y_5m"]
        yva_path = root / "y_val.parquet"
        if yva_path.exists():
            yva = pd.read_parquet(yva_path)
            yva["y_5m_dir"] = yva["y_5m"]
            yva.to_parquet(yva_path)
        ytr.to_parquet(root / "y_train.parquet")
        yte.to_parquet(root / "y_test.parquet")
    assert np.array_equal(yte["y_5m"].to_numpy(), yte["y_5m_dir"].to_numpy(), equal_nan=True)

    Xtr5, ytr5 = last5(Xtr, ytr)
    Xte5, yte5 = last5(Xte, yte)
    y_te = yte5["y_5m_dir"].to_numpy()
    y_tr = ytr5["y_5m_dir"].to_numpy()
    n_te = len(y_te)
    report: dict = {
        "row_set": "last_1m_of_each_5m",
        "n_test": int(n_te),
        "n_train": int(len(y_tr)),
        "y_definition": (
            "structure_direction(m1, '5m'): "
            "completed 5m = resample_ohlc left-closed open-time; "
            "tp = (H+L+C)/3; y = sign(tp[next_open] - tp[current_open]); "
            "current_open = 1m_open.floor('5min'); next_open = current_open + 5min; "
            "missing bar or exact-zero move → NaN, dropped, never class 0. "
            "y_5m_dir is an alias of y_5m (same array)."
        ),
        "seed": SEED,
        "shuffle_block_5m_bars": BLOCK,
    }
    if n_te != 117598:
        report["row_set_note"] = f"NOT 117598; code/data give {n_te}"
    else:
        report["row_set_note"] = "same 117,598 test bars as prior 5m logistic run"

    # --- 1 persistence, 2 lag-1 ---
    consec = consecutive_mask(yte5["open_time"])
    prev = pd.Series(y_te).shift(1).to_numpy()
    ok = consec & np.isfinite(prev)
    persist_pred = prev[ok]
    persist_true = y_te[ok]
    continue_n = int((persist_pred == persist_true).sum())
    flip_n = int((persist_pred != persist_true).sum())
    n_ok = int(ok.sum())
    persist_acc = continue_n / n_ok
    rho = float(np.corrcoef(persist_pred, persist_true)[0, 1])
    report["persistence"] = {
        "n_consecutive": n_ok,
        "n_gap_skipped": int((~ok).sum()),
        "acc_next_eq_this": persist_acc,
        "continue_rate": continue_n / n_ok,
        "flip_rate": flip_n / n_ok,
        "continue_n": continue_n,
        "flip_n": flip_n,
    }
    report["lag1"] = {
        "n": n_ok,
        "pearson_y_t_vs_y_tminus1": rho,
        "equals_2continue_minus_1": 2 * persist_acc - 1,
    }

    # --- 3 close-to-close (do not replace y_5m_dir) ---
    print("resample 5m closes", flush=True)
    m1 = load_mt5_m1(CSV)
    c5 = resample_ohlc(m1, "5min")["close"]

    def close_dir(frame: pd.DataFrame) -> np.ndarray:
        bar = pd.to_datetime(frame["open_time"]).dt.floor("5min")
        nxt = bar + pd.Timedelta(minutes=5)
        a = c5.reindex(bar).to_numpy(dtype=np.float64)
        b = c5.reindex(nxt).to_numpy(dtype=np.float64)
        move = b - a
        y = np.full(len(frame), np.nan)
        good = np.isfinite(move) & (move != 0.0)
        y[good] = np.sign(move[good])
        return y

    ytr_c = close_dir(ytr5)
    yte_c = close_dir(yte5)
    tr_c = np.isfinite(ytr_c)
    te_c = np.isfinite(yte_c)
    print("fit close-to-close logistic", int(tr_c.sum()), int(te_c.sum()), flush=True)
    cc_log = fit_acc(Xtr5.loc[tr_c], ytr_c[tr_c], Xte5.loc[te_c], yte_c[te_c])
    # persistence of close-to-close on test
    prev_c = pd.Series(yte_c).shift(1).to_numpy()
    ok_c = consecutive_mask(yte5["open_time"]) & np.isfinite(yte_c) & np.isfinite(prev_c)
    cc_persist = acc(yte_c[ok_c], prev_c[ok_c]) if ok_c.sum() else None
    same_as_tp = float(np.mean(yte_c[te_c] == y_te[te_c])) if te_c.any() else None
    report["close_to_close"] = {
        "definition": "sign(close[next 5m] - close[this 5m]); ties dropped; not stored as y_5m_dir",
        "n_test_nonzero": int(te_c.sum()),
        "n_test_ties_or_missing": int((~te_c).sum()),
        "logistic": cc_log,
        "persistence_acc": cc_persist,
        "n_persist_consecutive": int(ok_c.sum()),
        "frac_same_as_y_5m_dir_on_cc_rows": same_as_tp,
    }

    # --- 4 placebo: y shifted one extra 5m bar (y[t] <- y[t+1]) ---
    def shifted(y: np.ndarray, open_time) -> np.ndarray:
        bar = pd.to_datetime(open_time).dt.floor("5min")
        nxt_delta = (bar.shift(-1) - bar).dt.total_seconds().to_numpy() / 60.0
        out = pd.Series(y).shift(-1).to_numpy()
        out = out.copy()
        out[nxt_delta != 5.0] = np.nan
        return out

    ytr_p = shifted(y_tr, ytr5["open_time"])
    yte_p = shifted(y_te, yte5["open_time"])
    tr_p = np.isfinite(ytr_p)
    te_p = np.isfinite(yte_p)
    print("fit placebo logistic", int(tr_p.sum()), int(te_p.sum()), flush=True)
    placebo = fit_acc(Xtr5.loc[tr_p], ytr_p[tr_p], Xte5.loc[te_p], yte_p[te_p])
    report["placebo"] = {
        "definition": "y_placebo[t] = y_5m_dir[t+1] when next 5m bar is exactly 5 minutes later (sign of B_{t+2} vs B_{t+1})",
        "logistic": placebo,
        "stop_if_acc_gt": PLACEBO_STOP,
        "fail": bool(placebo["acc"] > PLACEBO_STOP),
    }

    # --- 5 block-shuffle train labels, real test y ---
    rng = np.random.default_rng(SEED)
    ytr_s = block_shuffle(y_tr, BLOCK, rng)
    assert np.isfinite(ytr_s).all()
    print("fit shuffle logistic", flush=True)
    shuffle = fit_acc(Xtr5, ytr_s, Xte5, y_te)
    report["shuffle"] = {
        "definition": f"permute train y_5m_dir in blocks of {BLOCK} consecutive 5m rows; X order unchanged; test y real",
        "logistic": shuffle,
        "stop_if_acc_gt": SHUFFLE_STOP,
        "fail": bool(shuffle["acc"] > SHUFFLE_STOP),
    }

    # --- real logistic sanity ---
    print("fit real y_5m_dir logistic", flush=True)
    real = fit_acc(Xtr5, y_tr, Xte5, y_te)
    report["real_logistic_y_5m_dir"] = real

    report["go"] = (not report["placebo"]["fail"]) and (not report["shuffle"]["fail"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
