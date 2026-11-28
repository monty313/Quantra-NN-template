#!/usr/bin/env python3
"""Phase D: one sealed test pass. Do not tune.

D1 reliability on VAL only → freeze high/low confidence bands.
D2 freeze logistic 180 (thin last-1m), score TEST once.
D3 placebo t+2 and 1h block-shuffle on the frozen model. Pass if acc <= 52%.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from holygrail.columns import FEATURE_COLUMNS, FEATURE_COLUMNS_V1_0
from holygrail.models.logistic import LogisticPolicy
from holygrail.models.majority import MajorityPolicy
from holygrail.models.metrics import evaluate
from holygrail.models.persistence import PersistencePolicy

HEAD = "y_5m_dir"
N_BINS = 10
PASS_ACC = 0.52
SHUFFLE_BLOCK = 12  # 12 consecutive 5m bars = 1h
SEED = 7


def last5(X, y):
    m = y["last_1m_of_5m"].to_numpy().astype(bool)
    return X.loc[m].reset_index(drop=True), y.loc[m].reset_index(drop=True)


def load_thin(root: Path, name: str):
    X = pd.read_parquet(root / f"X_{name}.parquet")[FEATURE_COLUMNS]
    y = pd.read_parquet(root / f"y_{name}.parquet")
    if HEAD not in y.columns:
        y[HEAD] = y["y_5m"]
    return last5(X, y)


def consecutive_mask(open_time) -> np.ndarray:
    bar = pd.to_datetime(open_time).dt.floor("5min")
    delta = (bar - bar.shift(1)).dt.total_seconds().to_numpy() / 60.0
    return delta == 5.0


def shifted_next_bar(y: np.ndarray, open_time) -> np.ndarray:
    """y_placebo[t] = y_5m_dir[t+1] when the next 5m bar is exactly 5 minutes later."""
    bar = pd.to_datetime(open_time).dt.floor("5min")
    nxt = (bar.shift(-1) - bar).dt.total_seconds().to_numpy() / 60.0
    out = pd.Series(y).shift(-1).to_numpy().astype(np.float64)
    out[nxt != 5.0] = np.nan
    return out


def block_shuffle(y: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    chunks = [y[i : i + block] for i in range(0, len(y), block)]
    order = rng.permutation(len(chunks))
    return np.concatenate([chunks[i] for i in order])


def p_up(proba: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(proba)[:, 1], 1e-7, 1.0 - 1e-7)


def quantile_edges(p: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    return np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))


def assign_bins(p: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin ids 0..n-1 using frozen edges. Last edge is inclusive."""
    b = np.digitize(p, edges[1:-1], right=False)
    b = np.clip(b, 0, len(edges) - 2)
    return b


def reliability_table(y: np.ndarray, p: np.ndarray, edges: np.ndarray) -> dict:
    y01 = (np.asarray(y) == 1).astype(np.float64)
    p = np.asarray(p, dtype=np.float64)
    bins = assign_bins(p, edges)
    n = len(p)
    rows = []
    ece = 0.0
    for i in range(len(edges) - 1):
        m = bins == i
        ni = int(m.sum())
        if ni == 0:
            rows.append(
                {
                    "bin": i + 1,
                    "p_lo": float(edges[i]),
                    "p_hi": float(edges[i + 1]),
                    "n": 0,
                    "mean_p": None,
                    "emp_plus": None,
                    "gap": None,
                    "hard_acc": None,
                }
            )
            continue
        mean_p = float(p[m].mean())
        emp = float(y01[m].mean())
        pred = (p[m] >= 0.5).astype(np.int8)
        hard = float((pred == y01[m]).mean())
        gap = abs(emp - mean_p)
        ece += (ni / n) * gap
        rows.append(
            {
                "bin": i + 1,
                "p_lo": float(edges[i]),
                "p_hi": float(edges[i + 1]),
                "n": ni,
                "mean_p": mean_p,
                "emp_plus": emp,
                "gap": gap,
                "hard_acc": hard,
            }
        )
    xs = np.array([r["mean_p"] for r in rows if r["n"] > 0], dtype=np.float64)
    ys = np.array([r["emp_plus"] for r in rows if r["n"] > 0], dtype=np.float64)
    ws = np.array([r["n"] for r in rows if r["n"] > 0], dtype=np.float64)
    slope, intercept = np.polyfit(xs, ys, 1, w=ws)
    return {
        "n_bins": len(edges) - 1,
        "n": n,
        "ece_quantile_reliability": float(ece),
        "bin_ols_slope": float(slope),
        "bin_ols_intercept": float(intercept),
        "bins": rows,
    }


def cox_slope_intercept(y: np.ndarray, p: np.ndarray) -> dict:
    """Cox calibration: logit P(y=+1) = a + b logit(p). Perfect is a=0, b=1."""
    y01 = (np.asarray(y) == 1).astype(np.int64)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    logit = np.log(p / (1.0 - p)).reshape(-1, 1)
    lr = LogisticRegression(solver="lbfgs", C=np.inf, max_iter=400)
    lr.fit(logit, y01)
    return {
        "intercept": float(lr.intercept_[0]),
        "slope": float(lr.coef_[0, 0]),
        "note": "logit P(y=+1) = intercept + slope * logit(p_up); perfect intercept 0 slope 1",
    }


def high_low_masks(p: np.ndarray, edges: np.ndarray) -> dict:
    """VAL-frozen equal-count bands.

    High: outer deciles (bin 1 ∪ bin 10). Low: middle two (bins 5–6).
    """
    bins = assign_bins(p, edges)
    n_bins = len(edges) - 1
    high = (bins == 0) | (bins == n_bins - 1)
    low = (bins == 4) | (bins == 5)
    return {"high": high, "low": low}


def slice_eval(y, proba, mask, name) -> dict:
    if mask.sum() == 0:
        return {"name": name, "n": 0}
    met = evaluate(np.asarray(y)[mask], proba[mask])
    return {"name": name, "n": int(mask.sum()), "frac": float(mask.mean()), **met}


def pack(model, features, split, met, extra=None):
    row = {
        "model": model,
        "features": features,
        "split": split,
        "log_loss": met["log_loss"],
        "accuracy": met["accuracy"],
        "accuracy_minus_50": met["accuracy_minus_50"],
        "ece_equal_width_confidence": met["ece"],
        "per_class": met["per_class"],
    }
    if extra:
        row.update(extra)
    return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed")
    p.add_argument("--out", default="artifacts/phase_d_test.json")
    args = p.parse_args()
    root = Path(args.data)
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))

    print("load train/val (D1) — test not touched yet", flush=True)
    Xtr, ytr = load_thin(root, "train")
    Xva, yva = load_thin(root, "val")
    y_tr = ytr[HEAD].to_numpy()
    y_va = yva[HEAD].to_numpy()
    assert np.array_equal(ytr["y_5m"].to_numpy(), y_tr)
    assert np.array_equal(yva["y_5m"].to_numpy(), y_va)

    cols180 = FEATURE_COLUMNS
    cols128 = FEATURE_COLUMNS_V1_0

    print("fit frozen models on train thin only", flush=True)
    maj = MajorityPolicy().fit(Xtr, y_tr)
    pers = PersistencePolicy().fit(Xtr, y_tr)
    log128 = LogisticPolicy().fit(Xtr[cols128], y_tr)
    log180 = LogisticPolicy().fit(Xtr[cols180], y_tr)

    # ----- D1 VAL reliability -----
    print("D1 val reliability", flush=True)
    proba_va = log180.predict_proba(Xva[cols180])
    p_va = p_up(proba_va)
    edges = quantile_edges(p_va, N_BINS)
    rel_va = reliability_table(y_va, p_va, edges)
    cox_va = cox_slope_intercept(y_va, p_va)
    bands_va = high_low_masks(p_va, edges)
    val_full = evaluate(y_va, proba_va)
    d1_high = slice_eval(y_va, proba_va, bands_va["high"], "val_high_outer_deciles")
    d1_low = slice_eval(y_va, proba_va, bands_va["low"], "val_low_middle_bins")
    frozen_bands = {
        "rule": (
            "10 equal-count bins of P(+1) on VAL. "
            "High-confidence = bin 1 ∪ bin 10 (outer deciles). "
            "Low-confidence = bins 5–6. Edges frozen; not recomputed on test."
        ),
        "n_bins": N_BINS,
        "p_up_edges": edges.tolist(),
        "high_p_lo": float(edges[0]),
        "high_p_q10": float(edges[1]),
        "high_p_q90": float(edges[9]),
        "high_p_hi": float(edges[10]),
        "low_p_q40": float(edges[4]),
        "low_p_q60": float(edges[6]),
    }
    print(
        f"D1 val ECE_q={rel_va['ece_quantile_reliability']:.4f} "
        f"cox_slope={cox_va['slope']:.3f} cox_int={cox_va['intercept']:.3f} "
        f"high_n={d1_high['n']} high_acc={d1_high.get('accuracy')} "
        f"low_n={d1_low['n']} low_acc={d1_low.get('accuracy')}",
        flush=True,
    )

    # ----- D2 TEST once -----
    print("D2 load test once", flush=True)
    Xte, yte = load_thin(root, "test")
    y_te = yte[HEAD].to_numpy()
    assert np.array_equal(yte["y_5m"].to_numpy(), y_te)

    proba_te = log180.predict_proba(Xte[cols180])
    p_te = p_up(proba_te)
    met_te = evaluate(y_te, proba_te)
    met_te_128 = evaluate(y_te, log128.predict_proba(Xte[cols128]))
    met_maj = evaluate(y_te, maj.predict_proba(Xte))
    met_pers = evaluate(y_te, pers.predict_proba(Xte))

    bands_te = high_low_masks(p_te, edges)
    d2_high = slice_eval(y_te, proba_te, bands_te["high"], "test_high_outer_deciles_val_edges")
    d2_low = slice_eval(y_te, proba_te, bands_te["low"], "test_low_middle_bins_val_edges")
    rel_te = reliability_table(y_te, p_te, edges)  # same VAL edges, not retuned
    cox_te = cox_slope_intercept(y_te, p_te)

    print(
        f"D2 test logistic180 acc={met_te['accuracy']:.4f} ll={met_te['log_loss']:.4f} "
        f"ece={met_te['ece']:.4f} n={len(y_te)}",
        flush=True,
    )
    print(
        f"D2 test logistic128 acc={met_te_128['accuracy']:.4f} ll={met_te_128['log_loss']:.4f}",
        flush=True,
    )
    print(
        f"D2 majority acc={met_maj['accuracy']:.4f} persistence acc={met_pers['accuracy']:.4f}",
        flush=True,
    )
    print(
        f"D2 high n={d2_high['n']} acc={d2_high.get('accuracy')} "
        f"low n={d2_low['n']} acc={d2_low.get('accuracy')}",
        flush=True,
    )

    # ----- D3 leak controls on frozen 180 -----
    print("D3 placebo t+2 and 1h block-shuffle (frozen)", flush=True)
    y_te_p = shifted_next_bar(y_te, yte["open_time"])
    ok_p = np.isfinite(y_te_p)
    placebo_acc = float("nan")
    placebo = {"n": int(ok_p.sum()), "definition": "frozen P(X_t) vs y_5m_dir[t+1] (B_{t+2} vs B_{t+1})"}
    if ok_p.any():
        met_p = evaluate(y_te_p[ok_p], proba_te[ok_p])
        placebo_acc = met_p["accuracy"]
        placebo.update(pack("logistic", "180", "test_placebo_tplus2", met_p))
        placebo["pass"] = bool(placebo_acc <= PASS_ACC)
        placebo["fail"] = not placebo["pass"]
    print(f"D3 placebo acc={placebo_acc:.4f} pass<={PASS_ACC} {placebo.get('pass')}", flush=True)

    rng = np.random.default_rng(SEED)
    y_te_s = block_shuffle(y_te, SHUFFLE_BLOCK, rng)
    met_s = evaluate(y_te_s, proba_te)
    shuffle = pack("logistic", "180", "test_y_1h_block_shuffled", met_s)
    shuffle["definition"] = (
        f"frozen P(X_test) vs test y_5m_dir permuted in blocks of {SHUFFLE_BLOCK} consecutive 5m rows"
    )
    shuffle["pass"] = bool(met_s["accuracy"] <= PASS_ACC)
    shuffle["fail"] = not shuffle["pass"]
    print(f"D3 shuffle acc={met_s['accuracy']:.4f} pass<={PASS_ACC} {shuffle['pass']}", flush=True)

    trials = [
        {"id": "D1", "step": "val_reliability_logistic180", **pack("logistic", "180", "val", val_full),
         "ece_quantile_reliability": rel_va["ece_quantile_reliability"],
         "cox_slope": cox_va["slope"], "cox_intercept": cox_va["intercept"],
         "bin_ols_slope": rel_va["bin_ols_slope"], "bin_ols_intercept": rel_va["bin_ols_intercept"]},
        {"id": "D1h", "step": "val_high_confidence", **{k: d1_high.get(k) for k in
            ("n", "frac", "log_loss", "accuracy", "accuracy_minus_50", "ece")}},
        {"id": "D1l", "step": "val_low_confidence", **{k: d1_low.get(k) for k in
            ("n", "frac", "log_loss", "accuracy", "accuracy_minus_50", "ece")}},
        {"id": "D2a", "step": "test_majority", **pack("majority", "none", "test", met_maj)},
        {"id": "D2b", "step": "test_persistence", **pack("persistence", "prev5m_dir", "test", met_pers),
         "p_continue_train": float(pers.p_continue_)},
        {"id": "D2c", "step": "test_logistic128", **pack("logistic", "128", "test", met_te_128)},
        {"id": "D2d", "step": "test_logistic180_winner", **pack("logistic", "180", "test", met_te)},
        {"id": "D2h", "step": "test_high_confidence_val_bands", **{k: d2_high.get(k) for k in
            ("n", "frac", "log_loss", "accuracy", "accuracy_minus_50", "ece")}},
        {"id": "D2l", "step": "test_low_confidence_val_bands", **{k: d2_low.get(k) for k in
            ("n", "frac", "log_loss", "accuracy", "accuracy_minus_50", "ece")}},
        {"id": "D3a", "step": "placebo_tplus2_frozen", "accuracy": placebo_acc,
         "pass_le_52": placebo.get("pass"), "n": placebo.get("n")},
        {"id": "D3b", "step": "block_shuffle_1h_frozen", "accuracy": met_s["accuracy"],
         "pass_le_52": shuffle["pass"], "n": int(len(y_te))},
    ]

    report = {
        "phase": "D",
        "note": "ONE sealed test pass. No tuning. y_5m_dir unchanged.",
        "head": HEAD,
        "winner": "logistic_180_thin_last1m",
        "feature_version": meta.get("feature_version"),
        "cutoff": meta.get("cutoff"),
        "inner_cutoff": meta.get("inner_cutoff"),
        "n_train_last1m": int(len(Xtr)),
        "n_val_last1m": int(len(Xva)),
        "n_test_last1m": int(len(Xte)),
        "frozen_bands": frozen_bands,
        "D1_val": {
            "evaluate": val_full,
            "reliability": rel_va,
            "cox": cox_va,
            "high_confidence": d1_high,
            "low_confidence": d1_low,
        },
        "D2_test": {
            "majority": met_maj,
            "persistence": met_pers,
            "logistic_128": met_te_128,
            "logistic_180": met_te,
            "high_confidence_val_bands": d2_high,
            "low_confidence_val_bands": d2_low,
            "reliability_val_edges": rel_te,
            "cox": cox_te,
        },
        "D3": {
            "pass_if_acc_le": PASS_ACC,
            "placebo_frozen": placebo,
            "shuffle_frozen": shuffle,
            "go": bool(placebo.get("pass") and shuffle["pass"]),
        },
        "trials": trials,
        "y_5m_dir_unchanged": True,
        "tuned_on_test": False,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("wrote", args.out, "D3_go", report["D3"]["go"], flush=True)


if __name__ == "__main__":
    main()
