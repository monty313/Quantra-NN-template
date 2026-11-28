"""Shallow regularized LightGBM. Early stopping on time-ordered val."""

from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

from holygrail.config import N_CLASSES
from holygrail.policy import encode_y


class LightGBMPolicy:
    name = "lightgbm"

    def __init__(
        self,
        n_estimators: int = 400,
        max_depth: int = 4,
        learning_rate: float = 0.03,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_samples: int = 200,
        reg_lambda: float = 2.0,
        num_leaves: int = 16,
        subsample_freq: int = 1,
    ):
        self.params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            subsample_freq=subsample_freq,
            colsample_bytree=colsample_bytree,
            min_child_samples=min_child_samples,
            reg_lambda=reg_lambda,
            num_leaves=num_leaves,
            objective="binary",
            verbosity=-1,
            n_jobs=-1,
        )
        self.model = LGBMClassifier(**self.params)
        self.feature_names_ = None

    def fit(self, X, y, sample_weight=None, X_val=None, y_val=None):
        Xn = np.asarray(X, dtype=np.float32)
        yn = encode_y(np.asarray(y))
        if hasattr(X, "columns"):
            self.feature_names_ = list(X.columns)
        callbacks = [log_evaluation(0)]
        fit_kw = dict(sample_weight=sample_weight)
        if X_val is not None and y_val is not None:
            Xv = np.asarray(X_val, dtype=np.float32)
            yv = encode_y(np.asarray(y_val))
            callbacks.append(early_stopping(30, verbose=False))
            fit_kw["eval_X"] = Xv
            fit_kw["eval_y"] = yv
            fit_kw["eval_metric"] = "binary_logloss"
        fit_kw["callbacks"] = callbacks
        self.model.fit(Xn, yn, **fit_kw)
        return self

    def predict_proba(self, X) -> np.ndarray:
        Xn = np.asarray(X, dtype=np.float32)
        proba = self.model.predict_proba(Xn)
        full = np.zeros((len(Xn), N_CLASSES), dtype=np.float64)
        for j, cls in enumerate(self.model.classes_):
            full[:, int(cls)] = proba[:, j]
        return full

    def explain(self, X) -> dict:
        gain = self.model.feature_importances_
        names = self.feature_names_ or [f"f{i}" for i in range(len(gain))]
        return {
            "kind": "lightgbm_gain",
            "feature_names": names,
            "gain": gain.astype(float).tolist(),
        }
