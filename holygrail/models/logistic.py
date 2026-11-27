"""Multinomial logistic regression (L2). Two separate heads via DualHead."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from holygrail.policy import encode_y


class LogisticPolicy:
    name = "logistic"

    def __init__(self, C: float = 1.0, max_iter: int = 400):
        self.C = C
        self.max_iter = max_iter
        self.model = LogisticRegression(
            solver="lbfgs",
            C=C,
            max_iter=max_iter,
            # sklearn >=1.5 uses multinomial for multinomial y automatically
        )

    def fit(self, X, y, sample_weight=None, X_val=None, y_val=None):
        Xn = np.asarray(X, dtype=np.float64)
        yn = encode_y(np.asarray(y))
        self.model.fit(Xn, yn, sample_weight=sample_weight)
        return self

    def predict_proba(self, X) -> np.ndarray:
        Xn = np.asarray(X, dtype=np.float64)
        proba = self.model.predict_proba(Xn)
        # Ensure columns are classes 0,1,2 even if a class is missing
        full = np.zeros((len(Xn), 3), dtype=np.float64)
        for j, cls in enumerate(self.model.classes_):
            full[:, int(cls)] = proba[:, j]
        return full

    def explain(self, X) -> dict:
        coef = self.model.coef_  # (n_classes, n_features)
        return {
            "kind": "logistic_coef",
            "coef": coef.tolist(),
            "intercept": self.model.intercept_.tolist(),
            "classes": self.model.classes_.tolist(),
        }
