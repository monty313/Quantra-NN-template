from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_m1(n: int = 400, start: str = "2024-03-04 00:00:00", seed: int = 0, step: float = 1e-4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="min")
    noise = rng.normal(0.0, step, size=n)
    close = 1.2000 + np.cumsum(noise)
    high = close + np.abs(rng.normal(0, step, n))
    low = close - np.abs(rng.normal(0, step, n))
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    df = pd.DataFrame({"open": open_, "high": np.maximum(high, np.maximum(open_, close)),
                       "low": np.minimum(low, np.minimum(open_, close)),
                       "close": close}, index=idx)
    df.index.name = "open_time"
    df["decision_time"] = df.index + pd.Timedelta(minutes=1)
    return df


@pytest.fixture
def m1_short():
    return make_m1(400, seed=1)


@pytest.fixture(scope="session")
def m1_long():
    # 220 hours of 1m ≈ 13.2k rows — enough for 199 1h CCI bars
    return make_m1(220 * 60, start="2024-03-04 00:00:00", seed=2)
