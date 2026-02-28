"""Exponential Smoothing model wrapper for training and prediction."""
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def train_exponential_smoothing(
    series: pd.Series,
    trend: str = "add",
    seasonal: str = "add",
    seasonal_periods: int = 7,
    **fit_kwargs: Any,
):  # returns HoltWintersResults
    """Fit Holt-Winters Exponential Smoothing on a univariate time series."""
    model = ExponentialSmoothing(
        series,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods,
    )
    return model.fit(**fit_kwargs)


def predict(fitted_model: Any, steps: int = 1) -> np.ndarray:
    """Forecast next `steps` periods."""
    return fitted_model.forecast(steps=steps)


def save_model(fitted_model: Any, path: Path) -> None:
    """Save fitted model to disk (pickle)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(fitted_model, f)


def load_model(path: Path) -> Any:
    """Load fitted model from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)
