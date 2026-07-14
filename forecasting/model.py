"""
Lightweight trend + seasonality forecaster.

Stand-in for the "classic statistical models (Prophet) combined with deep
learning architectures (Temporal Fusion Transformers)" from the
architecture brief. Prophet and TFT both pull in heavy dependencies
(cmdstanpy/pystan, PyTorch) that aren't worth the install cost for a demo --
this implements the same underlying idea (trend + weekly seasonality +
yearly seasonality, fit via regression) with only numpy/pandas/scikit-learn,
so the demo installs and runs anywhere in seconds. The interface
(`fit(df) -> ForecastModel`, `predict(model, n_days)`) is designed to be a
drop-in swap for a real Prophet/TFT model.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def _build_features(day_index: np.ndarray) -> np.ndarray:
    """day_index: integer days since series start. Builds trend + weekday
    one-hot + yearly Fourier features -- the same feature family Prophet
    uses internally, just fit with plain OLS instead of a Bayesian model."""
    n = len(day_index)
    weekday = day_index % 7
    weekday_onehot = np.eye(7)[weekday]

    fourier_terms = []
    for k in (1, 2):
        fourier_terms.append(np.sin(2 * np.pi * k * day_index / 365.25))
        fourier_terms.append(np.cos(2 * np.pi * k * day_index / 365.25))
    fourier = np.column_stack(fourier_terms)

    # Scale the trend term to roughly the same magnitude as the other
    # (bounded, O(1)) features. Without this, Ridge's L2 penalty -- which
    # assumes comparable coefficient scales -- under-penalizes the trend
    # term relative to the Fourier terms, and a short retrain window (where
    # trend and yearly Fourier terms are nearly collinear) can still produce
    # an unstable trend coefficient that blows up on extrapolation.
    trend = (day_index / 100.0).reshape(-1, 1).astype(float)

    return np.hstack([trend, weekday_onehot, fourier])


@dataclass
class ForecastModel:
    regressor: Ridge
    day_offset: int  # day_index=0 corresponds to this many days after the true series start
    trained_on_n_days: int


def fit(units_sold: np.ndarray, day_offset: int = 0) -> ForecastModel:
    day_index = np.arange(len(units_sold)) + day_offset
    X = _build_features(day_index)
    # Ridge (not plain OLS): on short retrain windows the trend term and the
    # yearly Fourier terms are nearly collinear, which lets OLS assign huge,
    # unstable coefficients that explode when extrapolating even a few weeks
    # past the training window. L2 regularization keeps coefficients bounded
    # and forecasts stable without changing the feature set.
    reg = Ridge(alpha=8.0)
    reg.fit(X, units_sold)
    return ForecastModel(regressor=reg, day_offset=day_offset, trained_on_n_days=len(units_sold))


def predict(model: ForecastModel, day_indices: np.ndarray) -> np.ndarray:
    X = _build_features(day_indices)
    preds = model.regressor.predict(X)
    return np.maximum(preds, 0)
