"""
Walk-forward simulation: train an initial forecaster, step through the
series one day at a time producing one-step-ahead forecasts, monitor
residuals for drift, and automatically retrain on a recent window whenever
drift is detected -- the "pipeline automatically triggers ... retrain the
model on the newest data window, preventing edge-case crashes" behavior
from the architecture brief, minus the Airflow/Lambda plumbing (simulated
here as a plain Python loop; see README for the production mapping).
"""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from forecasting.model import fit, predict, ForecastModel
from forecasting.drift import check_drift, DriftCheck

TRAIN_WINDOW = 180
BASELINE_RESIDUAL_WINDOW = 30
CHECK_INTERVAL = 5
RECENT_WINDOW = 14
RETRAIN_WINDOW = 120
DRIFT_Z_THRESHOLD = 4.0


@dataclass
class RetrainEvent:
    day: int
    z_score: float
    ks_statistic: float
    p_value: float


@dataclass
class PipelineOutput:
    days: np.ndarray
    dates: list
    actual: np.ndarray
    forecast: np.ndarray
    residual: np.ndarray
    drift_checks: list[DriftCheck]
    retrain_events: list[RetrainEvent]
    train_window: int


def run(sku_df: pd.DataFrame) -> PipelineOutput:
    sku_df = sku_df.sort_values("date").reset_index(drop=True)
    units = sku_df["units_sold"].to_numpy()
    dates = sku_df["date"].tolist()
    n = len(units)

    model: ForecastModel = fit(units[:TRAIN_WINDOW], day_offset=0)

    days_out, forecast_out, residual_out = [], [], []
    drift_checks: list[DriftCheck] = []
    retrain_events: list[RetrainEvent] = []

    baseline_residuals: list[float] = []
    days_since_last_retrain = 0

    for day in range(TRAIN_WINDOW, n):
        pred = predict(model, np.array([day]))[0]
        actual = units[day]
        resid = actual - pred

        days_out.append(day)
        forecast_out.append(pred)
        residual_out.append(resid)

        days_since_last_retrain += 1

        if len(baseline_residuals) < BASELINE_RESIDUAL_WINDOW:
            baseline_residuals.append(resid)
        elif days_since_last_retrain % CHECK_INTERVAL == 0:
            recent = residual_out[-RECENT_WINDOW:]
            check = check_drift(np.array(baseline_residuals), np.array(recent), day, z_threshold=DRIFT_Z_THRESHOLD)
            drift_checks.append(check)

            if check.is_drift:
                retrain_events.append(RetrainEvent(day=day, z_score=check.z_score, ks_statistic=check.ks_statistic, p_value=check.p_value))
                retrain_start = max(0, day + 1 - RETRAIN_WINDOW)
                model = fit(units[retrain_start:day + 1], day_offset=retrain_start)
                baseline_residuals = []
                days_since_last_retrain = 0

    return PipelineOutput(
        days=np.array(days_out),
        dates=[dates[d] for d in days_out],
        actual=units[TRAIN_WINDOW:],
        forecast=np.array(forecast_out),
        residual=np.array(residual_out),
        drift_checks=drift_checks,
        retrain_events=retrain_events,
        train_window=TRAIN_WINDOW,
    )
