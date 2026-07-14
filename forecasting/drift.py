"""
Data-drift detection over the forecaster's residuals.

Stand-in for the "MLOps drift-detection system using Evidently AI or Great
Expectations" from the architecture brief: when the statistical
distribution of real-world input data shifts beyond a threshold, this
should trigger a retrain.

The primary signal is a level-shift z-test: the recent residual window's
mean is compared against the baseline residual window's mean, in units of
the baseline's standard error. A sudden, sustained demand shock (like a
supply-chain disruption) shows up as a large, persistent mean shift in the
residuals -- exactly what a z-test on the mean is built to catch, and far
less noisy at small sample sizes than a full-distribution test. A two-sample
Kolmogorov-Smirnov test (scipy) is also computed and reported alongside, as
a secondary check on distribution *shape* changes (e.g. increased
variance) that a pure mean-shift test would miss.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class DriftCheck:
    day: int
    z_score: float
    ks_statistic: float
    p_value: float
    is_drift: bool


def check_drift(baseline_residuals: np.ndarray, recent_residuals: np.ndarray, day: int, z_threshold: float = 3.0) -> DriftCheck:
    if len(recent_residuals) < 10 or len(baseline_residuals) < 10:
        return DriftCheck(day=day, z_score=0.0, ks_statistic=0.0, p_value=1.0, is_drift=False)

    baseline_mean = baseline_residuals.mean()
    baseline_std = baseline_residuals.std(ddof=1) or 1e-6
    recent_mean = recent_residuals.mean()
    standard_error = baseline_std / np.sqrt(len(recent_residuals))
    z = (recent_mean - baseline_mean) / standard_error

    ks_result = stats.ks_2samp(baseline_residuals, recent_residuals)

    is_drift = abs(z) > z_threshold
    return DriftCheck(
        day=day, z_score=float(z),
        ks_statistic=float(ks_result.statistic), p_value=float(ks_result.pvalue),
        is_drift=is_drift,
    )
