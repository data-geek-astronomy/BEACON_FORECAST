"""
Generates SYNTHETIC daily SKU-level demand data: trend, weekly seasonality,
yearly seasonality, noise -- and, for one SKU, an injected supply-shock
event partway through the series (a sudden demand drop that persists at a
new, lower baseline) to exercise the drift-detection pipeline. No real
Walmart, retailer, or sales data is used anywhere in this repo.
"""
import csv
import os
from datetime import date, timedelta

import numpy as np

np.random.seed(11)

N_DAYS = 730
START = date(2024, 1, 1)

SKUS = {
    "SKU-001-PANTRY": dict(base=180, trend=0.05, weekly_amp=25, yearly_amp=35, noise=12, shock_day=None),
    "SKU-002-BEVERAGE": dict(base=260, trend=0.08, weekly_amp=40, yearly_amp=60, noise=18, shock_day=None),
    "SKU-003-SEASONAL-DECOR": dict(base=90, trend=0.02, weekly_amp=15, yearly_amp=140, noise=10, shock_day=None),
    "SKU-004-ELECTRONICS": dict(base=140, trend=0.06, weekly_amp=20, yearly_amp=30, noise=14, shock_day=500),
}

SHOCK_DROP_PCT = 0.42
SHOCK_RECOVERY_DAYS = 70
SHOCK_PERMANENT_LOSS_PCT = 0.15  # the new baseline never fully recovers -- a supply chain re-shaping, not a blip


def generate_series(cfg, n_days):
    days = np.arange(n_days)
    trend = cfg["base"] + cfg["trend"] * days

    weekday = days % 7
    weekly = cfg["weekly_amp"] * np.sin(2 * np.pi * weekday / 7 + 1.5)

    yearly = cfg["yearly_amp"] * np.sin(2 * np.pi * days / 365.25 - 1.2)

    noise = np.random.normal(0, cfg["noise"], size=n_days)

    values = trend + weekly + yearly + noise

    if cfg["shock_day"] is not None:
        shock_day = cfg["shock_day"]
        multiplier = np.ones(n_days)
        for d in range(shock_day, n_days):
            days_since_shock = d - shock_day
            if days_since_shock < SHOCK_RECOVERY_DAYS:
                # sharp drop, gradual partial recovery over SHOCK_RECOVERY_DAYS
                recovery_frac = days_since_shock / SHOCK_RECOVERY_DAYS
                dip = SHOCK_DROP_PCT * (1 - recovery_frac) + SHOCK_PERMANENT_LOSS_PCT * recovery_frac
            else:
                dip = SHOCK_PERMANENT_LOSS_PCT
            multiplier[d] = 1 - dip
        values = values * multiplier

    return np.maximum(values, 0).round(1)


def main():
    out_path = os.path.join(os.path.dirname(__file__), "data", "sku_demand.csv")
    rows = []
    for sku, cfg in SKUS.items():
        series = generate_series(cfg, N_DAYS)
        for i, v in enumerate(series):
            d = START + timedelta(days=i)
            rows.append({"sku": sku, "date": d.isoformat(), "units_sold": v})

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sku", "date", "units_sold"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows ({len(SKUS)} SKUs x {N_DAYS} days) to {out_path}")
    print("SKU-004-ELECTRONICS has an injected supply-shock event at day 500 (~2025-05-15)")


if __name__ == "__main__":
    main()
