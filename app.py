from __future__ import annotations
import os

import pandas as pd
import streamlit as st

from pipeline import run, TRAIN_WINDOW

st.set_page_config(page_title="Beacon Forecast | Adaptive Demand Forecasting", page_icon="◆", layout="wide")

# ---------------------------------------------------------------------------
# Theme: amber / dark navy, retail-warm
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --bc-amber: #F5A524;
        --bc-bg: #0E1420;
        --bc-panel: #171F30;
        --bc-border: #2A3448;
        --bc-grey: #9AA6BC;
    }
    .stApp { background-color: var(--bc-bg); color: #F3F5F9; }
    section[data-testid="stSidebar"] { background-color: var(--bc-panel); }
    h1, h2, h3 { font-family: -apple-system, 'Helvetica Neue', sans-serif; letter-spacing: -0.01em; }
    .bc-hero { font-size: 2.0rem; font-weight: 700; margin-bottom: 0; color: var(--bc-amber); }
    .bc-sub { color: var(--bc-grey); font-size: 0.95rem; margin-top: 0.2rem; }
    .bc-card {
        background: var(--bc-panel); border: 1px solid var(--bc-border); border-radius: 12px;
        padding: 14px 16px; margin-bottom: 10px;
    }
    div[data-testid="stMetricValue"] { color: var(--bc-amber); }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="bc-hero">Beacon Forecast</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="bc-sub">A demand forecaster that watches its own forecast errors, and automatically '
    'retrains itself the moment the real world stops matching what it learned.</div>',
    unsafe_allow_html=True,
)
st.write("")

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "sku_demand.csv")
df = pd.read_csv(DATA_PATH)

with st.sidebar:
    st.markdown("### ◆ Beacon Forecast")
    st.caption("Trend + seasonality forecaster with automated drift-triggered retraining.")
    sku = st.selectbox("SKU", options=sorted(df["sku"].unique()))
    st.caption(
        "SKU-004-ELECTRONICS has a synthetic supply-shock event injected at day 500 "
        "(~2025-05-15): a sudden ~42% demand drop with a slow, partial 70-day recovery."
    )

sub = df[df["sku"] == sku]
out = run(sub)

mae = abs(out.actual - out.forecast).mean()
m1, m2, m3, m4 = st.columns(4)
m1.metric("Forecast MAE (units/day)", f"{mae:.1f}")
m2.metric("Drift checks run", len(out.drift_checks))
m3.metric("Automated retrains", len(out.retrain_events))
m4.metric("Training window", f"{TRAIN_WINDOW} days")

tab1, tab2, tab3 = st.tabs(["Forecast vs actual", "Drift monitor", "Retrain log"])

chart_df = pd.DataFrame({
    "date": pd.to_datetime(out.dates),
    "actual": out.actual,
    "forecast": out.forecast,
})

with tab1:
    st.markdown("Actual daily demand vs. the pipeline's one-step-ahead forecast, walking forward day by day.")
    st.line_chart(chart_df.set_index("date")[["actual", "forecast"]])
    if len(out.retrain_events):
        retrain_dates = [out.dates[list(out.days).index(e.day)] for e in out.retrain_events]
        st.caption("Retrain events (model refit on the most recent 120 days): " + ", ".join(retrain_dates))

with tab2:
    st.markdown(
        "Every 5 days, the pipeline compares the mean forecast error over the last 14 days against a "
        "baseline error distribution established right after the last (re)training, using a z-test. "
        "A |z| beyond the threshold (4.0) signals the real world has drifted from what the model learned."
    )
    drift_df = pd.DataFrame({
        "date": [out.dates[list(out.days).index(c.day)] for c in out.drift_checks],
        "z_score": [c.z_score for c in out.drift_checks],
        "is_drift": [c.is_drift for c in out.drift_checks],
    })
    drift_df["date"] = pd.to_datetime(drift_df["date"])
    st.bar_chart(drift_df.set_index("date")["z_score"])
    st.caption("Bars beyond ±4.0 triggered an automatic retrain.")
    st.dataframe(
        pd.DataFrame([{
            "Day": c.day, "z-score": round(c.z_score, 2), "KS statistic": round(c.ks_statistic, 3),
            "Drift triggered": "Yes" if c.is_drift else "No",
        } for c in out.drift_checks]),
        use_container_width=True, hide_index=True,
    )

with tab3:
    if not out.retrain_events:
        st.info("No drift-triggered retrains for this SKU in the simulated window.")
    for e in out.retrain_events:
        st.markdown(
            f'<div class="bc-card"><b>Day {e.day}</b> ({out.dates[list(out.days).index(e.day)]}) &mdash; '
            f'z-score {e.z_score:+.2f}, KS statistic {e.ks_statistic:.3f} &mdash; model automatically retrained '
            f'on the most recent 120 days of data.</div>',
            unsafe_allow_html=True,
        )
    st.caption(
        "In production this trigger fires an Airflow DAG or AWS Lambda to retrain on the newest data "
        "window; here it's simulated as a direct in-process refit for the same behavior with no infra to run."
    )
