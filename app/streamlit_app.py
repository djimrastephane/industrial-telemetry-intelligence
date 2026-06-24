"""Phase 1 dashboard: Industrial Asset Monitoring and Decision Support System.

FastF1 telemetry is used as a public surrogate for ESP / SCADA / production
sensor data. Run with: streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.anomaly_detection import get_anomaly_table
from src.baseline_models import (
    average_lap_time_by_driver,
    consistency_score,
    degradation_summary,
    fastest_lap_per_driver,
)
from src.data_cleaning import load_and_clean_all
from src.visualisation import (
    plot_driver_comparison,
    plot_lap_times,
    plot_speed_trace,
    plot_throttle_brake_trace,
    plot_tyre_life_vs_laptime,
)

st.set_page_config(page_title="Industrial Telemetry Intelligence", layout="wide")

st.title("Industrial Asset Monitoring and Decision Support System")
st.caption("Phase 1 - demonstrated on Formula 1 telemetry as a public surrogate for industrial sensor data")

st.markdown(
    """
    **This is not a sports analytics project.** It is an industrial telemetry analytics
    project, demonstrated on Formula 1 data because plant-level ESP, SCADA, and production
    well sensor data is rarely public. F1 telemetry is high-frequency, multi-sensor, and
    failure-relevant in the same way: it lets us prototype anomaly detection and degradation
    analysis end-to-end before connecting it to proprietary industrial data.
    """
)

st.divider()
st.subheader("Data Loading Status")

required_files = [config.LAPS_FILE, config.WEATHER_FILE, config.TELEMETRY_FILE]
missing = [f for f in required_files if not f.exists()]

if missing:
    st.error(
        "Processed data not found. Run `python -m src.data_ingestion` first to download "
        f"and cache the {config.SEASON_YEAR} {config.EVENT_NAME} ({config.SESSION_NAME}) session."
    )
    st.write("Missing files:", [str(f) for f in missing])
    st.stop()

laps_df, weather_df, telemetry_df = load_and_clean_all()
st.success(
    f"Loaded {config.SEASON_YEAR} {config.EVENT_NAME} - Session {config.SESSION_NAME}: "
    f"{laps_df['Driver'].nunique()} drivers, {len(laps_df)} laps, {len(telemetry_df)} telemetry samples."
)

st.divider()
st.subheader("Driver Selection")

available_drivers = sorted(laps_df["Driver"].unique())
default_drivers = [d for d in config.COMPARISON_DRIVERS if d in available_drivers] or available_drivers[:2]

col1, col2 = st.columns(2)
with col1:
    driver_a = st.selectbox("Driver A", available_drivers, index=available_drivers.index(default_drivers[0]))
with col2:
    remaining = [d for d in available_drivers if d != driver_a]
    default_b_index = remaining.index(default_drivers[1]) if len(default_drivers) > 1 and default_drivers[1] in remaining else 0
    driver_b = st.selectbox("Driver B", remaining, index=default_b_index)

st.divider()
st.subheader("Lap Time Trend")
st.plotly_chart(plot_lap_times(laps_df[laps_df["Driver"].isin([driver_a, driver_b])]), use_container_width=True)

st.subheader("Tyre Degradation")
st.plotly_chart(plot_tyre_life_vs_laptime(laps_df[laps_df["Driver"].isin([driver_a, driver_b])]), use_container_width=True)
st.dataframe(degradation_summary(laps_df[laps_df["Driver"].isin([driver_a, driver_b])]), use_container_width=True)

st.subheader("Telemetry Trace (Fastest Lap)")
telemetry_drivers = set(telemetry_df["Driver"].unique()) if not telemetry_df.empty else set()
if driver_a in telemetry_drivers:
    st.plotly_chart(plot_speed_trace(telemetry_df, driver_a), use_container_width=True)
    st.plotly_chart(plot_throttle_brake_trace(telemetry_df, driver_a), use_container_width=True)
if {driver_a, driver_b}.issubset(telemetry_drivers):
    st.plotly_chart(plot_driver_comparison(telemetry_df, driver_a, driver_b), use_container_width=True)
else:
    st.info(
        f"No cached fastest-lap telemetry for {sorted({driver_a, driver_b} - telemetry_drivers)} "
        "(likely no completed timed lap for this driver in the session). "
        "Re-run `python -m src.data_ingestion` if data/processed is stale."
    )

st.divider()
st.subheader("Baseline Metrics")
b1, b2 = st.columns(2)
with b1:
    st.write("Average lap time by driver")
    st.dataframe(average_lap_time_by_driver(laps_df), use_container_width=True)
    st.write("Fastest lap per driver")
    st.dataframe(fastest_lap_per_driver(laps_df), use_container_width=True)
with b2:
    st.write("Consistency score (lap time std dev, lower = more consistent)")
    st.dataframe(consistency_score(laps_df), use_container_width=True)

st.subheader("Anomaly Table (|z-score| > threshold)")
st.dataframe(get_anomaly_table(laps_df), use_container_width=True)

st.divider()
st.subheader("From Lap Telemetry to Asset Monitoring")
st.markdown(
    """
    - **Lap = duty cycle.** Each lap is one operating cycle of the asset (e.g. one pump cycle, one ESP run interval).
    - **Tyre stint = run-to-failure interval.** Lap time climbing across a stint is the same shape as vibration,
      temperature, or vibration trending upward as a pump or ESP wears toward failure.
    - **Lap time z-score = SCADA alarm rule.** Flagging a lap because it deviates from that driver's own normal
      range is the simplest possible anomaly detector, identical in form to a sensor threshold alarm.
    - **Speed/throttle/brake trace = multi-sensor stream.** These are the closest public analogue to high-frequency
      pressure, flow, and current readings from a wellhead or ESP controller.

    Phase 1 deliberately uses only descriptive baselines (averages, std dev, linear degradation slope, z-score) so
    that later phases (better anomaly detection, degradation modelling, forecasting, decision support, and an
    LLM explanation layer) have a clear, honest improvement to measure against.
    """
)
