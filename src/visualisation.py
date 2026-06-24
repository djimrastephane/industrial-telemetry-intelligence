"""Plotly figures shared by the notebooks and the Streamlit dashboard."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_lap_times(laps_df: pd.DataFrame) -> go.Figure:
    fig = px.line(
        laps_df,
        x="LapNumber",
        y="LapTimeSeconds",
        color="Driver",
        markers=True,
        title="Lap Time by Lap Number",
        labels={"LapTimeSeconds": "Lap Time (s)", "LapNumber": "Lap"},
    )
    return fig


def plot_tyre_life_vs_laptime(laps_df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        laps_df,
        x="TyreLife",
        y="LapTimeSeconds",
        color="Compound",
        facet_col="Driver" if laps_df["Driver"].nunique() <= 3 else None,
        title="Tyre Life vs Lap Time",
        labels={"TyreLife": "Tyre Life (laps)", "LapTimeSeconds": "Lap Time (s)"},
    )
    return fig


def plot_speed_trace(telemetry_df: pd.DataFrame, driver: str) -> go.Figure:
    driver_telemetry = telemetry_df[telemetry_df["Driver"] == driver]
    fig = px.line(
        driver_telemetry,
        x="Distance",
        y="Speed",
        title=f"Speed Trace - Fastest Lap ({driver})",
        labels={"Distance": "Distance (m)", "Speed": "Speed (km/h)"},
    )
    return fig


def plot_throttle_brake_trace(telemetry_df: pd.DataFrame, driver: str) -> go.Figure:
    driver_telemetry = telemetry_df[telemetry_df["Driver"] == driver]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=driver_telemetry["Distance"],
            y=driver_telemetry["Throttle"],
            name="Throttle (%)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=driver_telemetry["Distance"],
            y=driver_telemetry["Brake"].astype(float) * 100,
            name="Brake (on/off x100)",
        )
    )
    fig.update_layout(
        title=f"Throttle and Brake Trace - Fastest Lap ({driver})",
        xaxis_title="Distance (m)",
        yaxis_title="Value",
    )
    return fig


def plot_driver_comparison(telemetry_df: pd.DataFrame, driver_a: str, driver_b: str) -> go.Figure:
    comparison = telemetry_df[telemetry_df["Driver"].isin([driver_a, driver_b])]
    fig = px.line(
        comparison,
        x="Distance",
        y="Speed",
        color="Driver",
        title=f"Speed Comparison: {driver_a} vs {driver_b}",
        labels={"Distance": "Distance (m)", "Speed": "Speed (km/h)"},
    )
    return fig
