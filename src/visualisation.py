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


def plot_position_trend(position_df: pd.DataFrame, entity_col: str) -> go.Figure:
    """Finishing position across the season. Lower is better, so the y-axis
    is inverted (1st place at the top), matching how standings are read."""
    fig = px.line(
        position_df,
        x="RoundNumber",
        y="FinishPosition",
        color=entity_col,
        markers=True,
        title=f"Season Finishing Position Trend by {entity_col}",
        labels={"RoundNumber": "Round", "FinishPosition": "Finishing Position"},
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def plot_speed_trap_trend(speed_df: pd.DataFrame) -> go.Figure:
    fig = px.line(
        speed_df,
        x="RoundNumber",
        y="AvgSpeedTrap",
        color="Driver",
        markers=True,
        title="Season Speed Trap Trend by Driver",
        labels={"RoundNumber": "Round", "AvgSpeedTrap": "Avg Speed Trap (km/h)"},
    )
    return fig


def plot_pace_gap_trend(pace_df: pd.DataFrame, entity_col: str, y_col: str, title: str, y_label: str) -> go.Figure:
    """Pace-gap trend across years for any of the fleet relative-pace
    benchmarks (TeammateGapPct, FieldAverageGapPct, FastestLapGapPct).
    Lower is better, so the y-axis is inverted to match how a gap is usually read."""
    fig = px.line(
        pace_df,
        x="Year",
        y=y_col,
        color=entity_col,
        markers=True,
        title=title,
        labels={"Year": "Year", y_col: y_label},
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def plot_degradation_by_year(degradation_df: pd.DataFrame) -> go.Figure:
    fig = px.line(
        degradation_df,
        x="Year",
        y="AvgDegradationSecondsPerLap",
        color="Driver",
        markers=True,
        title="Multi-Year Tyre Degradation Trend by Driver",
        labels={"Year": "Year", "AvgDegradationSecondsPerLap": "Avg Degradation (s/lap)"},
    )
    return fig
