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


def plot_model_comparison(results_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        results_df,
        x="Model",
        y="MAE",
        title="Lap Time Forecast: Model Comparison (lower MAE is better)",
        labels={"MAE": "Mean Absolute Error (s)"},
    )
    return fig


def plot_predicted_vs_actual(test_df: pd.DataFrame, y_test, y_pred, model_name: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=test_df["LapNumber"], y=y_test, mode="lines+markers", name="Actual")
    )
    fig.add_trace(
        go.Scatter(x=test_df["LapNumber"], y=y_pred, mode="lines+markers", name=f"Predicted ({model_name})")
    )
    fig.update_layout(
        title=f"Predicted vs Actual Lap Time - {model_name}",
        xaxis_title="Lap Number",
        yaxis_title="Lap Time (s)",
    )
    return fig


def plot_feature_importance(explain_df: pd.DataFrame, value_col: str, title: str) -> go.Figure:
    fig = px.bar(
        explain_df.sort_values(value_col),
        x=value_col,
        y="Feature",
        orientation="h",
        title=title,
    )
    return fig


def plot_degradation_forecast(observed_df: pd.DataFrame, forecast_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=observed_df["StintLap"], y=observed_df["LapTimeSeconds"],
            mode="lines+markers", name="Observed",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df["StintLap"], y=forecast_df["ForecastLapTimeSeconds"],
            mode="lines+markers", name="Forecast", line={"dash": "dash"},
        )
    )
    fig.update_layout(
        title="Stint Degradation: Observed vs Forecast",
        xaxis_title="Stint Lap",
        yaxis_title="Lap Time (s)",
    )
    return fig


def plot_pit_recommendations(recommendations_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar of laps remaining until the recommended pit/maintenance
    window, for stints with an actionable ("Pit now" / "Pit within N laps")
    recommendation only - stable/improving stints have nothing to show here."""
    actionable = recommendations_df[recommendations_df["RecommendedAction"].str.contains("Pit", na=False)].copy()
    if actionable.empty:
        return go.Figure().update_layout(title="No actionable pit recommendations right now")

    actionable["LapsUntilAction"] = (
        actionable["ProjectedCrossingStintLap"] - actionable["CurrentStintLap"]
    ).clip(lower=0)
    actionable["Label"] = actionable["Driver"] + " (stint " + actionable["Stint"].astype(int).astype(str) + ")"
    actionable = actionable.sort_values("LapsUntilAction")

    fig = px.bar(
        actionable,
        x="LapsUntilAction",
        y="Label",
        color="RiskCategory",
        orientation="h",
        title="Recommended Pit Window (laps until action, most urgent first)",
        labels={"LapsUntilAction": "Laps Until Recommended Pit", "Label": "Driver / Stint"},
    )
    return fig
