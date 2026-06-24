"""Derived columns used by the baseline metrics and dashboard.

In the industrial framing, StintLap is "operating hours since last
maintenance" and RollingLapTime is a smoothed sensor trend used to spot
slow drift before it becomes an alarm-worthy anomaly.
"""

import pandas as pd


def add_stint_lap_number(laps_df: pd.DataFrame) -> pd.DataFrame:
    """Lap count since the start of the current tyre stint (1-indexed)."""
    df = laps_df.copy()
    df["StintLap"] = df.groupby(["Driver", "Stint"]).cumcount() + 1
    return df


def add_rolling_lap_time(laps_df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Rolling mean lap time per driver, used to smooth lap-to-lap noise."""
    df = laps_df.copy()
    df["RollingLapTimeSeconds"] = (
        df.groupby("Driver")["LapTimeSeconds"]
        .transform(lambda s: s.rolling(window=window, min_periods=1).mean())
    )
    return df


def build_features(laps_df: pd.DataFrame) -> pd.DataFrame:
    df = add_stint_lap_number(laps_df)
    df = add_rolling_lap_time(df)
    return df


def add_forecast_features(laps_df: pd.DataFrame) -> pd.DataFrame:
    """Lag-only features for next-lap forecasting (Phase 5): unlike
    RollingLapTimeSeconds above (which includes the current lap and is fine
    for dashboard smoothing), these only use laps *before* the one being
    predicted, so a forecasting model can't peek at the answer.

    The first lap of each stint has no prior lap to lag from, so its
    PrevLapTimeSeconds/Rolling3PrevLapTimeSeconds are NaN by design.
    """
    df = add_stint_lap_number(laps_df)
    df = df.sort_values(["Driver", "Stint", "StintLap"])
    grouped = df.groupby(["Driver", "Stint"])["LapTimeSeconds"]
    df["PrevLapTimeSeconds"] = grouped.shift(1)
    df["Rolling3PrevLapTimeSeconds"] = grouped.transform(
        lambda s: s.shift(1).rolling(window=3, min_periods=1).mean()
    )
    return df


if __name__ == "__main__":
    from src.data_cleaning import load_and_clean_all

    laps, _, _ = load_and_clean_all()
    features = build_features(laps)
    print(features[["Driver", "LapNumber", "Stint", "StintLap", "RollingLapTimeSeconds"]].head(10))