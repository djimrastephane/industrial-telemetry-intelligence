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


if __name__ == "__main__":
    from src.data_cleaning import load_and_clean_all

    laps, _, _ = load_and_clean_all()
    features = build_features(laps)
    print(features[["Driver", "LapNumber", "Stint", "StintLap", "RollingLapTimeSeconds"]].head(10))