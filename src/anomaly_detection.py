"""Simple z-score anomaly flagging on lap time.

This mirrors the simplest SCADA alarm rule: flag a sensor reading when it
deviates too many standard deviations from that asset's own normal range.
"""

import pandas as pd

from src import config


def add_lap_time_zscore(laps_df: pd.DataFrame) -> pd.DataFrame:
    """Per-driver z-score of lap time, so each driver is its own baseline."""
    df = laps_df.copy()
    grouped = df.groupby("Driver")["LapTimeSeconds"]
    mean = grouped.transform("mean")
    std = grouped.transform("std")
    df["LapTimeZScore"] = (df["LapTimeSeconds"] - mean) / std
    return df


def flag_anomalies(
    laps_df: pd.DataFrame,
    threshold: float = config.ANOMALY_ZSCORE_THRESHOLD,
) -> pd.DataFrame:
    """Add a boolean IsAnomaly column where |z-score| exceeds the threshold."""
    df = add_lap_time_zscore(laps_df)
    df["IsAnomaly"] = df["LapTimeZScore"].abs() > threshold
    return df


def get_anomaly_table(laps_df: pd.DataFrame) -> pd.DataFrame:
    """Anomalous laps only, for the dashboard table."""
    flagged = flag_anomalies(laps_df)
    columns = ["Driver", "LapNumber", "LapTimeSeconds", "Compound", "TyreLife", "LapTimeZScore"]
    return flagged.loc[flagged["IsAnomaly"], columns].sort_values(
        "LapTimeZScore", key=abs, ascending=False
    )


if __name__ == "__main__":
    from src.data_cleaning import load_and_clean_all

    laps, _, _ = load_and_clean_all()
    print(get_anomaly_table(laps))