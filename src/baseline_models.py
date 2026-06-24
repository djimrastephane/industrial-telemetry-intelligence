"""Dumb baselines computed before any modelling: per-driver summary stats.

These are the "do the simplest possible thing first" metrics that any
later anomaly or degradation model must beat to be worth its complexity.
"""

import pandas as pd

from src.degradation_analysis import degradation_per_stint


def average_lap_time_by_driver(laps_df: pd.DataFrame) -> pd.DataFrame:
    return (
        laps_df.groupby("Driver")["LapTimeSeconds"]
        .mean()
        .reset_index(name="AvgLapTimeSeconds")
        .sort_values("AvgLapTimeSeconds")
        .reset_index(drop=True)
    )


def fastest_lap_per_driver(laps_df: pd.DataFrame) -> pd.DataFrame:
    idx = laps_df.groupby("Driver")["LapTimeSeconds"].idxmin()
    return (
        laps_df.loc[idx, ["Driver", "LapNumber", "LapTimeSeconds", "Compound"]]
        .sort_values("LapTimeSeconds")
        .reset_index(drop=True)
    )


def consistency_score(laps_df: pd.DataFrame) -> pd.DataFrame:
    """Lower standard deviation of lap time means a more consistent driver/asset."""
    return (
        laps_df.groupby("Driver")["LapTimeSeconds"]
        .std()
        .reset_index(name="LapTimeStdSeconds")
        .sort_values("LapTimeStdSeconds")
        .reset_index(drop=True)
    )


def degradation_summary(laps_df: pd.DataFrame) -> pd.DataFrame:
    return degradation_per_stint(laps_df)


def build_all_baselines(laps_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "average_lap_time": average_lap_time_by_driver(laps_df),
        "fastest_lap": fastest_lap_per_driver(laps_df),
        "consistency": consistency_score(laps_df),
        "degradation": degradation_summary(laps_df),
    }


if __name__ == "__main__":
    from src.data_cleaning import load_and_clean_all

    laps, _, _ = load_and_clean_all()
    for name, table in build_all_baselines(laps).items():
        print(f"\n--- {name} ---")
        print(table.head())
