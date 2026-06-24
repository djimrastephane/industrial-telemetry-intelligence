"""Lap time degradation per tyre stint.

A tyre stint is treated as a single run-to-failure cycle: lap time should
climb as the tyre (the "component") wears, the same way vibration or
temperature trends upward as a pump or ESP degrades.
"""

import numpy as np
import pandas as pd

from src.feature_engineering import add_stint_lap_number


def degradation_per_stint(laps_df: pd.DataFrame) -> pd.DataFrame:
    """Linear slope (seconds/lap) of lap time within each driver's stint.

    A positive slope means lap time is getting worse as the stint progresses.
    """
    df = add_stint_lap_number(laps_df)

    records = []
    for (driver, stint), group in df.groupby(["Driver", "Stint"]):
        group = group.dropna(subset=["LapTimeSeconds"])
        if len(group) < 2:
            continue
        slope, intercept = np.polyfit(group["StintLap"], group["LapTimeSeconds"], 1)
        records.append(
            {
                "Driver": driver,
                "Stint": stint,
                "Compound": group["Compound"].iloc[0],
                "Laps": len(group),
                "DegradationSecondsPerLap": slope,
                "StartingLapTimeEstimate": intercept,
            }
        )

    return pd.DataFrame(records).sort_values(["Driver", "Stint"]).reset_index(drop=True)


if __name__ == "__main__":
    from src.data_cleaning import load_and_clean_all

    laps, _, _ = load_and_clean_all()
    print(degradation_per_stint(laps))
