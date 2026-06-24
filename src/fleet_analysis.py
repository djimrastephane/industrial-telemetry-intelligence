"""Phase 4: multi-year fleet monitoring.

Same race (Bahrain GP), every year. Raw lap times aren't directly
comparable across years - cars and regulations change - so the benchmark
here is each driver's pace *relative to that year's fastest lap*, the same
way you'd benchmark a fleet of wells/pumps against the best performer in
the same operating period rather than against an absolute number that
drifts as conditions change year to year.
"""

import numpy as np
import pandas as pd


def year_fastest_lap(fleet_laps_df: pd.DataFrame) -> pd.DataFrame:
    """Single fastest lap time recorded each year, across the whole field."""
    return (
        fleet_laps_df.groupby("Year")["LapTimeSeconds"]
        .min()
        .reset_index(name="YearFastestLapSeconds")
    )


def driver_relative_pace_by_year(fleet_laps_df: pd.DataFrame) -> pd.DataFrame:
    """Each driver's average lap time per year, and how far behind (in %)
    that year's fastest lap they were - the cross-year benchmarking metric."""
    year_fastest = year_fastest_lap(fleet_laps_df)
    avg_pace = (
        fleet_laps_df.groupby(["Year", "Driver", "Team"])["LapTimeSeconds"]
        .mean()
        .reset_index(name="AvgLapTimeSeconds")
    )
    merged = avg_pace.merge(year_fastest, on="Year")
    merged["RelativePacePct"] = (
        (merged["AvgLapTimeSeconds"] - merged["YearFastestLapSeconds"])
        / merged["YearFastestLapSeconds"]
        * 100
    )
    return merged.sort_values(["Driver", "Year"]).reset_index(drop=True)


def team_relative_pace_by_year(fleet_laps_df: pd.DataFrame) -> pd.DataFrame:
    """Average of each team's drivers' relative pace, per year."""
    driver_pace = driver_relative_pace_by_year(fleet_laps_df)
    return (
        driver_pace.groupby(["Year", "Team"])["RelativePacePct"]
        .mean()
        .reset_index()
        .sort_values(["Team", "Year"])
        .reset_index(drop=True)
    )


def degradation_by_year(fleet_laps_df: pd.DataFrame) -> pd.DataFrame:
    """Average within-stint lap-time degradation slope per driver per year.

    Grouped by (Year, Driver, Stint) - not just (Driver, Stint) like the
    single-race degradation_analysis.py - since the same stint number
    recurs every year and must not be conflated across years.
    """
    df = fleet_laps_df.copy()
    df["StintLap"] = df.groupby(["Year", "Driver", "Stint"]).cumcount() + 1

    records = []
    for (year, driver, stint), group in df.groupby(["Year", "Driver", "Stint"]):
        group = group.dropna(subset=["LapTimeSeconds"])
        if len(group) < 2:
            continue
        slope, _ = np.polyfit(group["StintLap"], group["LapTimeSeconds"], 1)
        records.append({"Year": year, "Driver": driver, "Stint": stint, "DegradationSecondsPerLap": slope})

    stint_slopes = pd.DataFrame(records)
    if stint_slopes.empty:
        return stint_slopes

    return (
        stint_slopes.groupby(["Year", "Driver"])["DegradationSecondsPerLap"]
        .mean()
        .reset_index(name="AvgDegradationSecondsPerLap")
        .sort_values(["Driver", "Year"])
        .reset_index(drop=True)
    )


def year_over_year_shift(fleet_laps_df: pd.DataFrame, shift_threshold_pct: float = 0.3) -> pd.DataFrame:
    """Flag each driver's year-over-year change in relative pace.

    This is the long-horizon analogue of PositionTrendSlope (Phase 3): a
    one-off bad result is normal variation, but a sustained shift in
    relative pace between consecutive years this race is the kind of
    structural change worth flagging, the same way a multi-year decline in
    a fleet asset's output is treated differently from day-to-day noise.
    """
    driver_pace = driver_relative_pace_by_year(fleet_laps_df).sort_values(["Driver", "Year"])

    records = []
    for driver, group in driver_pace.groupby("Driver"):
        group = group.sort_values("Year").reset_index(drop=True)
        for i in range(1, len(group)):
            prev_year, curr_year = group.loc[i - 1, "Year"], group.loc[i, "Year"]
            delta = group.loc[i, "RelativePacePct"] - group.loc[i - 1, "RelativePacePct"]
            if delta > shift_threshold_pct:
                shift = "Declined"
            elif delta < -shift_threshold_pct:
                shift = "Improved"
            else:
                shift = "Stable"
            records.append(
                {
                    "Driver": driver,
                    "FromYear": prev_year,
                    "ToYear": curr_year,
                    "RelativePaceDeltaPct": delta,
                    "Shift": shift,
                }
            )

    return pd.DataFrame(records)


def fleet_benchmark_table(fleet_laps_df: pd.DataFrame) -> pd.DataFrame:
    """One row per driver per year: relative pace, degradation, and the
    year-over-year shift label - the core benchmarking dashboard table."""
    pace = driver_relative_pace_by_year(fleet_laps_df)
    degradation = degradation_by_year(fleet_laps_df)
    shifts = year_over_year_shift(fleet_laps_df)

    table = pace.merge(degradation, on=["Year", "Driver"], how="left")
    table = table.merge(
        shifts.rename(columns={"ToYear": "Year"})[["Driver", "Year", "Shift"]],
        on=["Driver", "Year"],
        how="left",
    )
    table["Shift"] = table["Shift"].fillna("N/A (first year)")
    return table.sort_values(["Driver", "Year"]).reset_index(drop=True)


if __name__ == "__main__":
    from src.data_cleaning import load_and_clean_fleet

    fleet_laps = load_and_clean_fleet()
    print("--- year fastest lap ---")
    print(year_fastest_lap(fleet_laps))
    print("\n--- benchmark table (head) ---")
    print(fleet_benchmark_table(fleet_laps).head(15))
