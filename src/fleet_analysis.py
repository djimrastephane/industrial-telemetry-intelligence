"""Phase 4: multi-year fleet monitoring.

Same race (Bahrain GP), every year. Raw lap times aren't directly comparable
across years on their own - cars and regulations change - so most of these
benchmarks compare a driver against something measured in the *same* year:
their own teammate (same car, the most robust comparison), the field average,
or that year's single fastest lap (kept as a secondary, noise-sensitive
reference). This mirrors how you'd benchmark one well/pump against a sibling
asset on the same site rather than against an absolute number that drifts as
equipment generations change.
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


def year_field_average_lap(fleet_laps_df: pd.DataFrame) -> pd.DataFrame:
    """Average lap time across the entire field, per year."""
    return (
        fleet_laps_df.groupby("Year")["LapTimeSeconds"]
        .mean()
        .reset_index(name="YearFieldAverageLapSeconds")
    )


def driver_avg_lap_time_by_year(fleet_laps_df: pd.DataFrame) -> pd.DataFrame:
    """Each driver's average lap time per year - the shared building block
    for all of the relative-pace benchmarks below."""
    return (
        fleet_laps_df.groupby(["Year", "Driver", "Team"])["LapTimeSeconds"]
        .mean()
        .reset_index(name="AvgLapTimeSeconds")
    )


def driver_relative_pace_by_year(fleet_laps_df: pd.DataFrame) -> pd.DataFrame:
    """Each driver's pace vs. that year's single fastest lap (FastestLapGapPct).

    Kept as a secondary reference: a single lap is one noisy data point (an
    early red flag can leave an artificially slow "fastest" lap that year),
    so TeammateGapPct/FieldAverageGapPct are the more defensible benchmarks.
    """
    year_fastest = year_fastest_lap(fleet_laps_df)
    avg_pace = driver_avg_lap_time_by_year(fleet_laps_df)
    merged = avg_pace.merge(year_fastest, on="Year")
    merged["FastestLapGapPct"] = (
        (merged["AvgLapTimeSeconds"] - merged["YearFastestLapSeconds"])
        / merged["YearFastestLapSeconds"]
        * 100
    )
    return merged.sort_values(["Driver", "Year"]).reset_index(drop=True)


def field_average_gap_by_year(fleet_laps_df: pd.DataFrame) -> pd.DataFrame:
    """Each driver's pace vs. the full field average that year (FieldAverageGapPct)."""
    avg_pace = driver_avg_lap_time_by_year(fleet_laps_df)
    field_avg = year_field_average_lap(fleet_laps_df)
    merged = avg_pace.merge(field_avg, on="Year")
    merged["FieldAverageGapPct"] = (
        (merged["AvgLapTimeSeconds"] - merged["YearFieldAverageLapSeconds"])
        / merged["YearFieldAverageLapSeconds"]
        * 100
    )
    return merged.sort_values(["Driver", "Year"]).reset_index(drop=True)


def teammate_gap_by_year(fleet_laps_df: pd.DataFrame) -> pd.DataFrame:
    """Each driver's pace vs. their own teammate(s) that year (TeammateGapPct).

    The most robust cross-year benchmark here: same chassis and engine, so
    almost all of the regulation/car-development effect cancels out, leaving
    mostly driver (and luck/strategy) difference. NaN for a driver whose team
    had no other driver that year (e.g. a one-off seat).
    """
    driver_avg = driver_avg_lap_time_by_year(fleet_laps_df)

    records = []
    for (year, team), group in driver_avg.groupby(["Year", "Team"]):
        if len(group) < 2:
            continue
        for _, row in group.iterrows():
            teammate_avg = group.loc[group["Driver"] != row["Driver"], "AvgLapTimeSeconds"].mean()
            gap_pct = (row["AvgLapTimeSeconds"] - teammate_avg) / teammate_avg * 100
            records.append(
                {"Year": year, "Driver": row["Driver"], "Team": team, "TeammateGapPct": gap_pct}
            )

    return pd.DataFrame(records).sort_values(["Driver", "Year"]).reset_index(drop=True)


def team_relative_pace_by_year(fleet_laps_df: pd.DataFrame) -> pd.DataFrame:
    """Average of each team's drivers' fastest-lap-gap pace, per year."""
    driver_pace = driver_relative_pace_by_year(fleet_laps_df)
    return (
        driver_pace.groupby(["Year", "Team"])["FastestLapGapPct"]
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


def race_completion_rate_by_year(fleet_laps_df: pd.DataFrame) -> pd.DataFrame:
    """Laps completed as a % of the most laps anyone completed that year.

    A reliability *proxy*: FastF1's results data (which would carry an
    official DNF/retirement reason) isn't reliably available for every
    season, so this uses lap count instead of a true mechanical-retirement
    classification - see the README limitations.
    """
    laps_completed = (
        fleet_laps_df.groupby(["Year", "Driver"])["LapNumber"].max().reset_index(name="LapsCompleted")
    )
    max_laps_by_year = (
        laps_completed.groupby("Year")["LapsCompleted"].max().reset_index(name="MaxLapsCompletedThatYear")
    )
    merged = laps_completed.merge(max_laps_by_year, on="Year")
    merged["RaceCompletionRatePct"] = merged["LapsCompleted"] / merged["MaxLapsCompletedThatYear"] * 100
    return merged.sort_values(["Driver", "Year"]).reset_index(drop=True)


def lap_time_consistency_by_year(fleet_laps_df: pd.DataFrame) -> pd.DataFrame:
    """Lap time standard deviation per driver per year - lower means more
    consistent, the multi-year extension of the single-race consistency score."""
    return (
        fleet_laps_df.groupby(["Year", "Driver"])["LapTimeSeconds"]
        .std()
        .reset_index(name="LapTimeConsistencyStd")
        .sort_values(["Driver", "Year"])
        .reset_index(drop=True)
    )


def pit_stop_count_by_year(fleet_laps_df: pd.DataFrame) -> pd.DataFrame:
    """Number of pit stops per driver per year, from recorded pit-in laps."""
    return (
        fleet_laps_df[fleet_laps_df["PitInTime"].notna()]
        .groupby(["Year", "Driver"])
        .size()
        .reset_index(name="PitStopCount")
    )


def pit_stop_recovery_laps_by_year(fleet_laps_df: pd.DataFrame) -> pd.DataFrame:
    """Average laps needed to regain pre-stop running position after a pit
    stop - the operational-efficiency proxy for "how costly was the stop".

    Skips any stop where the driver never recovers to their pre-stop
    position before the data for that race ends (e.g. final stop of the
    race), rather than guessing a value.
    """
    pit_laps = fleet_laps_df.loc[fleet_laps_df["PitInTime"].notna(), ["Year", "Driver", "LapNumber"]]

    records = []
    for _, pit_lap in pit_laps.iterrows():
        year, driver, lap_number = pit_lap["Year"], pit_lap["Driver"], pit_lap["LapNumber"]
        driver_laps = fleet_laps_df[
            (fleet_laps_df["Year"] == year) & (fleet_laps_df["Driver"] == driver)
        ].sort_values("LapNumber")

        pre_stop = driver_laps.loc[driver_laps["LapNumber"] == lap_number, "Position"]
        if pre_stop.empty:
            continue
        pre_stop_position = pre_stop.iloc[0]

        after_stop = driver_laps[driver_laps["LapNumber"] > lap_number]
        recovered = after_stop[after_stop["Position"] <= pre_stop_position]
        if recovered.empty:
            continue
        laps_to_recover = recovered["LapNumber"].iloc[0] - lap_number
        records.append({"Year": year, "Driver": driver, "LapsToRecover": laps_to_recover})

    if not records:
        return pd.DataFrame(columns=["Year", "Driver", "AvgPitStopRecoveryLaps"])

    recovery_df = pd.DataFrame(records)
    return (
        recovery_df.groupby(["Year", "Driver"])["LapsToRecover"]
        .mean()
        .reset_index(name="AvgPitStopRecoveryLaps")
    )


def year_over_year_shift(
    metric_df: pd.DataFrame,
    value_col: str,
    shift_threshold_pct: float = 0.3,
) -> pd.DataFrame:
    """Flag each driver's year-over-year change in a given metric column.

    This is the long-horizon analogue of PositionTrendSlope (Phase 3): a
    one-off bad result is normal variation, but a sustained shift between
    consecutive years this race is the kind of structural change worth
    flagging, the same way a multi-year decline in a fleet asset's output is
    treated differently from day-to-day noise. Works on whichever pace
    metric is passed in (TeammateGapPct is the recommended default).
    """
    records = []
    for driver, group in metric_df.sort_values(["Driver", "Year"]).groupby("Driver"):
        group = group.reset_index(drop=True)
        for i in range(1, len(group)):
            prev_year, curr_year = group.loc[i - 1, "Year"], group.loc[i, "Year"]
            delta = group.loc[i, value_col] - group.loc[i - 1, value_col]
            if pd.isna(delta):
                continue
            if delta > shift_threshold_pct:
                shift = "Declined"
            elif delta < -shift_threshold_pct:
                shift = "Improved"
            else:
                shift = "Stable"
            records.append(
                {"Driver": driver, "FromYear": prev_year, "ToYear": curr_year, "Delta": delta, "Shift": shift}
            )

    return pd.DataFrame(records)


def fleet_benchmark_table(fleet_laps_df: pd.DataFrame) -> pd.DataFrame:
    """One row per driver per year: relative-pace benchmarks, degradation,
    reliability, consistency, pit-stop efficiency, and the year-over-year
    shift label - the core benchmarking dashboard table.

    TeammateGapPct is the lead benchmark (most robust to regulation
    changes); FieldAverageGapPct and FastestLapGapPct are kept as secondary
    context columns.
    """
    fastest = driver_relative_pace_by_year(fleet_laps_df)[["Year", "Driver", "Team", "FastestLapGapPct"]]
    field = field_average_gap_by_year(fleet_laps_df)[["Year", "Driver", "FieldAverageGapPct"]]
    teammate = teammate_gap_by_year(fleet_laps_df)
    degradation = degradation_by_year(fleet_laps_df)
    completion = race_completion_rate_by_year(fleet_laps_df)[["Year", "Driver", "RaceCompletionRatePct"]]
    consistency = lap_time_consistency_by_year(fleet_laps_df)
    pit_count = pit_stop_count_by_year(fleet_laps_df)
    recovery = pit_stop_recovery_laps_by_year(fleet_laps_df)

    table = fastest.merge(field, on=["Year", "Driver"], how="left")
    table = table.merge(teammate[["Year", "Driver", "TeammateGapPct"]], on=["Year", "Driver"], how="left")
    table = table.merge(degradation, on=["Year", "Driver"], how="left")
    table = table.merge(completion, on=["Year", "Driver"], how="left")
    table = table.merge(consistency, on=["Year", "Driver"], how="left")
    table = table.merge(pit_count, on=["Year", "Driver"], how="left")
    table = table.merge(recovery, on=["Year", "Driver"], how="left")
    table["PitStopCount"] = table["PitStopCount"].fillna(0).astype(int)

    if not teammate.empty:
        shifts = year_over_year_shift(teammate, "TeammateGapPct")
        if not shifts.empty:
            table = table.merge(
                shifts.rename(columns={"ToYear": "Year"})[["Driver", "Year", "Shift"]],
                on=["Driver", "Year"],
                how="left",
            )
    if "Shift" not in table.columns:
        table["Shift"] = None
    table["Shift"] = table["Shift"].fillna("N/A (no teammate data, or first year)")

    column_order = [
        "Year", "Driver", "Team",
        "TeammateGapPct", "FieldAverageGapPct", "FastestLapGapPct",
        "AvgDegradationSecondsPerLap", "RaceCompletionRatePct", "LapTimeConsistencyStd",
        "PitStopCount", "AvgPitStopRecoveryLaps", "Shift",
    ]
    table = table[[c for c in column_order if c in table.columns]]
    return table.sort_values(["Driver", "Year"]).reset_index(drop=True)


if __name__ == "__main__":
    from src.data_cleaning import load_and_clean_fleet

    fleet_laps = load_and_clean_fleet()
    print("--- year fastest lap ---")
    print(year_fastest_lap(fleet_laps))
    print("\n--- benchmark table (head) ---")
    print(fleet_benchmark_table(fleet_laps).head(15))
