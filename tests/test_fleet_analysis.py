import pandas as pd
import pytest

from src.fleet_analysis import (
    degradation_by_year,
    driver_relative_pace_by_year,
    fleet_benchmark_table,
    team_relative_pace_by_year,
    year_fastest_lap,
    year_over_year_shift,
)


@pytest.fixture
def sample_fleet_laps_df():
    rows = []
    # LEC: consistently the fastest driver both years, so they set each year's benchmark lap.
    for year, lap_times in [(2023, [90.0, 91.0]), (2024, [90.0, 91.0])]:
        for i, lt in enumerate(lap_times):
            rows.append(
                {
                    "Year": year, "Driver": "LEC", "Team": "Ferrari", "LapNumber": i + 1,
                    "Position": 1.0, "SpeedST": 300.0, "Stint": 1, "LapTimeSeconds": lt,
                }
            )
    # VER: close to LEC's pace in 2023, then a clear relative decline in 2024.
    for year, lap_times in [(2023, [91.0, 92.0]), (2024, [100.0, 101.0])]:
        for i, lt in enumerate(lap_times):
            rows.append(
                {
                    "Year": year, "Driver": "VER", "Team": "Red Bull", "LapNumber": i + 1,
                    "Position": 2.0, "SpeedST": 298.0, "Stint": 1, "LapTimeSeconds": lt,
                }
            )
    # HAM: stays roughly stable relative to the fastest lap each year.
    for year, lap_times in [(2023, [92.0, 93.0]), (2024, [92.0, 93.0])]:
        for i, lt in enumerate(lap_times):
            rows.append(
                {
                    "Year": year, "Driver": "HAM", "Team": "Mercedes", "LapNumber": i + 1,
                    "Position": 3.0, "SpeedST": 295.0, "Stint": 1, "LapTimeSeconds": lt,
                }
            )
    return pd.DataFrame(rows)


def test_year_fastest_lap(sample_fleet_laps_df):
    result = year_fastest_lap(sample_fleet_laps_df).set_index("Year")
    assert result.loc[2023, "YearFastestLapSeconds"] == 90.0
    assert result.loc[2024, "YearFastestLapSeconds"] == 90.0


def test_driver_relative_pace_by_year_zero_for_fastest(sample_fleet_laps_df):
    pace = driver_relative_pace_by_year(sample_fleet_laps_df)
    ver_2023 = pace[(pace["Driver"] == "VER") & (pace["Year"] == 2023)]
    # VER's laps (91, 92) average to 91.5, fastest lap that year (LEC's) is 90.0
    assert ver_2023["RelativePacePct"].iloc[0] == pytest.approx((91.5 - 90.0) / 90.0 * 100)


def test_team_relative_pace_by_year_matches_single_driver_team(sample_fleet_laps_df):
    team_pace = team_relative_pace_by_year(sample_fleet_laps_df)
    driver_pace = driver_relative_pace_by_year(sample_fleet_laps_df)
    red_bull_2023 = team_pace[(team_pace["Team"] == "Red Bull") & (team_pace["Year"] == 2023)]
    ver_2023 = driver_pace[(driver_pace["Driver"] == "VER") & (driver_pace["Year"] == 2023)]
    assert red_bull_2023["RelativePacePct"].iloc[0] == pytest.approx(ver_2023["RelativePacePct"].iloc[0])


def test_degradation_by_year_separates_years(sample_fleet_laps_df):
    result = degradation_by_year(sample_fleet_laps_df)
    ver_years = set(result[result["Driver"] == "VER"]["Year"])
    assert ver_years == {2023, 2024}


def test_year_over_year_shift_flags_decline(sample_fleet_laps_df):
    shifts = year_over_year_shift(sample_fleet_laps_df)
    ver_shift = shifts[(shifts["Driver"] == "VER") & (shifts["ToYear"] == 2024)]
    assert ver_shift["Shift"].iloc[0] == "Declined"


def test_fleet_benchmark_table_has_expected_columns(sample_fleet_laps_df):
    table = fleet_benchmark_table(sample_fleet_laps_df)
    expected = {"Year", "Driver", "Team", "RelativePacePct", "AvgDegradationSecondsPerLap", "Shift"}
    assert expected.issubset(table.columns)
    # First year for each driver has no prior year to compare against.
    ver_2023 = table[(table["Driver"] == "VER") & (table["Year"] == 2023)]
    assert ver_2023["Shift"].iloc[0] == "N/A (first year)"
