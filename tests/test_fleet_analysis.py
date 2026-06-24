import pandas as pd
import pytest

from src.fleet_analysis import (
    degradation_by_year,
    driver_relative_pace_by_year,
    field_average_gap_by_year,
    fleet_benchmark_table,
    lap_time_consistency_by_year,
    pit_stop_count_by_year,
    pit_stop_recovery_laps_by_year,
    race_completion_rate_by_year,
    team_relative_pace_by_year,
    teammate_gap_by_year,
    year_fastest_lap,
    year_over_year_shift,
)

NaT = pd.NaT


def _row(year, driver, team, lap_number, position, lap_time, pit_in=NaT, stint=1):
    return {
        "Year": year, "Driver": driver, "Team": team, "LapNumber": lap_number,
        "Position": position, "SpeedST": 300.0, "Compound": "MEDIUM", "TyreLife": lap_number,
        "Stint": stint, "PitInTime": pit_in, "PitOutTime": NaT, "LapTimeSeconds": lap_time,
    }


@pytest.fixture
def sample_fleet_laps_df():
    rows = []

    # LEC (Ferrari, no teammate in this fixture): consistently fastest both years.
    for year in (2023, 2024):
        for lap in range(1, 6):
            rows.append(_row(year, "LEC", "Ferrari", lap, 1.0, 90.0))

    # VER (Red Bull): close to teammate PER in 2023, then clearly slower in 2024 (decline).
    for lap in range(1, 6):
        rows.append(_row(2023, "VER", "Red Bull", lap, 2.0, 91.0))
    for lap in range(1, 6):
        rows.append(_row(2024, "VER", "Red Bull", lap, 2.0, 99.0))

    # PER (Red Bull): retires after lap 4 in 2023 (DNF proxy), with a pit stop on lap 2
    # whose position trace recovers by lap 4 (laps_to_recover == 2). Stays on-pace in 2024.
    per_2023_positions = {1: 3.0, 2: 6.0, 3: 7.0, 4: 6.0}
    for lap in range(1, 5):
        pit_in = pd.to_timedelta(1000, unit="s") if lap == 2 else NaT
        rows.append(_row(2023, "PER", "Red Bull", lap, per_2023_positions[lap], 93.0, pit_in=pit_in))
    for lap in range(1, 6):
        rows.append(_row(2024, "PER", "Red Bull", lap, 3.0, 93.0))

    # HAM (Mercedes, solo - no teammate that year): used to check teammate gap is NaN/absent.
    for lap in range(1, 3):
        rows.append(_row(2023, "HAM", "Mercedes", lap, 4.0, 92.0))

    return pd.DataFrame(rows)


def test_year_fastest_lap(sample_fleet_laps_df):
    result = year_fastest_lap(sample_fleet_laps_df).set_index("Year")
    assert result.loc[2023, "YearFastestLapSeconds"] == 90.0
    assert result.loc[2024, "YearFastestLapSeconds"] == 90.0


def test_driver_relative_pace_by_year_fastest_lap_gap(sample_fleet_laps_df):
    pace = driver_relative_pace_by_year(sample_fleet_laps_df)
    ver_2023 = pace[(pace["Driver"] == "VER") & (pace["Year"] == 2023)]
    assert ver_2023["FastestLapGapPct"].iloc[0] == pytest.approx((91.0 - 90.0) / 90.0 * 100)


def test_field_average_gap_by_year(sample_fleet_laps_df):
    gap = field_average_gap_by_year(sample_fleet_laps_df)
    ham_2023 = gap[(gap["Driver"] == "HAM") & (gap["Year"] == 2023)]
    # Field average is the per-lap mean across every lap in 2023 (not a simple
    # average of driver averages, since drivers ran different lap counts).
    expected = sample_fleet_laps_df.loc[sample_fleet_laps_df["Year"] == 2023, "LapTimeSeconds"].mean()
    assert ham_2023["YearFieldAverageLapSeconds"].iloc[0] == pytest.approx(expected)


def test_teammate_gap_by_year_detects_decline(sample_fleet_laps_df):
    gap = teammate_gap_by_year(sample_fleet_laps_df)
    ver_2023 = gap[(gap["Driver"] == "VER") & (gap["Year"] == 2023)]["TeammateGapPct"].iloc[0]
    ver_2024 = gap[(gap["Driver"] == "VER") & (gap["Year"] == 2024)]["TeammateGapPct"].iloc[0]
    # 2023: VER (91) vs PER (93) -> faster than teammate (negative gap)
    assert ver_2023 == pytest.approx((91.0 - 93.0) / 93.0 * 100)
    # 2024: VER (99) vs PER (93) -> much slower than teammate (positive gap)
    assert ver_2024 == pytest.approx((99.0 - 93.0) / 93.0 * 100)
    assert ver_2024 > ver_2023


def test_teammate_gap_by_year_excludes_solo_driver(sample_fleet_laps_df):
    gap = teammate_gap_by_year(sample_fleet_laps_df)
    assert gap[gap["Driver"] == "HAM"].empty


def test_team_relative_pace_by_year_averages_teammates(sample_fleet_laps_df):
    team_pace = team_relative_pace_by_year(sample_fleet_laps_df)
    driver_pace = driver_relative_pace_by_year(sample_fleet_laps_df)
    red_bull_2023 = team_pace[(team_pace["Team"] == "Red Bull") & (team_pace["Year"] == 2023)]
    ver_gap = driver_pace[(driver_pace["Driver"] == "VER") & (driver_pace["Year"] == 2023)]["FastestLapGapPct"].iloc[0]
    per_gap = driver_pace[(driver_pace["Driver"] == "PER") & (driver_pace["Year"] == 2023)]["FastestLapGapPct"].iloc[0]
    assert red_bull_2023["FastestLapGapPct"].iloc[0] == pytest.approx((ver_gap + per_gap) / 2)


def test_degradation_by_year_separates_years(sample_fleet_laps_df):
    result = degradation_by_year(sample_fleet_laps_df)
    ver_years = set(result[result["Driver"] == "VER"]["Year"])
    assert ver_years == {2023, 2024}


def test_race_completion_rate_flags_short_race(sample_fleet_laps_df):
    rate = race_completion_rate_by_year(sample_fleet_laps_df)
    per_2023 = rate[(rate["Driver"] == "PER") & (rate["Year"] == 2023)]
    ver_2023 = rate[(rate["Driver"] == "VER") & (rate["Year"] == 2023)]
    assert per_2023["RaceCompletionRatePct"].iloc[0] == pytest.approx(80.0)
    assert ver_2023["RaceCompletionRatePct"].iloc[0] == pytest.approx(100.0)


def test_lap_time_consistency_by_year(sample_fleet_laps_df):
    consistency = lap_time_consistency_by_year(sample_fleet_laps_df)
    lec_2023 = consistency[(consistency["Driver"] == "LEC") & (consistency["Year"] == 2023)]
    assert lec_2023["LapTimeConsistencyStd"].iloc[0] == pytest.approx(0.0)


def test_pit_stop_count_by_year(sample_fleet_laps_df):
    counts = pit_stop_count_by_year(sample_fleet_laps_df)
    per_2023 = counts[(counts["Driver"] == "PER") & (counts["Year"] == 2023)]
    assert per_2023["PitStopCount"].iloc[0] == 1
    assert counts[counts["Driver"] == "VER"].empty  # no recorded stops


def test_pit_stop_recovery_laps_by_year(sample_fleet_laps_df):
    recovery = pit_stop_recovery_laps_by_year(sample_fleet_laps_df)
    per_2023 = recovery[(recovery["Driver"] == "PER") & (recovery["Year"] == 2023)]
    assert per_2023["AvgPitStopRecoveryLaps"].iloc[0] == pytest.approx(2.0)


def test_year_over_year_shift_flags_decline_on_teammate_gap(sample_fleet_laps_df):
    teammate_pace = teammate_gap_by_year(sample_fleet_laps_df)
    shifts = year_over_year_shift(teammate_pace, "TeammateGapPct")
    ver_shift = shifts[(shifts["Driver"] == "VER") & (shifts["ToYear"] == 2024)]
    assert ver_shift["Shift"].iloc[0] == "Declined"


def test_fleet_benchmark_table_has_expected_columns(sample_fleet_laps_df):
    table = fleet_benchmark_table(sample_fleet_laps_df)
    expected = {
        "Year", "Driver", "Team", "TeammateGapPct", "FieldAverageGapPct", "FastestLapGapPct",
        "AvgDegradationSecondsPerLap", "RaceCompletionRatePct", "LapTimeConsistencyStd",
        "PitStopCount", "Shift",
    }
    assert expected.issubset(table.columns)
    ham_2023 = table[(table["Driver"] == "HAM") & (table["Year"] == 2023)]
    assert ham_2023["Shift"].iloc[0] == "N/A (no teammate data, or first year)"