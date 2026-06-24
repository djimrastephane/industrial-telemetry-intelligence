import numpy as np
import pandas as pd
import pytest

from src.seasonal_analysis import (
    finishing_position_per_race,
    season_driver_kpis,
    season_team_kpis,
    speed_trap_trend,
    team_position_trend,
)


@pytest.fixture
def sample_season_laps_df():
    # VER finishes P1 in round 1, P3 in round 2 -> declining trend (slope > 0)
    # HAM finishes P5 in round 1, P2 in round 2 -> improving trend (slope < 0)
    rows = [
        {"RoundNumber": 1, "EventName": "Race A", "Driver": "VER", "Team": "Red Bull", "LapNumber": 1, "Position": 2.0, "SpeedST": 300.0},
        {"RoundNumber": 1, "EventName": "Race A", "Driver": "VER", "Team": "Red Bull", "LapNumber": 2, "Position": 1.0, "SpeedST": 310.0},
        {"RoundNumber": 2, "EventName": "Race B", "Driver": "VER", "Team": "Red Bull", "LapNumber": 1, "Position": 3.0, "SpeedST": 305.0},
        {"RoundNumber": 2, "EventName": "Race B", "Driver": "VER", "Team": "Red Bull", "LapNumber": 2, "Position": 3.0, "SpeedST": 308.0},
        {"RoundNumber": 1, "EventName": "Race A", "Driver": "HAM", "Team": "Mercedes", "LapNumber": 1, "Position": 6.0, "SpeedST": 295.0},
        {"RoundNumber": 1, "EventName": "Race A", "Driver": "HAM", "Team": "Mercedes", "LapNumber": 2, "Position": 5.0, "SpeedST": 297.0},
        {"RoundNumber": 2, "EventName": "Race B", "Driver": "HAM", "Team": "Mercedes", "LapNumber": 1, "Position": 3.0, "SpeedST": 300.0},
        {"RoundNumber": 2, "EventName": "Race B", "Driver": "HAM", "Team": "Mercedes", "LapNumber": 2, "Position": 2.0, "SpeedST": 302.0},
    ]
    return pd.DataFrame(rows)


def test_finishing_position_per_race_takes_last_lap(sample_season_laps_df):
    result = finishing_position_per_race(sample_season_laps_df)
    ver_round1 = result[(result["Driver"] == "VER") & (result["RoundNumber"] == 1)]
    assert ver_round1["FinishPosition"].iloc[0] == 1.0


def test_team_position_trend_averages_drivers(sample_season_laps_df):
    result = team_position_trend(sample_season_laps_df)
    red_bull_round1 = result[(result["Team"] == "Red Bull") & (result["RoundNumber"] == 1)]
    assert red_bull_round1["FinishPosition"].iloc[0] == pytest.approx(1.0)


def test_speed_trap_trend_averages_speed(sample_season_laps_df):
    result = speed_trap_trend(sample_season_laps_df)
    ver_round1 = result[(result["Driver"] == "VER") & (result["RoundNumber"] == 1)]
    assert ver_round1["AvgSpeedTrap"].iloc[0] == pytest.approx((300.0 + 310.0) / 2)


def test_season_driver_kpis_trend_direction(sample_season_laps_df):
    kpis = season_driver_kpis(sample_season_laps_df).set_index("Driver")
    # VER: P1 -> P3, getting worse, positive slope
    assert kpis.loc["VER", "PositionTrendSlope"] > 0
    # HAM: P5 -> P2, improving, negative slope
    assert kpis.loc["HAM", "PositionTrendSlope"] < 0


def test_season_team_kpis_has_expected_columns(sample_season_laps_df):
    kpis = season_team_kpis(sample_season_laps_df)
    assert {"Team", "RacesCompleted", "AvgFinishPosition", "PositionTrendSlope"}.issubset(kpis.columns)
    assert len(kpis) == 2
