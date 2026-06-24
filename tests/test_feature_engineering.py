import pandas as pd
import pytest

from src.feature_engineering import (
    add_forecast_features,
    add_rolling_lap_time,
    add_stint_lap_number,
    build_features,
)


@pytest.fixture
def sample_laps_df():
    return pd.DataFrame(
        {
            "Driver": ["VER", "VER", "VER", "VER", "LEC", "LEC"],
            "Stint": [1, 1, 2, 2, 1, 1],
            "LapNumber": [1, 2, 3, 4, 1, 2],
            "LapTimeSeconds": [90.0, 91.0, 89.0, 89.5, 92.0, 93.0],
        }
    )


def test_add_stint_lap_number_resets_per_stint(sample_laps_df):
    result = add_stint_lap_number(sample_laps_df)
    ver_rows = result[result["Driver"] == "VER"]["StintLap"].tolist()
    assert ver_rows == [1, 2, 1, 2]


def test_add_stint_lap_number_resets_per_driver(sample_laps_df):
    result = add_stint_lap_number(sample_laps_df)
    lec_rows = result[result["Driver"] == "LEC"]["StintLap"].tolist()
    assert lec_rows == [1, 2]


def test_add_rolling_lap_time_matches_manual_mean(sample_laps_df):
    result = add_rolling_lap_time(sample_laps_df, window=2)
    ver_rolling = result[result["Driver"] == "VER"]["RollingLapTimeSeconds"].tolist()
    assert ver_rolling[0] == pytest.approx(90.0)
    assert ver_rolling[1] == pytest.approx((90.0 + 91.0) / 2)
    assert ver_rolling[2] == pytest.approx((91.0 + 89.0) / 2)


def test_build_features_adds_expected_columns(sample_laps_df):
    result = build_features(sample_laps_df)
    assert "StintLap" in result.columns
    assert "RollingLapTimeSeconds" in result.columns
    assert len(result) == len(sample_laps_df)


@pytest.fixture
def sample_three_lap_stint_df():
    return pd.DataFrame(
        {
            "Driver": ["VER", "VER", "VER", "VER"],
            "Stint": [1, 1, 1, 2],
            "LapNumber": [1, 2, 3, 4],
            "LapTimeSeconds": [90.0, 91.0, 92.5, 88.0],
        }
    )


def test_add_forecast_features_first_lap_of_stint_is_nan(sample_three_lap_stint_df):
    result = add_forecast_features(sample_three_lap_stint_df)
    first_lap = result[result["StintLap"] == 1]
    assert first_lap["PrevLapTimeSeconds"].isna().all()
    assert first_lap["Rolling3PrevLapTimeSeconds"].isna().all()


def test_add_forecast_features_lag_uses_only_previous_laps(sample_three_lap_stint_df):
    result = add_forecast_features(sample_three_lap_stint_df).sort_values("LapNumber")
    stint1 = result[result["Stint"] == 1]
    assert stint1["PrevLapTimeSeconds"].isna().tolist() == [True, False, False]
    assert stint1["PrevLapTimeSeconds"].dropna().tolist() == [90.0, 91.0]
    # Rolling3Prev at lap 3 is the mean of laps before it (90.0, 91.0), not including lap 3 itself.
    assert stint1["Rolling3PrevLapTimeSeconds"].iloc[2] == pytest.approx((90.0 + 91.0) / 2)


def test_add_forecast_features_does_not_leak_across_stints(sample_three_lap_stint_df):
    result = add_forecast_features(sample_three_lap_stint_df)
    stint2_first_lap = result[(result["Stint"] == 2) & (result["StintLap"] == 1)]
    assert stint2_first_lap["PrevLapTimeSeconds"].isna().all()
