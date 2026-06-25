import pandas as pd
import pytest

from src.context_engine import align_context_to_session
from src.health_assessment import (
    HEALTH_EXPLAINED,
    HEALTH_UNEXPLAINED,
    assess_anomaly_health,
    health_summary,
)


@pytest.fixture
def laps_df():
    # 10 laps per driver: 9 identical "normal" laps plus 1 outlier. With a
    # small sample, an outlier among otherwise-identical values caps at
    # z = k/sqrt(k+1) (k = number of normal laps) - 9 normal laps gives
    # z ≈ 2.85, safely above ANOMALY_ZSCORE_THRESHOLD (2.5); fewer laps
    # mathematically can't cross it regardless of how extreme the outlier is.
    num_laps = 10
    anomaly_index = 4  # lap 5

    # VER: the slow lap has no pit stop, moderate tyre age, Green track -
    # the z-score flags it, and context has nothing to explain it with, so
    # it should be escalated as Unexplained.
    ver_lap_times = [90.0] * num_laps
    ver_lap_times[anomaly_index] = 200.0
    ver_tyre_life = list(range(1, num_laps + 1))

    # BOT: the slow lap is a real pit out-lap on a fresh tyre - the same
    # shape as the real BOT lap 13 validation case - so it should be
    # Explained.
    bot_lap_times = [90.0] * num_laps
    bot_lap_times[anomaly_index] = 200.0
    bot_tyre_life = [5, 6, 7, 8, 1, 2, 3, 4, 5, 6]
    bot_pit_in = [pd.NaT] * num_laps
    bot_pit_in[anomaly_index] = pd.to_timedelta(300.0, unit="s")

    rows = []
    for i in range(num_laps):
        lap_number = i + 1
        rows.append(
            {
                "Driver": "VER",
                "LapNumber": lap_number,
                "LapTimeSeconds": ver_lap_times[i],
                "Compound": "SOFT",
                "TyreLife": float(ver_tyre_life[i]),
                "TrackStatus": "1",
                "LapStartTimeSeconds": float(i * 90),
                "PitInTime": pd.NaT,
                "PitOutTime": pd.NaT,
            }
        )
        rows.append(
            {
                "Driver": "BOT",
                "LapNumber": lap_number,
                "LapTimeSeconds": bot_lap_times[i],
                "Compound": "HARD" if i >= anomaly_index else "SOFT",
                "TyreLife": float(bot_tyre_life[i]),
                "TrackStatus": "1",
                "LapStartTimeSeconds": float(i * 90),
                "PitInTime": bot_pit_in[i],
                "PitOutTime": pd.NaT,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def weather_df():
    return pd.DataFrame(
        {
            "Time": pd.to_timedelta([0, 450], unit="s"),
            "AirTemp": [28.0, 28.0],
            "TrackTemp": [35.0, 35.0],
            "Humidity": [40.0, 40.0],
            "WindSpeed": [9.0, 9.0],
            "WindDirection": [200, 200],
            "Pressure": [1012.0, 1012.0],
            "Rainfall": [False, False],
        }
    )


@pytest.fixture
def context(laps_df, weather_df):
    race_control_df = pd.DataFrame(columns=["Category", "Message", "SessionTimeSeconds"])
    return align_context_to_session({"weather": weather_df, "laps": laps_df, "race_control": race_control_df})


def test_assess_anomaly_health_explains_real_pit_out_lap_pattern(laps_df, context):
    assessed = assess_anomaly_health(laps_df, context)
    bot_row = assessed[(assessed["Driver"] == "BOT") & (assessed["LapNumber"] == 5)]
    assert not bot_row.empty
    assert bot_row["HealthStatus"].iloc[0] == HEALTH_EXPLAINED
    assert "pit stop" in bot_row["Explanation"].iloc[0].lower()


def test_assess_anomaly_health_escalates_unexplained_slow_lap(laps_df, context):
    assessed = assess_anomaly_health(laps_df, context)
    ver_row = assessed[(assessed["Driver"] == "VER") & (assessed["LapNumber"] == 5)]
    assert not ver_row.empty
    assert ver_row["HealthStatus"].iloc[0] == HEALTH_UNEXPLAINED


def test_assess_anomaly_health_excludes_fast_laps(laps_df, context):
    # None of the fixture's fast laps should be flagged - only slow
    # (positive z-score) laps are health-assessed.
    assessed = assess_anomaly_health(laps_df, context)
    assert (assessed["LapTimeZScore"] > 0).all()


def test_assess_anomaly_health_empty_when_no_anomalies():
    flat_laps = pd.DataFrame(
        {
            "Driver": ["VER"] * 3,
            "LapNumber": [1, 2, 3],
            "LapTimeSeconds": [90.0, 90.1, 90.0],
            "Compound": ["SOFT"] * 3,
            "TyreLife": [1.0, 2.0, 3.0],
            "TrackStatus": ["1"] * 3,
            "LapStartTimeSeconds": [0.0, 90.0, 180.0],
            "PitInTime": [pd.NaT] * 3,
            "PitOutTime": [pd.NaT] * 3,
        }
    )
    context = align_context_to_session(
        {
            "weather": pd.DataFrame(columns=["Time"]),
            "laps": flat_laps,
            "race_control": pd.DataFrame(columns=["Category", "Message", "SessionTimeSeconds"]),
        }
    )
    assessed = assess_anomaly_health(flat_laps, context)
    assert assessed.empty
    assert health_summary(assessed) == {
        "TotalFlagged": 0,
        "Explained": 0,
        "PartiallyExplained": 0,
        "Unexplained": 0,
        "NoiseReductionPct": 0.0,
    }


def test_health_summary_counts_and_noise_reduction(laps_df, context):
    assessed = assess_anomaly_health(laps_df, context)
    summary = health_summary(assessed)
    assert summary["TotalFlagged"] == len(assessed)
    assert summary["Explained"] + summary["PartiallyExplained"] + summary["Unexplained"] == summary["TotalFlagged"]
    assert summary["NoiseReductionPct"] == pytest.approx(100.0 * summary["Explained"] / summary["TotalFlagged"])
