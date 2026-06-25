import pandas as pd
import pytest

from src import config
from src.context_engine import (
    OBSERVED_LAPTIME_INCREASE,
    OBSERVED_LOW_SPEED,
    OBSERVED_SPEED_ANOMALY,
    align_context_to_session,
    calculate_context_changes,
    generate_context_summary,
    get_context_at_timestamp,
    load_context,
)


@pytest.fixture
def weather_df():
    return pd.DataFrame(
        {
            "Time": pd.to_timedelta([0, 60, 120, 180], unit="s"),
            "AirTemp": [28.0, 28.0, 29.0, 31.0],
            "TrackTemp": [35.0, 35.0, 38.0, 43.0],
            "Humidity": [40.0, 40.0, 38.0, 18.0],
            "WindSpeed": [9.0, 9.0, 9.0, 12.0],
            "WindDirection": [200, 200, 220, 245],
            "Pressure": [1012.0, 1012.0, 1012.0, 1012.0],
            "Rainfall": [False, False, False, False],
        }
    )


@pytest.fixture
def laps_df():
    return pd.DataFrame(
        {
            "Driver": ["VER", "VER", "VER", "LEC"],
            "LapNumber": [1.0, 2.0, 3.0, 1.0],
            "LapStartTimeSeconds": [0.0, 90.0, 180.0, 0.0],
            "Compound": ["SOFT", "SOFT", "MEDIUM", "HARD"],
            "TyreLife": [1.0, 2.0, 1.0, 30.0],
            "FreshTyre": [True, False, True, False],
            "Stint": [1.0, 1.0, 2.0, 1.0],
            "TrackStatus": ["1", "1", "2", "1"],
            "PitInTime": pd.to_timedelta([None, None, 178.0, None], unit="s"),
            "PitOutTime": [pd.NaT, pd.NaT, pd.NaT, pd.NaT],
        }
    )


@pytest.fixture
def race_control_df():
    return pd.DataFrame(
        {
            "Category": ["Flag", "Flag"],
            "Message": ["DOUBLE YELLOW IN TRACK SECTOR 2", "CLEAR IN TRACK SECTOR 2"],
            "SessionTimeSeconds": [85.0, 95.0],
        }
    )


@pytest.fixture
def context(weather_df, laps_df, race_control_df):
    return align_context_to_session({"weather": weather_df, "laps": laps_df, "race_control": race_control_df})


def test_load_context_reads_all_three_sources(tmp_path, monkeypatch, weather_df, laps_df, race_control_df):
    monkeypatch.setattr(config, "WEATHER_FILE", tmp_path / "weather.parquet")
    monkeypatch.setattr(config, "LAPS_FILE", tmp_path / "laps.parquet")
    monkeypatch.setattr(config, "RACE_CONTROL_FILE", tmp_path / "race_control.parquet")

    weather_df.to_parquet(config.WEATHER_FILE, index=False)
    laps_df.to_parquet(config.LAPS_FILE, index=False)
    race_control_df.to_parquet(config.RACE_CONTROL_FILE, index=False)

    context = load_context()
    assert set(context.keys()) == {"weather", "laps", "race_control"}
    assert len(context["weather"]) == 4
    assert len(context["laps"]) == 4
    assert len(context["race_control"]) == 2


def test_load_context_handles_missing_race_control_file(tmp_path, monkeypatch, weather_df, laps_df):
    monkeypatch.setattr(config, "WEATHER_FILE", tmp_path / "weather.parquet")
    monkeypatch.setattr(config, "LAPS_FILE", tmp_path / "laps.parquet")
    monkeypatch.setattr(config, "RACE_CONTROL_FILE", tmp_path / "missing_race_control.parquet")

    weather_df.to_parquet(config.WEATHER_FILE, index=False)
    laps_df.to_parquet(config.LAPS_FILE, index=False)

    context = load_context()
    assert context["race_control"].empty


def test_align_context_to_session_adds_weather_time_seconds(weather_df, laps_df, race_control_df):
    context = align_context_to_session({"weather": weather_df, "laps": laps_df, "race_control": race_control_df})
    assert "TimeSeconds" in context["weather"].columns
    assert context["weather"]["TimeSeconds"].tolist() == [0.0, 60.0, 120.0, 180.0]


def test_get_context_at_timestamp_interpolates_weather(context):
    # t=30 sits halfway between the t=0 and t=60 samples, both TrackTemp=35.0.
    result = get_context_at_timestamp(context, 30.0, "VER")
    assert result["AirTemp"] == pytest.approx(28.0)
    assert result["TrackTemp"] == pytest.approx(35.0)

    # t=90 sits halfway between the t=60 (35.0) and t=120 (38.0) samples.
    midpoint = get_context_at_timestamp(context, 90.0, "VER")
    assert midpoint["TrackTemp"] == pytest.approx(36.5)


def test_get_context_at_timestamp_picks_correct_lap(context):
    result = get_context_at_timestamp(context, 100.0, "VER")
    assert result["LapNumber"] == 2.0
    assert result["Compound"] == "SOFT"
    assert result["TyreLife"] == 2.0


def test_get_context_at_timestamp_detects_track_status(context):
    before = get_context_at_timestamp(context, 10.0, "VER")
    after = get_context_at_timestamp(context, 190.0, "VER")
    assert before["TrackStatus"] == "Green"
    assert after["TrackStatus"] == "Yellow"


def test_get_context_at_timestamp_detects_pit_stop(context):
    result = get_context_at_timestamp(context, 190.0, "VER")
    assert result["PitThisLap"] is True


def test_get_context_at_timestamp_finds_recent_race_control_event(context):
    # Events fire at t=85 and t=95; at t=100, both are in the past and the
    # most recent (t=95) should win.
    result = get_context_at_timestamp(context, 100.0, "VER")
    assert result["RecentEvent"] == "CLEAR IN TRACK SECTOR 2"


def test_get_context_at_timestamp_no_recent_event_outside_window(context):
    result = get_context_at_timestamp(context, 5.0, "VER")
    assert result["RecentEvent"] == "No significant events"


def test_get_context_at_timestamp_unknown_driver_is_graceful(context):
    result = get_context_at_timestamp(context, 30.0, "ZZZ")
    assert result["TrackStatus"] == "Unknown"
    assert result["PitThisLap"] is False


def test_get_context_at_timestamp_handles_empty_weather(laps_df, race_control_df):
    context = align_context_to_session(
        {"weather": pd.DataFrame(columns=["Time"]), "laps": laps_df, "race_control": race_control_df}
    )
    result = get_context_at_timestamp(context, 30.0, "VER")
    assert "AirTemp" not in result
    assert result["Compound"] == "SOFT"


def test_calculate_context_changes_detects_tyre_aging(context):
    changes = calculate_context_changes(context, 100.0, "VER")
    assert changes["TyreLife"]["current"] == 2.0
    assert changes["TyreLife"]["previous"] == 1.0
    assert changes["TyreLife"]["trend"] == "▲ Aging"


def test_calculate_context_changes_detects_tyre_reset_after_pit(context):
    changes = calculate_context_changes(context, 190.0, "VER")
    assert changes["TyreLife"]["trend"] == "▼ Reset"
    assert changes["Compound"]["changed"] is True


def test_calculate_context_changes_detects_track_status_change(context):
    changes = calculate_context_changes(context, 190.0, "VER")
    assert changes["TrackStatus"]["changed"] is True
    assert changes["TrackStatus"]["previous"] == "Green"
    assert changes["TrackStatus"]["current"] == "Yellow"


def test_calculate_context_changes_weather_trend_direction(context):
    changes = calculate_context_changes(context, 180.0, "VER")
    assert changes["TrackTemp"]["trend"] == "▲"
    assert changes["Humidity"]["trend"] == "▼"


def test_calculate_context_changes_returns_empty_for_first_sample(context):
    changes = calculate_context_changes(context, 0.0, "VER")
    assert "AirTemp" not in changes
    assert "TyreLife" not in changes


def test_calculate_context_changes_handles_missing_driver(context):
    changes = calculate_context_changes(context, 100.0, "ZZZ")
    assert "TyreLife" not in changes
    assert "AirTemp" in changes  # weather changes are driver-independent


def test_generate_context_summary_stable_with_no_changes():
    summary = generate_context_summary({"TrackStatus": "Green"}, {})
    assert summary["status"] == "Stable"
    assert summary["color"] == "green"
    assert summary["confidence"] is None
    assert "stable" in summary["interpretation"][0].lower()


def test_generate_context_summary_low_speed_explained_by_tyre_degradation():
    summary = generate_context_summary(
        {"TrackStatus": "Green", "TrackTemp": 45.0, "TyreLife": 32}, {}, OBSERVED_LOW_SPEED
    )
    assert summary["confidence"] == "High"
    assert "tyre degradation" in summary["interpretation"][-1].lower()


def test_generate_context_summary_low_speed_explained_by_pit_out_lap():
    # Real validation case: BOT lap 13 of the 2024 Bahrain race - a pit
    # out-lap on a fresh tyre (TyreLife=1, below the degradation threshold)
    # on a Green track. Before the PitThisLap rule existed, this fell
    # through to "Low confidence: anomaly", which was wrong - it's a known,
    # direct cause, not an unexplained anomaly.
    summary = generate_context_summary(
        {"TrackStatus": "Green", "TyreLife": 1, "PitThisLap": True}, {}, OBSERVED_LOW_SPEED
    )
    assert summary["confidence"] == "High"
    assert "pit stop" in summary["interpretation"][-1].lower()


def test_generate_context_summary_laptime_increase_also_explained_by_pit_out_lap():
    # A z-score anomaly is flagged on LapTimeSeconds, so its natural
    # observed_effect is "laptime_increase", not "low_speed" - the same pit
    # out-lap fact must explain it under either label, since they're the
    # same physical symptom (the car went slower than expected).
    summary = generate_context_summary(
        {"TrackStatus": "Green", "TyreLife": 1, "PitThisLap": True}, {}, OBSERVED_LAPTIME_INCREASE
    )
    assert summary["confidence"] == "High"
    assert "pit stop" in summary["interpretation"][-1].lower()


def test_generate_context_summary_pit_out_lap_takes_priority_over_anomaly():
    # Without the PitThisLap flag, the same low tyre life + Green track is
    # correctly read as an unexplained anomaly instead.
    summary = generate_context_summary({"TrackStatus": "Green", "TyreLife": 1}, {}, OBSERVED_LOW_SPEED)
    assert summary["confidence"] == "Low"


def test_generate_context_summary_laptime_increase_explained_by_vsc():
    summary = generate_context_summary(
        {"TrackStatus": "Virtual Safety Car"}, {}, OBSERVED_LAPTIME_INCREASE
    )
    assert summary["confidence"] == "High"
    assert "track condition" in summary["interpretation"][-1].lower()
    assert summary["status"] == "Rapidly Changing"


def test_generate_context_summary_anomaly_with_no_explanation():
    summary = generate_context_summary({"TrackStatus": "Green", "TyreLife": 3}, {}, OBSERVED_SPEED_ANOMALY)
    assert summary["confidence"] == "Low"
    assert "anomaly" in summary["interpretation"][-1].lower()


def test_generate_context_summary_partial_confidence_fallback():
    summary = generate_context_summary({"TrackStatus": "Yellow", "TyreLife": 3}, {}, OBSERVED_LOW_SPEED)
    assert summary["confidence"] in ("High", "Medium")  # Yellow is significant -> High via track condition


def test_generate_context_summary_escalates_severity_on_weather_change():
    changes = {
        "TrackTemp": {"current": 45.0, "previous": 38.0, "diff": 7.0, "trend": "▲"},
    }
    summary = generate_context_summary({"TrackStatus": "Green"}, changes)
    assert summary["status"] == "Rapidly Changing"
    assert "tyre degradation" in summary["interpretation"][0].lower()


def test_generate_context_summary_handles_missing_track_status_key():
    summary = generate_context_summary({}, {})
    assert summary["status"] == "Stable"
