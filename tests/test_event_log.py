import pandas as pd
import pytest

from src.context_engine import align_context_to_session
from src.event_log import (
    EVENT_ANOMALY,
    EVENT_COLUMNS,
    EVENT_PIT,
    EVENT_RACE_CONTROL,
    EVENT_RECOMMENDATION,
    EVENT_TRACK_STATUS,
    SEVERITY_CRITICAL,
    SEVERITY_INFO,
    SEVERITY_WARNING,
    build_event_log,
)
from src.health_assessment import HEALTH_PARTIALLY_EXPLAINED, HEALTH_UNEXPLAINED


@pytest.fixture
def minimal_context():
    """Bare-minimum aligned context with no race control events."""
    race_control = pd.DataFrame(columns=["Category", "Message", "SessionTimeSeconds"])
    weather = pd.DataFrame(columns=["Time"])
    laps = pd.DataFrame(columns=["Driver", "LapNumber", "LapStartTimeSeconds"])
    return align_context_to_session({"weather": weather, "laps": laps, "race_control": race_control})


@pytest.fixture
def laps_with_pit():
    """Two drivers; VER pits in on lap 2, BOT pits out on lap 3."""
    return pd.DataFrame({
        "Driver": ["VER", "VER", "VER", "BOT", "BOT", "BOT"],
        "LapNumber": [1, 2, 3, 1, 2, 3],
        "LapTimeSeconds": [90.0, 120.0, 91.0, 90.0, 90.0, 120.0],
        "Compound": ["SOFT", "SOFT", "MEDIUM", "SOFT", "SOFT", "HARD"],
        "TyreLife": [1.0, 2.0, 1.0, 1.0, 2.0, 1.0],
        "TrackStatus": ["1", "1", "1", "1", "1", "2"],
        "LapStartTimeSeconds": [0.0, 90.0, 210.0, 0.0, 90.0, 180.0],
        "PitInTime": [pd.NaT, pd.to_timedelta(180.0, unit="s"), pd.NaT, pd.NaT, pd.NaT, pd.NaT],
        "PitOutTime": [pd.NaT, pd.NaT, pd.to_timedelta(212.0, unit="s"), pd.NaT, pd.NaT, pd.NaT],
        "Stint": [1, 1, 2, 1, 1, 2],
        "Position": [1, 2, 3, 4, 5, 6],
    })


def test_build_event_log_returns_dataframe(laps_with_pit, minimal_context):
    log = build_event_log(laps_with_pit, minimal_context)
    assert isinstance(log, pd.DataFrame)


def test_build_event_log_has_required_columns(laps_with_pit, minimal_context):
    log = build_event_log(laps_with_pit, minimal_context)
    for col in EVENT_COLUMNS:
        assert col in log.columns, f"Missing column: {col}"


def test_build_event_log_sorted_by_session_time(laps_with_pit, minimal_context):
    log = build_event_log(laps_with_pit, minimal_context)
    if len(log) > 1:
        assert (log["SessionTimeSeconds"].diff().dropna() >= 0).all()


def test_pit_events_detected(laps_with_pit, minimal_context):
    log = build_event_log(laps_with_pit, minimal_context)
    pit_rows = log[log["EventType"] == EVENT_PIT]
    assert len(pit_rows) >= 1
    # VER pit in should appear
    ver_pit_in = pit_rows[(pit_rows["Driver"] == "VER") & (pit_rows["Description"].str.contains("pit in"))]
    assert len(ver_pit_in) == 1
    assert ver_pit_in.iloc[0]["Severity"] == SEVERITY_INFO


def test_pit_out_event_detected(laps_with_pit, minimal_context):
    log = build_event_log(laps_with_pit, minimal_context)
    pit_out = log[log["Description"].str.contains("pit out") & (log["Driver"] == "VER")]
    assert len(pit_out) == 1


def test_track_status_change_detected(laps_with_pit, minimal_context):
    log = build_event_log(laps_with_pit, minimal_context)
    status_rows = log[log["EventType"] == EVENT_TRACK_STATUS]
    # TrackStatus goes 1 → 2 (Green → Yellow) starting from BOT lap 3
    assert len(status_rows) >= 1
    yellow_event = status_rows[status_rows["Description"].str.contains("Yellow")]
    assert len(yellow_event) == 1
    assert yellow_event.iloc[0]["Severity"] == SEVERITY_WARNING


def test_race_control_events_included(laps_with_pit):
    race_control = pd.DataFrame({
        "Category": ["Flag"],
        "Message": ["YELLOW FLAG"],
        "SessionTimeSeconds": [95.0],
    })
    weather = pd.DataFrame(columns=["Time"])
    laps = pd.DataFrame(columns=["Driver", "LapNumber", "LapStartTimeSeconds"])
    context = align_context_to_session({"weather": weather, "laps": laps, "race_control": race_control})
    log = build_event_log(laps_with_pit, context)
    rc_rows = log[log["EventType"] == EVENT_RACE_CONTROL]
    assert len(rc_rows) == 1
    assert "YELLOW FLAG" in rc_rows.iloc[0]["Description"]
    assert rc_rows.iloc[0]["Severity"] == SEVERITY_WARNING


def test_anomaly_events_included_for_actionable_anomalies(laps_with_pit, minimal_context):
    assessed = pd.DataFrame({
        "Driver": ["VER", "BOT"],
        "LapNumber": [2, 3],
        "LapTimeSeconds": [120.0, 120.0],
        "Compound": ["SOFT", "HARD"],
        "TyreLife": [2.0, 1.0],
        "LapTimeZScore": [3.0, 2.8],
        "HealthStatus": [HEALTH_UNEXPLAINED, HEALTH_PARTIALLY_EXPLAINED],
        "Confidence": ["Low", "Medium"],
        "Explanation": ["No context.", "Partial context."],
    })
    log = build_event_log(laps_with_pit, minimal_context, assessed_anomalies=assessed)
    anomaly_rows = log[log["EventType"] == EVENT_ANOMALY]
    assert len(anomaly_rows) == 2
    ver_row = anomaly_rows[anomaly_rows["Driver"] == "VER"]
    assert ver_row.iloc[0]["Severity"] == SEVERITY_CRITICAL
    bot_row = anomaly_rows[anomaly_rows["Driver"] == "BOT"]
    assert bot_row.iloc[0]["Severity"] == SEVERITY_WARNING


def test_explained_anomalies_not_surfaced(laps_with_pit, minimal_context):
    from src.health_assessment import HEALTH_EXPLAINED
    assessed = pd.DataFrame({
        "Driver": ["VER"],
        "LapNumber": [2],
        "LapTimeSeconds": [120.0],
        "Compound": ["SOFT"],
        "TyreLife": [2.0],
        "LapTimeZScore": [3.0],
        "HealthStatus": [HEALTH_EXPLAINED],
        "Confidence": ["High"],
        "Explanation": ["Pit out-lap."],
    })
    log = build_event_log(laps_with_pit, minimal_context, assessed_anomalies=assessed)
    anomaly_rows = log[log["EventType"] == EVENT_ANOMALY]
    assert len(anomaly_rows) == 0


def test_recommendation_events_included(laps_with_pit, minimal_context):
    recs = pd.DataFrame({
        "Driver": ["VER"],
        "Stint": [1],
        "Compound": ["SOFT"],
        "CurrentStintLap": [5],
        "DegradationSecondsPerLap": [1.5],
        "ProjectedCrossingStintLap": [5],
        "RiskCategory": ["High"],
        "RecommendedAction": ["Pit now"],
    })
    log = build_event_log(laps_with_pit, minimal_context, recommendations=recs)
    rec_rows = log[log["EventType"] == EVENT_RECOMMENDATION]
    assert len(rec_rows) == 1
    assert rec_rows.iloc[0]["Severity"] == SEVERITY_CRITICAL
    assert rec_rows.iloc[0]["Driver"] == "VER"


def test_no_action_recommendation_not_surfaced(laps_with_pit, minimal_context):
    recs = pd.DataFrame({
        "Driver": ["VER"],
        "Stint": [1],
        "Compound": ["SOFT"],
        "CurrentStintLap": [5],
        "DegradationSecondsPerLap": [-0.1],
        "ProjectedCrossingStintLap": [None],
        "RiskCategory": ["Low"],
        "RecommendedAction": ["No action needed (lap times stable or improving)"],
    })
    log = build_event_log(laps_with_pit, minimal_context, recommendations=recs)
    rec_rows = log[log["EventType"] == EVENT_RECOMMENDATION]
    assert len(rec_rows) == 0


def test_empty_laps_returns_empty_dataframe(minimal_context):
    empty_laps = pd.DataFrame(columns=[
        "Driver", "LapNumber", "LapTimeSeconds", "Compound", "TyreLife",
        "TrackStatus", "LapStartTimeSeconds", "PitInTime", "PitOutTime", "Stint",
    ])
    log = build_event_log(empty_laps, minimal_context)
    assert isinstance(log, pd.DataFrame)
    for col in EVENT_COLUMNS:
        assert col in log.columns
