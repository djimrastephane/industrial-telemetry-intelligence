import pandas as pd
import pytest

from src.decision_support import (
    build_recommendations_table,
    project_lap_to_threshold,
    recommend_pit_window,
)


def test_project_lap_to_threshold_returns_none_when_improving():
    assert project_lap_to_threshold(intercept=90.0, slope=-0.2, current_stint_lap=5) is None


def test_project_lap_to_threshold_returns_current_lap_when_already_past_threshold():
    # intercept=90, threshold 5% -> 94.5. At lap 10 with slope 1.0, projected = 100, already past.
    result = project_lap_to_threshold(intercept=90.0, slope=1.0, current_stint_lap=10, threshold_pct=5.0)
    assert result == 10


def test_project_lap_to_threshold_returns_none_beyond_horizon():
    # Tiny slope means the 5% threshold is laps and laps away.
    result = project_lap_to_threshold(
        intercept=90.0, slope=0.01, current_stint_lap=1, threshold_pct=5.0, max_horizon=10
    )
    assert result is None


def test_project_lap_to_threshold_computes_correct_crossing_lap():
    # intercept=100, threshold 10% -> 110. slope=2/lap -> crosses at lap (110-100)/2 = 5.
    result = project_lap_to_threshold(
        intercept=100.0, slope=2.0, current_stint_lap=1, threshold_pct=10.0, max_horizon=10
    )
    assert result == 5


@pytest.fixture
def degrading_stint_df():
    # Lap time climbs steadily from ~90s - will cross a 5% threshold within a few laps.
    rows = []
    for lap in range(1, 6):
        rows.append(
            {
                "Driver": "VER", "Team": "Red Bull", "LapNumber": lap, "LapTimeSeconds": 90.0 + 1.5 * (lap - 1),
                "Compound": "SOFT", "TyreLife": lap, "Stint": 1,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def improving_stint_df():
    rows = []
    for lap in range(1, 6):
        rows.append(
            {
                "Driver": "LEC", "Team": "Ferrari", "LapNumber": lap, "LapTimeSeconds": 95.0 - 0.3 * (lap - 1),
                "Compound": "HARD", "TyreLife": lap, "Stint": 1,
            }
        )
    return pd.DataFrame(rows)


def test_recommend_pit_window_for_degrading_stint(degrading_stint_df):
    rec = recommend_pit_window(degrading_stint_df, "VER", 1)
    assert rec["found"] is True
    assert rec["degradation_seconds_per_lap"] > 0
    assert "Pit" in rec["recommended_action"]


def test_recommend_pit_window_for_improving_stint(improving_stint_df):
    rec = recommend_pit_window(improving_stint_df, "LEC", 1)
    assert rec["found"] is True
    assert rec["recommended_action"] == "No action needed (lap times stable or improving)"


def test_recommend_pit_window_unknown_driver_or_stint(degrading_stint_df):
    rec = recommend_pit_window(degrading_stint_df, "NOBODY", 1)
    assert rec["found"] is False


def test_build_recommendations_table_covers_all_stints(degrading_stint_df, improving_stint_df):
    combined = pd.concat([degrading_stint_df, improving_stint_df], ignore_index=True)
    table = build_recommendations_table(combined, min_stint_laps=3)
    assert set(table["Driver"]) == {"VER", "LEC"}
    assert "RecommendedAction" in table.columns
    assert "RiskCategory" in table.columns


def test_build_recommendations_table_skips_short_stints():
    short_stint = pd.DataFrame(
        {
            "Driver": ["VER"], "Team": ["Red Bull"], "LapNumber": [1], "LapTimeSeconds": [90.0],
            "Compound": ["SOFT"], "TyreLife": [1], "Stint": [1],
        }
    )
    table = build_recommendations_table(short_stint, min_stint_laps=3)
    assert table.empty
