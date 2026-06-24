import pandas as pd
import pytest

from src.degradation_analysis import degradation_per_stint


@pytest.fixture
def sample_laps_df():
    rows = []
    for lap, lt in enumerate([90.0, 91.0, 92.0, 93.0], start=1):
        rows.append({"Driver": "VER", "Compound": "SOFT", "Stint": 1, "LapNumber": lap, "LapTimeSeconds": lt})
    return pd.DataFrame(rows)


def test_degradation_per_stint_fits_positive_slope(sample_laps_df):
    result = degradation_per_stint(sample_laps_df)
    ver_row = result[(result["Driver"] == "VER") & (result["Stint"] == 1)]
    assert ver_row["DegradationSecondsPerLap"].iloc[0] == pytest.approx(1.0)
    assert ver_row["Laps"].iloc[0] == 4


def test_degradation_per_stint_skips_single_lap_stints():
    # Every stint has fewer than 2 laps, so no slope can be fit anywhere -
    # regression test for a bug where this returned a columnless empty
    # DataFrame and crashed downstream .sort_values(["Driver", "Stint"]) calls.
    single_lap_df = pd.DataFrame(
        {"Driver": ["VER"], "Compound": ["SOFT"], "Stint": [1], "LapNumber": [1], "LapTimeSeconds": [90.0]}
    )
    result = degradation_per_stint(single_lap_df)
    assert result.empty
    assert list(result.columns) == [
        "Driver", "Stint", "Compound", "Laps", "DegradationSecondsPerLap", "StartingLapTimeEstimate",
    ]


def test_degradation_per_stint_empty_input_has_expected_columns():
    empty_df = pd.DataFrame(columns=["Driver", "Compound", "Stint", "LapNumber", "LapTimeSeconds"])
    result = degradation_per_stint(empty_df)
    assert result.empty
    assert "Driver" in result.columns
