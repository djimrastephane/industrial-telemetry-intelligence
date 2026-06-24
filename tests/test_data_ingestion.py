"""Tests for data_ingestion: parquet round-trip and schema checks.

No network calls here - FastF1 downloads are exercised manually via
`python -m src.data_ingestion`, not in the test suite.
"""

import pandas as pd
import pytest

from src import config
from src.data_ingestion import save_processed


@pytest.fixture
def sample_laps_df():
    return pd.DataFrame(
        {
            "Driver": ["VER", "VER", "LEC"],
            "Team": ["Red Bull", "Red Bull", "Ferrari"],
            "LapNumber": [1, 2, 1],
            "Compound": ["SOFT", "SOFT", "MEDIUM"],
            "TyreLife": [1, 2, 1],
            "Stint": [1, 1, 1],
            "LapTimeSeconds": [91.2, 91.5, 92.0],
        }
    )


@pytest.fixture
def sample_weather_df():
    return pd.DataFrame({"AirTemp": [28.0, 28.2], "TrackTemp": [35.0, 35.5]})


@pytest.fixture
def sample_telemetry_df():
    return pd.DataFrame(
        {
            "Driver": ["VER", "VER"],
            "Distance": [0.0, 10.0],
            "Speed": [100, 150],
            "Throttle": [50, 100],
            "Brake": [False, False],
        }
    )


def test_save_processed_round_trip(
    tmp_path, monkeypatch, sample_laps_df, sample_weather_df, sample_telemetry_df
):
    # Redirect to a temp directory so the test never overwrites real processed data.
    monkeypatch.setattr(config, "LAPS_FILE", tmp_path / "laps.parquet")
    monkeypatch.setattr(config, "WEATHER_FILE", tmp_path / "weather.parquet")
    monkeypatch.setattr(config, "TELEMETRY_FILE", tmp_path / "telemetry.parquet")

    save_processed(sample_laps_df, sample_weather_df, sample_telemetry_df)

    loaded_laps = pd.read_parquet(config.LAPS_FILE)
    loaded_weather = pd.read_parquet(config.WEATHER_FILE)
    loaded_telemetry = pd.read_parquet(config.TELEMETRY_FILE)

    pd.testing.assert_frame_equal(loaded_laps, sample_laps_df)
    pd.testing.assert_frame_equal(loaded_weather, sample_weather_df)
    pd.testing.assert_frame_equal(loaded_telemetry, sample_telemetry_df)


def test_processed_dir_is_created():
    assert config.PROCESSED_DIR.exists()


def test_cache_dir_is_created():
    assert config.CACHE_DIR.exists()
