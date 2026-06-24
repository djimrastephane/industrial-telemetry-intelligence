"""Basic cleaning of the processed FastF1 parquet files."""

import pandas as pd

from src import config


def load_laps() -> pd.DataFrame:
    return pd.read_parquet(config.LAPS_FILE)


def load_weather() -> pd.DataFrame:
    return pd.read_parquet(config.WEATHER_FILE)


def load_telemetry() -> pd.DataFrame:
    return pd.read_parquet(config.TELEMETRY_FILE)


def clean_laps(laps_df: pd.DataFrame) -> pd.DataFrame:
    """Drop laps with no recorded lap time (in/out laps, red flags, etc.)."""
    cleaned = laps_df.dropna(subset=["LapTimeSeconds"]).copy()
    cleaned = cleaned.sort_values(["Driver", "LapNumber"]).reset_index(drop=True)
    return cleaned


def clean_weather(weather_df: pd.DataFrame) -> pd.DataFrame:
    return weather_df.dropna(how="all").reset_index(drop=True)


def clean_telemetry(telemetry_df: pd.DataFrame) -> pd.DataFrame:
    if telemetry_df.empty:
        return telemetry_df
    cleaned = telemetry_df.dropna(subset=["Speed", "Distance"]).copy()
    return cleaned.reset_index(drop=True)


def load_and_clean_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    laps = clean_laps(load_laps())
    weather = clean_weather(load_weather())
    telemetry = clean_telemetry(load_telemetry())
    return laps, weather, telemetry


if __name__ == "__main__":
    laps, weather, telemetry = load_and_clean_all()
    print(f"Clean laps: {laps.shape}")
    print(f"Clean weather: {weather.shape}")
    print(f"Clean telemetry: {telemetry.shape}")
