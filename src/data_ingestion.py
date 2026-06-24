"""Load one FastF1 session and extract laps, weather, and telemetry.

FastF1 telemetry stands in for high-frequency industrial sensor streams
(ESP/SCADA): laps are duty cycles, tyre stints are run-to-failure intervals,
and lap time degradation mirrors equipment performance decay.
"""

import fastf1
import pandas as pd

from src import config


def enable_cache() -> None:
    fastf1.Cache.enable_cache(str(config.CACHE_DIR))


def load_session(
    year: int = config.SEASON_YEAR,
    event: str = config.EVENT_NAME,
    session_name: str = config.SESSION_NAME,
) -> fastf1.core.Session:
    enable_cache()
    session = fastf1.get_session(year, event, session_name)
    session.load()
    return session


def get_laps_df(session: fastf1.core.Session) -> pd.DataFrame:
    """Lap times, sector times, tyre compound, and tyre life."""
    columns = [
        "Driver",
        "Team",
        "LapNumber",
        "LapTime",
        "Sector1Time",
        "Sector2Time",
        "Sector3Time",
        "Compound",
        "TyreLife",
        "Stint",
        "TrackStatus",
        "IsAccurate",
    ]
    laps = session.laps.loc[:, columns].copy()

    for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
        laps[f"{col}Seconds"] = laps[col].dt.total_seconds()

    return laps


def get_weather_df(session: fastf1.core.Session) -> pd.DataFrame:
    return session.weather_data.copy()


def get_telemetry_df(
    session: fastf1.core.Session,
    drivers: list[str] = config.COMPARISON_DRIVERS,
) -> pd.DataFrame:
    """Telemetry for each driver's fastest lap, tagged with Driver and LapNumber."""
    frames = []
    for driver in drivers:
        driver_laps = session.laps.pick_drivers(driver)
        if driver_laps.empty:
            continue
        fastest_lap = driver_laps.pick_fastest()
        telemetry = fastest_lap.get_car_data().add_distance()
        telemetry["Driver"] = driver
        telemetry["LapNumber"] = fastest_lap["LapNumber"]
        frames.append(telemetry)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def save_processed(
    laps_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    telemetry_df: pd.DataFrame,
) -> None:
    laps_df.to_parquet(config.LAPS_FILE, index=False)
    weather_df.to_parquet(config.WEATHER_FILE, index=False)
    telemetry_df.to_parquet(config.TELEMETRY_FILE, index=False)


def run_ingestion() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    session = load_session()
    laps_df = get_laps_df(session)
    weather_df = get_weather_df(session)
    telemetry_df = get_telemetry_df(session)
    save_processed(laps_df, weather_df, telemetry_df)
    return laps_df, weather_df, telemetry_df


if __name__ == "__main__":
    laps, weather, telemetry = run_ingestion()
    print(f"Laps: {laps.shape}")
    print(f"Weather: {weather.shape}")
    print(f"Telemetry: {telemetry.shape}")
