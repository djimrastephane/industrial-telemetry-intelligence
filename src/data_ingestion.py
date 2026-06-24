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
    drivers: list[str] | None = None,
) -> pd.DataFrame:
    """Telemetry for each driver's fastest lap, tagged with Driver and LapNumber.

    Defaults to every driver in the session so any pair can be compared in the
    dashboard, not just the COMPARISON_DRIVERS default selection.
    """
    if drivers is None:
        drivers = sorted(session.laps["Driver"].unique())

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


def get_season_lap_summary(
    session: fastf1.core.Session,
    round_number: int,
    event_name: str,
) -> pd.DataFrame:
    """Lean per-lap summary for season-wide monitoring.

    Keeps only what's needed for cross-race trends: lap time, running
    position (finishing-position proxy), and speed-trap reading (a
    telemetry-aggregate proxy that avoids downloading full car telemetry
    for every lap of every race in the season).
    """
    columns = [
        "Driver",
        "Team",
        "LapNumber",
        "LapTime",
        "Position",
        "SpeedST",
        "Compound",
        "TyreLife",
        "Stint",
    ]
    laps = session.laps.loc[:, columns].copy()
    laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()
    laps = laps.drop(columns=["LapTime"])
    laps["RoundNumber"] = round_number
    laps["EventName"] = event_name
    return laps


def build_season_dataset(
    year: int = config.SEASON_YEAR,
    rounds: list[int] = config.SEASON_ROUNDS,
) -> pd.DataFrame:
    """Load every Race session in `rounds` and concatenate the lean lap summary.

    Skips a round (with a warning) rather than failing the whole season if a
    single session can't be loaded, since this is a long-running batch job
    over the network.
    """
    enable_cache()
    frames = []
    for round_number in rounds:
        try:
            session = fastf1.get_session(year, round_number, "R")
            session.load(telemetry=False, weather=False)
            event_name = session.event["EventName"]
            frames.append(get_season_lap_summary(session, round_number, event_name))
            print(f"Round {round_number} ({event_name}): {len(frames[-1])} laps")
        except Exception as exc:
            print(f"Skipping round {round_number}: {exc}")

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


def save_season_laps(season_laps_df: pd.DataFrame) -> None:
    season_laps_df.to_parquet(config.SEASON_LAPS_FILE, index=False)


def run_ingestion() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    session = load_session()
    laps_df = get_laps_df(session)
    weather_df = get_weather_df(session)
    telemetry_df = get_telemetry_df(session)
    save_processed(laps_df, weather_df, telemetry_df)
    return laps_df, weather_df, telemetry_df


def run_season_ingestion(
    year: int = config.SEASON_YEAR,
    rounds: list[int] = config.SEASON_ROUNDS,
) -> pd.DataFrame:
    season_laps_df = build_season_dataset(year, rounds)
    save_season_laps(season_laps_df)
    return season_laps_df


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "season":
        season_laps = run_season_ingestion()
        print(f"Season laps: {season_laps.shape}")
    else:
        laps, weather, telemetry = run_ingestion()
        print(f"Laps: {laps.shape}")
        print(f"Weather: {weather.shape}")
        print(f"Telemetry: {telemetry.shape}")
