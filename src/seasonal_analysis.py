"""Phase 3: season-wide monitoring.

Within one race, a tyre stint is the run-to-failure interval (see
degradation_analysis.py). Across a season, a *driver/team* is the asset and
each race is one operating cycle. FinishPosition is the per-cycle KPI, and
its slope across the season is the same kind of long-term drift indicator
as a stint's lap-time degradation slope - just at a longer time horizon.
"""

import numpy as np
import pandas as pd


def finishing_position_per_race(season_laps_df: pd.DataFrame) -> pd.DataFrame:
    """Last recorded running position for each driver in each round, used as a
    finishing-position proxy (DNFs simply stop contributing laps)."""
    idx = season_laps_df.groupby(["RoundNumber", "Driver"])["LapNumber"].idxmax()
    columns = ["RoundNumber", "EventName", "Driver", "Team", "Position"]
    result = season_laps_df.loc[idx, columns].rename(columns={"Position": "FinishPosition"})
    return result.sort_values(["Driver", "RoundNumber"]).reset_index(drop=True)


def driver_position_trend(season_laps_df: pd.DataFrame) -> pd.DataFrame:
    return finishing_position_per_race(season_laps_df)


def team_position_trend(season_laps_df: pd.DataFrame) -> pd.DataFrame:
    """Average of each team's drivers' finishing positions, per round."""
    positions = finishing_position_per_race(season_laps_df)
    return (
        positions.groupby(["RoundNumber", "EventName", "Team"])["FinishPosition"]
        .mean()
        .reset_index()
        .sort_values(["Team", "RoundNumber"])
        .reset_index(drop=True)
    )


def speed_trap_trend(season_laps_df: pd.DataFrame) -> pd.DataFrame:
    """Average speed-trap reading per driver per round (telemetry-aggregate KPI)."""
    return (
        season_laps_df.groupby(["RoundNumber", "EventName", "Driver", "Team"])["SpeedST"]
        .mean()
        .reset_index(name="AvgSpeedTrap")
        .sort_values(["Driver", "RoundNumber"])
        .reset_index(drop=True)
    )


def _trend_slope(group: pd.DataFrame, x_col: str, y_col: str) -> float:
    if len(group) < 2:
        return np.nan
    slope, _ = np.polyfit(group[x_col], group[y_col], 1)
    return slope


def season_driver_kpis(season_laps_df: pd.DataFrame) -> pd.DataFrame:
    """Season-wide KPIs and an asset-health indicator per driver.

    PositionTrendSlope is the finishing-position slope across rounds:
    positive means finishing positions are getting worse (numerically
    higher) as the season goes on, negative means improving.
    """
    positions = finishing_position_per_race(season_laps_df)
    speed = speed_trap_trend(season_laps_df)

    records = []
    for driver, group in positions.groupby("Driver"):
        group = group.sort_values("RoundNumber")
        speed_avg = speed.loc[speed["Driver"] == driver, "AvgSpeedTrap"].mean()
        records.append(
            {
                "Driver": driver,
                "Team": group["Team"].iloc[-1],
                "RacesCompleted": len(group),
                "AvgFinishPosition": group["FinishPosition"].mean(),
                "BestFinish": group["FinishPosition"].min(),
                "WorstFinish": group["FinishPosition"].max(),
                "AvgSpeedTrap": speed_avg,
                "PositionTrendSlope": _trend_slope(group, "RoundNumber", "FinishPosition"),
            }
        )

    return pd.DataFrame(records).sort_values("AvgFinishPosition").reset_index(drop=True)


def season_team_kpis(season_laps_df: pd.DataFrame) -> pd.DataFrame:
    team_positions = team_position_trend(season_laps_df)

    records = []
    for team, group in team_positions.groupby("Team"):
        group = group.sort_values("RoundNumber")
        records.append(
            {
                "Team": team,
                "RacesCompleted": len(group),
                "AvgFinishPosition": group["FinishPosition"].mean(),
                "PositionTrendSlope": _trend_slope(group, "RoundNumber", "FinishPosition"),
            }
        )

    return pd.DataFrame(records).sort_values("AvgFinishPosition").reset_index(drop=True)


if __name__ == "__main__":
    from src.data_cleaning import load_and_clean_season

    season_laps = load_and_clean_season()
    print("--- driver KPIs ---")
    print(season_driver_kpis(season_laps).head(10))
    print("\n--- team KPIs ---")
    print(season_team_kpis(season_laps))
