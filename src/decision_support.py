"""Phase 7: decision support recommendations.

Turns the existing degradation forecast (Phase 5) into an actionable
recommendation: pit/intervene now, pit within N laps, or no action needed -
the same shape as recommending a maintenance window for a pump or ESP once
its projected wear crosses an acceptable threshold, rather than only
reporting the wear trend and leaving the decision to a human every time.

Deliberately reuses degradation_analysis.degradation_per_stint's existing
linear fit instead of a new model - the smallest version that turns a
forecast into a decision.
"""

import math

import pandas as pd

from src import config
from src.degradation_analysis import degradation_per_stint
from src.feature_engineering import add_stint_lap_number
from src.predictive_models import degradation_risk_scores


def project_lap_to_threshold(
    intercept: float,
    slope: float,
    current_stint_lap: int,
    threshold_pct: float = config.DECISION_PIT_THRESHOLD_PCT,
    max_horizon: int = config.DECISION_MAX_HORIZON_LAPS,
) -> int | None:
    """The stint lap at which the linear degradation forecast crosses
    `threshold_pct` above the stint's own starting pace (`intercept`).

    Returns None if lap times aren't projected to degrade (slope <= 0) or
    the crossing point is beyond `max_horizon` laps from now - in both
    cases there's nothing actionable to recommend yet.
    """
    if slope <= 0:
        return None

    threshold_value = intercept * (1 + threshold_pct / 100)
    # Rounded to absorb float noise (e.g. 100 * 1.1 != 110.0 exactly), which
    # would otherwise push an exact crossing lap up by one via math.ceil.
    crossing_lap = round((threshold_value - intercept) / slope, 6)

    if crossing_lap <= current_stint_lap:
        return current_stint_lap
    if crossing_lap > current_stint_lap + max_horizon:
        return None
    return math.ceil(crossing_lap)


def recommend_pit_window(
    laps_df: pd.DataFrame,
    driver: str,
    stint: int,
    threshold_pct: float = config.DECISION_PIT_THRESHOLD_PCT,
    max_horizon: int = config.DECISION_MAX_HORIZON_LAPS,
) -> dict:
    """One driver/stint's recommendation, derived from the existing
    degradation_per_stint fit - "pit now", "pit within N laps", or "no
    action needed", each with the data behind it."""
    df = add_stint_lap_number(laps_df)
    stint_laps = df[(df["Driver"] == driver) & (df["Stint"] == stint)]
    if stint_laps.empty:
        return {"driver": driver, "stint": stint, "found": False, "reason": "No data for this driver/stint."}

    current_stint_lap = int(stint_laps["StintLap"].max())

    fits = degradation_per_stint(laps_df)
    fit = fits[(fits["Driver"] == driver) & (fits["Stint"] == stint)]
    if fit.empty:
        return {"driver": driver, "stint": stint, "found": False, "reason": "Not enough laps to fit a degradation slope."}

    slope = fit["DegradationSecondsPerLap"].iloc[0]
    intercept = fit["StartingLapTimeEstimate"].iloc[0]
    compound = fit["Compound"].iloc[0]

    crossing_lap = project_lap_to_threshold(intercept, slope, current_stint_lap, threshold_pct, max_horizon)

    if slope <= 0:
        action = "No action needed (lap times stable or improving)"
    elif crossing_lap is None:
        action = f"No action needed within the next {max_horizon} laps"
    elif crossing_lap <= current_stint_lap:
        action = "Pit now"
    else:
        action = f"Pit within {crossing_lap - current_stint_lap} laps (by stint lap {crossing_lap})"

    return {
        "driver": driver,
        "stint": stint,
        "found": True,
        "compound": compound,
        "current_stint_lap": current_stint_lap,
        "degradation_seconds_per_lap": slope,
        "starting_lap_time_estimate": intercept,
        "threshold_pct": threshold_pct,
        "projected_crossing_stint_lap": crossing_lap,
        "recommended_action": action,
    }


def build_recommendations_table(
    laps_df: pd.DataFrame,
    threshold_pct: float = config.DECISION_PIT_THRESHOLD_PCT,
    max_horizon: int = config.DECISION_MAX_HORIZON_LAPS,
    min_stint_laps: int = 3,
) -> pd.DataFrame:
    """Recommendation for every driver/stint with enough laps to fit a
    degradation slope, alongside the existing Phase 5 risk category for
    cross-reference (a high-risk stint should generally also show up here
    with a "pit soon" recommendation, not as a contradiction)."""
    fits = degradation_per_stint(laps_df)
    fits = fits[fits["Laps"] >= min_stint_laps]

    risk = degradation_risk_scores(laps_df, laps_ahead=5, min_stint_laps=min_stint_laps)
    risk_lookup = risk.set_index(["Driver", "Stint"])["RiskCategory"].to_dict() if not risk.empty else {}

    records = []
    for _, row in fits.iterrows():
        rec = recommend_pit_window(laps_df, row["Driver"], row["Stint"], threshold_pct, max_horizon)
        if not rec["found"]:
            continue
        rec["risk_category"] = risk_lookup.get((row["Driver"], row["Stint"]), "N/A")
        records.append(rec)

    if not records:
        return pd.DataFrame()

    table = pd.DataFrame(records).rename(
        columns={
            "driver": "Driver", "stint": "Stint", "compound": "Compound",
            "current_stint_lap": "CurrentStintLap",
            "degradation_seconds_per_lap": "DegradationSecondsPerLap",
            "projected_crossing_stint_lap": "ProjectedCrossingStintLap",
            "recommended_action": "RecommendedAction", "risk_category": "RiskCategory",
        }
    )
    return table[
        [
            "Driver", "Stint", "Compound", "CurrentStintLap", "DegradationSecondsPerLap",
            "ProjectedCrossingStintLap", "RiskCategory", "RecommendedAction",
        ]
    ].sort_values(["Driver", "Stint"]).reset_index(drop=True)


if __name__ == "__main__":
    from src.data_cleaning import load_and_clean_all

    laps, _, _ = load_and_clean_all()
    table = build_recommendations_table(laps)
    print(table.to_string())
