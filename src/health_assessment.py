"""Phase 10: Context-Aware Health Assessment.

    Telemetry + Operational Context -> Health Assessment -> Recommendation

This module is the "Health Assessment" step in that diagram. Phase 2's
z-score anomaly flag is context-blind: it raises an alarm on every lap that
deviates from a driver's own baseline, regardless of *why*. This module
re-scores each of those alarms through Phase 9's context engine, so a lap
that's fully explained by context (a pit out-lap, tyre degradation, a
significant track status) is downgraded from "investigate this" to
"explained", while one with no contextual explanation is escalated instead.

No new model: every input already exists in `anomaly_detection.py` (Phase 2)
and `context_engine.py` (Phase 9). This module only combines them.
"""

import pandas as pd

from src.anomaly_detection import get_anomaly_table
from src.context_engine import OBSERVED_LAPTIME_INCREASE, calculate_context_changes, generate_context_summary, get_context_at_timestamp

HEALTH_EXPLAINED = "Explained"
HEALTH_PARTIALLY_EXPLAINED = "Partially Explained"
HEALTH_UNEXPLAINED = "Unexplained - Investigate"

CONFIDENCE_TO_HEALTH = {
    "High": HEALTH_EXPLAINED,
    "Medium": HEALTH_PARTIALLY_EXPLAINED,
    "Low": HEALTH_UNEXPLAINED,
}


def assess_anomaly_health(laps_df: pd.DataFrame, context: dict) -> pd.DataFrame:
    """Re-score every *slow*-lap anomaly (Phase 2) through operational
    context (Phase 9): each flagged anomaly gets a `HealthStatus`,
    `Confidence`, and `Explanation`, instead of a single undifferentiated
    "anomaly" flag.

    Only anomalies slower than the driver's own baseline (positive z-score)
    are assessed - a lap unusually *fast* isn't a health concern, so there's
    nothing to explain or escalate. Laps missing `LapStartTimeSeconds` (no
    context to align to) are conservatively marked Unexplained rather than
    silently dropped.
    """
    anomalies = get_anomaly_table(laps_df)
    slow_anomalies = anomalies[anomalies["LapTimeZScore"] > 0].copy()
    if slow_anomalies.empty:
        return slow_anomalies.assign(HealthStatus=[], Confidence=[], Explanation=[])

    statuses, confidences, explanations = [], [], []
    for _, row in slow_anomalies.iterrows():
        driver, lap_number = row["Driver"], row["LapNumber"]
        lap_start = laps_df.loc[
            (laps_df["Driver"] == driver) & (laps_df["LapNumber"] == lap_number), "LapStartTimeSeconds"
        ]
        if lap_start.empty or pd.isna(lap_start.iloc[0]):
            statuses.append(HEALTH_UNEXPLAINED)
            confidences.append(None)
            explanations.append("No context available for this lap.")
            continue

        session_time = float(lap_start.iloc[0])
        context_now = get_context_at_timestamp(context, session_time, driver)
        changes = calculate_context_changes(context, session_time, driver)
        summary = generate_context_summary(context_now, changes, OBSERVED_LAPTIME_INCREASE)

        confidence = summary["confidence"]
        statuses.append(CONFIDENCE_TO_HEALTH.get(confidence, HEALTH_UNEXPLAINED))
        confidences.append(confidence)
        explanations.append(" ".join(summary["interpretation"]))

    slow_anomalies["HealthStatus"] = statuses
    slow_anomalies["Confidence"] = confidences
    slow_anomalies["Explanation"] = explanations
    return slow_anomalies


def health_summary(assessed_df: pd.DataFrame) -> dict:
    """Counts per `HealthStatus`, plus the noise-reduction percentage - the
    share of raw anomaly-detector alarms that context explains away."""
    total = len(assessed_df)
    if total == 0:
        return {"TotalFlagged": 0, "Explained": 0, "PartiallyExplained": 0, "Unexplained": 0, "NoiseReductionPct": 0.0}

    explained = int((assessed_df["HealthStatus"] == HEALTH_EXPLAINED).sum())
    partially = int((assessed_df["HealthStatus"] == HEALTH_PARTIALLY_EXPLAINED).sum())
    unexplained = int((assessed_df["HealthStatus"] == HEALTH_UNEXPLAINED).sum())

    return {
        "TotalFlagged": total,
        "Explained": explained,
        "PartiallyExplained": partially,
        "Unexplained": unexplained,
        "NoiseReductionPct": round(100.0 * explained / total, 1),
    }


if __name__ == "__main__":
    from src.context_engine import align_context_to_session, load_context
    from src.data_cleaning import load_and_clean_all

    laps, _, _ = load_and_clean_all()
    context = align_context_to_session(load_context())
    assessed = assess_anomaly_health(laps, context)

    print(
        assessed[["Driver", "LapNumber", "LapTimeZScore", "HealthStatus", "Confidence", "Explanation"]].to_string()
    )
    print(health_summary(assessed))
