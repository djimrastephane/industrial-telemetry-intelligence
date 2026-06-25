"""Unified operational event log for the industrial monitoring interface.

Aggregates pit events, track status changes, anomaly alerts, pit
recommendations, and race control messages into a single chronological
DataFrame.  No new analytics: every event comes from an existing module.

Column contract (EVENT_COLUMNS):
    SessionTimeSeconds  float  – session-elapsed seconds, used for ordering
    LapNumber           int    – lap context (None for fleet-wide events)
    Driver              str    – driver code (None for fleet-wide events)
    Severity            str    – "Info" / "Warning" / "Critical"
    EventType           str    – one of EVENT_* constants
    Description         str    – what happened
    Context             str    – supporting detail
    Action              str    – recommended response (empty string if none)
"""

import pandas as pd

from src.context_engine import SIGNIFICANT_TRACK_STATUSES, TRACK_STATUS_LABELS
from src.health_assessment import HEALTH_PARTIALLY_EXPLAINED, HEALTH_UNEXPLAINED

EVENT_COLUMNS = [
    "SessionTimeSeconds",
    "LapNumber",
    "Driver",
    "Severity",
    "EventType",
    "Description",
    "Context",
    "Action",
]

SEVERITY_INFO = "Info"
SEVERITY_WARNING = "Warning"
SEVERITY_CRITICAL = "Critical"

EVENT_PIT = "Pit"
EVENT_TRACK_STATUS = "Track Status"
EVENT_ANOMALY = "Anomaly"
EVENT_RECOMMENDATION = "Recommendation"
EVENT_RACE_CONTROL = "Race Control"

_ANOMALY_SEVERITY = {
    HEALTH_PARTIALLY_EXPLAINED: SEVERITY_WARNING,
    HEALTH_UNEXPLAINED: SEVERITY_CRITICAL,
}


def _pit_events(laps_df: pd.DataFrame) -> list[dict]:
    rows = []
    pit_mask = laps_df["PitInTime"].notna() | laps_df["PitOutTime"].notna()
    for _, lap in laps_df[pit_mask].iterrows():
        t = lap.get("LapStartTimeSeconds")
        if pd.isna(t):
            continue
        driver = lap["Driver"]
        lap_number = int(lap["LapNumber"]) if pd.notna(lap.get("LapNumber")) else None
        compound = lap.get("Compound", "Unknown")
        tyre_life = lap.get("TyreLife")
        age_str = f"{int(tyre_life)} laps" if pd.notna(tyre_life) else "? laps"

        if pd.notna(lap.get("PitInTime")):
            rows.append({
                "SessionTimeSeconds": float(t),
                "LapNumber": lap_number,
                "Driver": driver,
                "Severity": SEVERITY_INFO,
                "EventType": EVENT_PIT,
                "Description": f"{driver} pit in — lap {lap_number}",
                "Context": f"Tyre age before stop: {age_str} on {compound}",
                "Action": "Monitor warm-up lap pace on next stint.",
            })
        if pd.notna(lap.get("PitOutTime")):
            rows.append({
                "SessionTimeSeconds": float(t),
                "LapNumber": lap_number,
                "Driver": driver,
                "Severity": SEVERITY_INFO,
                "EventType": EVENT_PIT,
                "Description": f"{driver} pit out — lap {lap_number}",
                "Context": f"Fresh {compound} fitted",
                "Action": "Expect reduced pace for 1–2 warm-up laps.",
            })
    return rows


def _track_status_events(laps_df: pd.DataFrame) -> list[dict]:
    """Detect fleet-wide track status changes from the per-lap status column."""
    if "TrackStatus" not in laps_df.columns or "LapStartTimeSeconds" not in laps_df.columns:
        return []

    # One representative row per lap (earliest LapStartTimeSeconds to get the
    # first driver's record, which carries the session-wide track status)
    per_lap = (
        laps_df.dropna(subset=["LapStartTimeSeconds"])
        .sort_values("LapStartTimeSeconds")
        .groupby("LapNumber")
        .first()
        .reset_index()
        .sort_values("LapNumber")
    )

    rows = []
    prev_label: str | None = None
    for _, row in per_lap.iterrows():
        raw = str(row.get("TrackStatus", "")).strip()
        label = TRACK_STATUS_LABELS.get(raw[-1] if raw else "", "Unknown")
        t = row.get("LapStartTimeSeconds")
        if pd.isna(t) or label == prev_label:
            if prev_label is None:
                prev_label = label
            continue

        if prev_label is not None:
            severity = (
                SEVERITY_CRITICAL
                if label in {"Safety Car", "Red Flag"}
                else SEVERITY_WARNING
                if label in SIGNIFICANT_TRACK_STATUSES
                else SEVERITY_INFO
            )
            lap_number = int(row["LapNumber"]) if pd.notna(row.get("LapNumber")) else None
            rows.append({
                "SessionTimeSeconds": float(t),
                "LapNumber": lap_number,
                "Driver": None,
                "Severity": severity,
                "EventType": EVENT_TRACK_STATUS,
                "Description": f"Track status: {prev_label} → {label}",
                "Context": f"Lap {lap_number}",
                "Action": (
                    "Adjust pace expectations during conditions change."
                    if label in SIGNIFICANT_TRACK_STATUSES
                    else ""
                ),
            })
        prev_label = label
    return rows


def _race_control_events(race_control_df: pd.DataFrame) -> list[dict]:
    rows = []
    for _, row in race_control_df.iterrows():
        t = row.get("SessionTimeSeconds")
        if pd.isna(t):
            continue
        category = row.get("Category", "") or ""
        message = row.get("Message", "") or ""
        severity = (
            SEVERITY_WARNING
            if str(category).lower() in ("flag", "safetycar", "vsc", "drs")
            else SEVERITY_INFO
        )
        rows.append({
            "SessionTimeSeconds": float(t),
            "LapNumber": None,
            "Driver": None,
            "Severity": severity,
            "EventType": EVENT_RACE_CONTROL,
            "Description": str(message),
            "Context": f"Category: {category}" if category else "",
            "Action": "",
        })
    return rows


def _anomaly_events(assessed_anomalies: pd.DataFrame, laps_df: pd.DataFrame) -> list[dict]:
    rows = []
    # Only surface anomalies that need attention (not fully explained ones)
    actionable = assessed_anomalies[
        assessed_anomalies["HealthStatus"].isin([HEALTH_PARTIALLY_EXPLAINED, HEALTH_UNEXPLAINED])
    ]
    for _, row in actionable.iterrows():
        driver = row["Driver"]
        lap_number = row.get("LapNumber")
        health_status = row.get("HealthStatus", HEALTH_UNEXPLAINED)
        explanation = row.get("Explanation", "")
        severity = _ANOMALY_SEVERITY.get(health_status, SEVERITY_WARNING)

        t_series = laps_df.loc[
            (laps_df["Driver"] == driver) & (laps_df["LapNumber"] == lap_number),
            "LapStartTimeSeconds",
        ]
        if t_series.empty or pd.isna(t_series.iloc[0]):
            continue
        lap_int = int(lap_number) if pd.notna(lap_number) else "?"
        rows.append({
            "SessionTimeSeconds": float(t_series.iloc[0]),
            "LapNumber": lap_int if isinstance(lap_int, int) else None,
            "Driver": driver,
            "Severity": severity,
            "EventType": EVENT_ANOMALY,
            "Description": f"{driver} lap {lap_int} — {health_status}",
            "Context": explanation,
            "Action": (
                "Investigate further — no contextual explanation found."
                if health_status == HEALTH_UNEXPLAINED
                else "Monitor; partial contextual explanation available."
            ),
        })
    return rows


def _recommendation_events(recommendations: pd.DataFrame, laps_df: pd.DataFrame) -> list[dict]:
    rows = []
    actionable = recommendations[recommendations["RecommendedAction"].str.startswith("Pit")]
    for _, row in actionable.iterrows():
        driver = row["Driver"]
        action_text = row["RecommendedAction"]
        is_now = action_text == "Pit now"
        severity = SEVERITY_CRITICAL if is_now else SEVERITY_WARNING

        driver_laps = laps_df[laps_df["Driver"] == driver].dropna(subset=["LapStartTimeSeconds"])
        if driver_laps.empty:
            continue
        t = float(driver_laps["LapStartTimeSeconds"].max())
        last_lap = driver_laps["LapNumber"].max()
        compound = row.get("Compound", "Unknown")
        stint = row.get("Stint")
        slope = row.get("DegradationSecondsPerLap")
        slope_str = f"{slope:.3f} s/lap" if pd.notna(slope) else "?"
        rows.append({
            "SessionTimeSeconds": t,
            "LapNumber": int(last_lap) if pd.notna(last_lap) else None,
            "Driver": driver,
            "Severity": severity,
            "EventType": EVENT_RECOMMENDATION,
            "Description": f"{driver} — {action_text}",
            "Context": f"Stint {stint}, {compound}, degradation slope: {slope_str}",
            "Action": action_text,
        })
    return rows


def build_event_log(
    laps_df: pd.DataFrame,
    context_data: dict,
    assessed_anomalies: pd.DataFrame | None = None,
    recommendations: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the unified chronological event log from all available sources.

    Parameters
    ----------
    laps_df:
        Cleaned lap data from ``load_and_clean_all()``.
    context_data:
        Aligned context from ``align_context_to_session(load_context())``.
    assessed_anomalies:
        Output of ``assess_anomaly_health()``.  If None, anomaly rows are
        omitted.
    recommendations:
        Output of ``build_recommendations_table()``.  If None, recommendation
        rows are omitted.

    Returns
    -------
    pd.DataFrame with columns ``EVENT_COLUMNS``, sorted by
    ``SessionTimeSeconds`` ascending.
    """
    all_rows: list[dict] = []
    all_rows.extend(_pit_events(laps_df))
    all_rows.extend(_track_status_events(laps_df))
    all_rows.extend(_race_control_events(context_data.get("race_control", pd.DataFrame())))

    if assessed_anomalies is not None and not assessed_anomalies.empty:
        all_rows.extend(_anomaly_events(assessed_anomalies, laps_df))
    if recommendations is not None and not recommendations.empty:
        all_rows.extend(_recommendation_events(recommendations, laps_df))

    if not all_rows:
        return pd.DataFrame(columns=EVENT_COLUMNS)

    df = pd.DataFrame(all_rows, columns=EVENT_COLUMNS)
    return df.sort_values("SessionTimeSeconds").reset_index(drop=True)
