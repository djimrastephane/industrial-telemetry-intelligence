"""Operational Context Engine.

Telemetry should never be interpreted in isolation - industrial SCADA systems
combine sensor measurements with operating conditions before assessing
equipment health. This module is the Phase 9 demonstration of that
principle: it loads, aligns, and summarizes the operational context behind
a moment of telemetry, so the platform can move from

    Telemetry -> Health Assessment -> Recommendation

to

    Telemetry + Operational Context -> Health Assessment -> Recommendation

This module is deliberately domain-independent in its *architecture*: the
five public functions (`load_context`, `align_context_to_session`,
`get_context_at_timestamp`, `calculate_context_changes`,
`generate_context_summary`) don't assume the context is weather. Only the
column names in `WEATHER_VARIABLES`/`TRACK_STATUS_LABELS` and the sentences
in `INTERPRETATION_RULES` are specific to this F1 demonstration. A future
ESP/SCADA implementation would point `config.WEATHER_FILE`/`LAPS_FILE`/
`RACE_CONTROL_FILE` at different feeds with different column names and
rewrite those constants - no other module (dashboard, replay, analytics)
would need to change. See README "Operational Context Layer" for the
documented F1 -> ESP variable mapping.

No machine learning, no forecasting, no synthetic data: every number here is
either read directly from a context source, linearly interpolated between
two real samples, or a deterministic rule applied to a real measured change.
"""

import numpy as np
import pandas as pd

from src import config

# --- Weather context -------------------------------------------------------

# Numeric weather variables: interpolated between adjacent real samples for
# smooth display. WindDirection is excluded - it's circular (0/360 wrap), so
# interpolating it would be wrong; it uses the nearest real sample instead.
WEATHER_INTERPOLATED_VARIABLES = ["AirTemp", "TrackTemp", "Humidity", "WindSpeed", "Pressure"]
WEATHER_NEAREST_ONLY_VARIABLES = ["WindDirection", "Rainfall"]

# Conservative, explicitly not-fitted-to-data thresholds for "moderate" vs
# "rapid" change between two consecutive real weather samples. Simple
# constants, not derived from this race's distribution - see README
# limitations for why, and how to override them.
WEATHER_CHANGE_THRESHOLDS = {
    "AirTemp": {"moderate": 0.5, "rapid": 1.5},
    "TrackTemp": {"moderate": 0.5, "rapid": 1.5},
    "Humidity": {"moderate": 2.0, "rapid": 5.0},
    "WindSpeed": {"moderate": 2.0, "rapid": 5.0},
    "Pressure": {"moderate": 1.0, "rapid": 3.0},
}

# Dead-band below which a non-zero diff still reads as "stable" (→) rather
# than flickering ▲/▼ on floating-point/sensor noise.
WEATHER_TREND_EPSILON = {
    "AirTemp": 0.05,
    "TrackTemp": 0.05,
    "Humidity": 0.5,
    "WindSpeed": 0.3,
    "Pressure": 0.1,
}

# --- Track status ------------------------------------------------------------

# FastF1's `TrackStatus` is a string of one or more digit codes, one per
# status raised during that lap, in order. The *last* digit is the status in
# effect when the lap ended - the representative "current" status for
# context purposes.
TRACK_STATUS_LABELS = {
    "1": "Green",
    "2": "Yellow",
    "4": "Safety Car",
    "5": "Red Flag",
    "6": "Virtual Safety Car",
    "7": "VSC Ending",
}
TRACK_STATUS_UNKNOWN = "Unknown"

# How "significant" each track status is for context severity/explanations.
TRACK_STATUS_SEVERITY = {
    "Green": 0,
    "Yellow": 1,
    "VSC Ending": 1,
    "Virtual Safety Car": 2,
    "Safety Car": 2,
    "Red Flag": 2,
    TRACK_STATUS_UNKNOWN: 0,
}
SIGNIFICANT_TRACK_STATUSES = {"Yellow", "Virtual Safety Car", "VSC Ending", "Safety Car", "Red Flag"}

STATUS_LEVELS = ["Stable", "Moderately Changing", "Rapidly Changing"]
COLOR_LEVELS = ["green", "amber", "red"]

STABLE_MESSAGE = "Environmental conditions currently stable. Telemetry interpretation has higher confidence."
NO_RECENT_EVENT_MESSAGE = "No significant events"

INTERPRETATION_RULES = {
    ("TrackTemp", "▲"): "Track temperature increasing. Higher tyre degradation may occur.",
    ("TrackTemp", "▼"): "Track temperature decreasing. Tyre degradation rate may ease.",
    ("Humidity", "▲"): "Humidity increasing. Possible reduction in tyre warm-up efficiency.",
    ("Humidity", "▼"): "Humidity decreasing. Tyre warm-up may be less affected.",
    ("WindSpeed", "▲"): "Wind speed increasing. Possible effect on high-speed stability.",
    ("WindSpeed", "▼"): "Wind speed decreasing. Reduced effect on high-speed stability expected.",
    ("Rainfall", "▲"): "Rain detected. Track grip may be reduced.",
    ("Rainfall", "▼"): "Rain has stopped. Track grip may be recovering.",
}

TRACK_CONDITION_EXPLANATION = "Performance reduction expected due to track conditions."
PIT_OUT_LAP_EXPLANATION = (
    "Lap follows a pit stop onto a fresh tyre. Out-lap pace is not representative of normal performance."
)
TYRE_DEGRADATION_EXPLANATION = "Observed behaviour is consistent with tyre degradation."
ANOMALY_EXPLANATION = "Potential performance anomaly requiring further investigation."
PARTIAL_EXPLANATION = "Context partially explains observed change."

# Observed-telemetry-effect labels `generate_context_summary` understands.
OBSERVED_LOW_SPEED = "low_speed"
OBSERVED_LAPTIME_INCREASE = "laptime_increase"
OBSERVED_SPEED_ANOMALY = "speed_anomaly"


def load_context() -> dict[str, pd.DataFrame]:
    """Load the raw operational-context sources, each sorted by time.

    Currently backed by FastF1 weather, lap (tyre/track-status), and race
    control data - the public stand-ins for a SCADA operating-conditions
    feed, equipment state, and an event/alarm log. Swapping the context
    source later means changing this one function; nothing else reads these
    files directly.
    """
    weather_df = pd.read_parquet(config.WEATHER_FILE).dropna(how="all").sort_values("Time").reset_index(drop=True)

    laps_df = pd.read_parquet(config.LAPS_FILE)
    laps_df = laps_df.dropna(subset=["LapStartTimeSeconds"]).sort_values(
        ["Driver", "LapStartTimeSeconds"]
    ).reset_index(drop=True)

    if config.RACE_CONTROL_FILE.exists():
        race_control_df = pd.read_parquet(config.RACE_CONTROL_FILE)
        race_control_df = race_control_df.dropna(subset=["SessionTimeSeconds"]).sort_values(
            "SessionTimeSeconds"
        ).reset_index(drop=True)
    else:
        race_control_df = pd.DataFrame(columns=["Category", "Message", "SessionTimeSeconds"])

    return {"weather": weather_df, "laps": laps_df, "race_control": race_control_df}


def align_context_to_session(context: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Ensure every context source has a numeric, session-elapsed-seconds
    time column ready for lookups, sorted ascending. Safe to call on an
    already-aligned context (idempotent)."""
    weather_df = context["weather"].copy()
    if "TimeSeconds" not in weather_df.columns:
        if weather_df.empty:
            weather_df["TimeSeconds"] = pd.Series(dtype=float)
        else:
            weather_df["TimeSeconds"] = weather_df["Time"].dt.total_seconds()
    weather_df = weather_df.sort_values("TimeSeconds").reset_index(drop=True)

    laps_df = context["laps"].copy()
    laps_df = laps_df.sort_values(["Driver", "LapStartTimeSeconds"]).reset_index(drop=True)

    race_control_df = context["race_control"].copy()
    race_control_df = race_control_df.sort_values("SessionTimeSeconds").reset_index(drop=True)

    return {"weather": weather_df, "laps": laps_df, "race_control": race_control_df}


def _bracket(times: np.ndarray, session_time_seconds: float) -> tuple[int, int, float]:
    """Bracketing sample indices and interpolation fraction for
    `session_time_seconds`, clamped to the available range."""
    clamped = min(max(session_time_seconds, times[0]), times[-1])
    idx_current = int(np.searchsorted(times, clamped, side="right") - 1)
    idx_current = min(max(idx_current, 0), len(times) - 1)
    idx_next = min(idx_current + 1, len(times) - 1)
    span = times[idx_next] - times[idx_current]
    fraction = 0.0 if span == 0 else (clamped - times[idx_current]) / span
    return idx_current, idx_next, fraction


def _track_status_label(track_status: object) -> str:
    if track_status is None or (isinstance(track_status, float) and pd.isna(track_status)):
        return TRACK_STATUS_UNKNOWN
    text = str(track_status).strip()
    if not text:
        return TRACK_STATUS_UNKNOWN
    return TRACK_STATUS_LABELS.get(text[-1], TRACK_STATUS_UNKNOWN)


def _weather_at(weather_df: pd.DataFrame, session_time_seconds: float) -> dict:
    if weather_df.empty:
        return {}
    times = weather_df["TimeSeconds"].to_numpy()
    idx_current, idx_next, fraction = _bracket(times, session_time_seconds)
    row_current, row_next = weather_df.iloc[idx_current], weather_df.iloc[idx_next]

    result = {"WeatherLastUpdateSeconds": float(times[idx_current])}
    for var in WEATHER_INTERPOLATED_VARIABLES:
        if var not in weather_df.columns:
            continue
        current_value, next_value = row_current[var], row_next[var]
        if pd.isna(current_value) and pd.isna(next_value):
            continue
        if pd.isna(current_value):
            result[var] = float(next_value)
        elif pd.isna(next_value):
            result[var] = float(current_value)
        else:
            result[var] = float(current_value) + fraction * (float(next_value) - float(current_value))

    for var in WEATHER_NEAREST_ONLY_VARIABLES:
        if var not in weather_df.columns:
            continue
        value = row_current[var] if not pd.isna(row_current[var]) else row_next[var]
        if pd.isna(value):
            continue
        result[var] = bool(value) if var == "Rainfall" else int(value)

    return result


def _current_lap_for_driver(laps_df: pd.DataFrame, driver: str, session_time_seconds: float):
    driver_laps = laps_df[laps_df["Driver"] == driver]
    if driver_laps.empty:
        return None
    times = driver_laps["LapStartTimeSeconds"].to_numpy()
    idx = int(np.searchsorted(times, session_time_seconds, side="right") - 1)
    idx = min(max(idx, 0), len(driver_laps) - 1)
    return driver_laps.iloc[idx]


def _recent_event(race_control_df: pd.DataFrame, session_time_seconds: float) -> dict | None:
    if race_control_df.empty:
        return None
    window_start = session_time_seconds - config.CONTEXT_RECENT_EVENT_WINDOW_SECONDS
    candidates = race_control_df[
        (race_control_df["SessionTimeSeconds"] <= session_time_seconds)
        & (race_control_df["SessionTimeSeconds"] >= window_start)
    ]
    if candidates.empty:
        return None
    latest = candidates.iloc[-1]
    return {"Category": latest.get("Category"), "Message": latest.get("Message")}


def get_context_at_timestamp(
    context: dict[str, pd.DataFrame], session_time_seconds: float, driver: str
) -> dict:
    """The full operational context for `driver` at `session_time_seconds`
    (absolute session-elapsed seconds - the same clock as telemetry's
    `SessionTime` column): weather (interpolated), this driver's current
    tyre/track state, and the most recent race control event, if any within
    `config.CONTEXT_RECENT_EVENT_WINDOW_SECONDS`.

    Missing sources are handled gracefully: an empty/missing weather, laps,
    or race-control table simply omits the corresponding fields rather than
    raising.
    """
    result: dict = {"SessionTimeSeconds": float(session_time_seconds)}
    result.update(_weather_at(context["weather"], session_time_seconds))

    current_lap = _current_lap_for_driver(context["laps"], driver, session_time_seconds)
    if current_lap is not None:
        result["LapNumber"] = current_lap.get("LapNumber")
        result["Compound"] = current_lap.get("Compound")
        result["TyreLife"] = current_lap.get("TyreLife")
        result["FreshTyre"] = current_lap.get("FreshTyre")
        result["Stint"] = current_lap.get("Stint")
        result["TrackStatus"] = _track_status_label(current_lap.get("TrackStatus"))
        pit_in, pit_out = current_lap.get("PitInTime"), current_lap.get("PitOutTime")
        result["PitThisLap"] = bool(pd.notna(pit_in) or pd.notna(pit_out))
    else:
        result["TrackStatus"] = TRACK_STATUS_UNKNOWN
        result["PitThisLap"] = False

    event = _recent_event(context["race_control"], session_time_seconds)
    if event is not None:
        result["RecentEvent"] = event["Message"]
        result["RecentEventCategory"] = event["Category"]
    elif result.get("PitThisLap"):
        result["RecentEvent"] = "Pit stop (in/out)"
        result["RecentEventCategory"] = "Pit"
    else:
        result["RecentEvent"] = NO_RECENT_EVENT_MESSAGE
        result["RecentEventCategory"] = None

    return result


def _trend_arrow(diff: float, epsilon: float) -> str:
    if diff > epsilon:
        return "▲"
    if diff < -epsilon:
        return "▼"
    return "→"


def calculate_context_changes(
    context: dict[str, pd.DataFrame], session_time_seconds: float, driver: str
) -> dict:
    """Current vs. previous *real* sample for weather and tyre state - not
    two interpolated values a fraction of a second apart, which is what
    keeps trend arrows from flickering on interpolation noise.

    Returns a dict keyed by variable name (`AirTemp`, `TrackTemp`,
    `Humidity`, `WindSpeed`, `Pressure`, `TyreLife`, `Compound`,
    `TrackStatus`), each with `current`, `previous`, `diff` (numeric
    variables only), and `trend`. A variable is omitted if there isn't yet
    an earlier real sample to compare against, or either value is missing.
    """
    changes: dict = {}

    weather_df = context["weather"]
    if not weather_df.empty:
        times = weather_df["TimeSeconds"].to_numpy()
        clamped = min(max(session_time_seconds, times[0]), times[-1])
        idx_current = int(np.searchsorted(times, clamped, side="right") - 1)
        idx_current = min(max(idx_current, 0), len(weather_df) - 1)
        if idx_current > 0:
            current_row = weather_df.iloc[idx_current]
            previous_row = weather_df.iloc[idx_current - 1]
            for var in WEATHER_INTERPOLATED_VARIABLES:
                if var not in weather_df.columns:
                    continue
                current_value, previous_value = current_row[var], previous_row[var]
                if pd.isna(current_value) or pd.isna(previous_value):
                    continue
                diff = float(current_value) - float(previous_value)
                changes[var] = {
                    "current": float(current_value),
                    "previous": float(previous_value),
                    "diff": diff,
                    "trend": _trend_arrow(diff, WEATHER_TREND_EPSILON.get(var, 0.01)),
                }

    driver_laps = context["laps"][context["laps"]["Driver"] == driver]
    if not driver_laps.empty:
        times = driver_laps["LapStartTimeSeconds"].to_numpy()
        idx_current = int(np.searchsorted(times, session_time_seconds, side="right") - 1)
        idx_current = min(max(idx_current, 0), len(driver_laps) - 1)
        if idx_current > 0:
            current_lap = driver_laps.iloc[idx_current]
            previous_lap = driver_laps.iloc[idx_current - 1]

            current_life, previous_life = current_lap.get("TyreLife"), previous_lap.get("TyreLife")
            if pd.notna(current_life) and pd.notna(previous_life):
                diff = float(current_life) - float(previous_life)
                trend = "▼ Reset" if diff < 0 else ("▲ Aging" if diff > 0 else "→ Stable")
                changes["TyreLife"] = {
                    "current": float(current_life),
                    "previous": float(previous_life),
                    "diff": diff,
                    "trend": trend,
                }

            current_compound, previous_compound = current_lap.get("Compound"), previous_lap.get("Compound")
            changes["Compound"] = {
                "current": current_compound,
                "previous": previous_compound,
                "changed": current_compound != previous_compound,
            }

            current_status = _track_status_label(current_lap.get("TrackStatus"))
            previous_status = _track_status_label(previous_lap.get("TrackStatus"))
            changes["TrackStatus"] = {
                "current": current_status,
                "previous": previous_status,
                "changed": current_status != previous_status,
            }

    return changes


def generate_context_summary(
    context_now: dict, changes: dict, observed_effect: str | None = None
) -> dict:
    """Deterministic status/colour/interpretation/confidence from
    `get_context_at_timestamp`'s and `calculate_context_changes`'s output.
    Only describes what the measured context supports - no speculation
    beyond `INTERPRETATION_RULES`.

    `observed_effect` is an optional telemetry-side signal
    (`OBSERVED_LOW_SPEED`, `OBSERVED_LAPTIME_INCREASE`, or
    `OBSERVED_SPEED_ANOMALY`) describing what was *seen* in telemetry, so the
    same context can be checked for whether it plausibly explains it. With
    no `observed_effect`, only the context-only status/interpretation is
    produced and `confidence` is `None`.
    """
    severity = TRACK_STATUS_SEVERITY.get(context_now.get("TrackStatus", TRACK_STATUS_UNKNOWN), 0)
    sentences = []

    track_status = context_now.get("TrackStatus", TRACK_STATUS_UNKNOWN)
    if track_status in SIGNIFICANT_TRACK_STATUSES:
        sentences.append(f"Track status: {track_status}. {TRACK_CONDITION_EXPLANATION}")

    for variable, change in changes.items():
        if variable in ("Compound", "TrackStatus"):
            continue
        if variable == "Rainfall":
            if change["trend"] != "→":
                sentences.append(INTERPRETATION_RULES[(variable, change["trend"])])
                severity = max(severity, 1)
            continue
        if variable == "TyreLife":
            continue  # tyre-age interpretation is handled via `observed_effect` below

        thresholds = WEATHER_CHANGE_THRESHOLDS.get(variable)
        if thresholds is None:
            continue
        abs_diff = abs(change["diff"])
        if abs_diff >= thresholds["rapid"]:
            severity = max(severity, 2)
        elif abs_diff >= thresholds["moderate"]:
            severity = max(severity, 1)

        if abs_diff >= thresholds["moderate"] and change["trend"] in ("▲", "▼"):
            rule = INTERPRETATION_RULES.get((variable, change["trend"]))
            if rule:
                sentences.append(rule)

    confidence = None
    if observed_effect is not None:
        tyre_life = context_now.get("TyreLife")
        if track_status in SIGNIFICANT_TRACK_STATUSES:
            confidence = "High"
            if TRACK_CONDITION_EXPLANATION not in " ".join(sentences):
                sentences.append(TRACK_CONDITION_EXPLANATION)
        elif observed_effect in (OBSERVED_LOW_SPEED, OBSERVED_SPEED_ANOMALY) and context_now.get("PitThisLap"):
            # A pit out-lap on a cold, fresh tyre is a known, direct cause of a
            # slow lap - distinct from gradual tyre wear (TyreLife threshold
            # below), which is why this check comes first. Found via a real
            # validation pass (BOT lap 13 of the 2024 Bahrain race, the same
            # lap Phase 6's assistant explains) before this rule existed: the
            # engine fell through to "Low confidence: anomaly" for a lap that
            # was fully explained by the pit stop it already had context for.
            confidence = "High"
            sentences.append(PIT_OUT_LAP_EXPLANATION)
        elif observed_effect in (OBSERVED_LOW_SPEED, OBSERVED_SPEED_ANOMALY) and (
            tyre_life is not None and pd.notna(tyre_life) and tyre_life >= config.CONTEXT_TYRE_LIFE_HIGH_THRESHOLD
        ):
            confidence = "High"
            sentences.append(TYRE_DEGRADATION_EXPLANATION)
        elif observed_effect in (OBSERVED_LOW_SPEED, OBSERVED_SPEED_ANOMALY) and track_status == "Green":
            confidence = "Low"
            sentences.append(ANOMALY_EXPLANATION)
        else:
            confidence = "Medium"
            sentences.append(PARTIAL_EXPLANATION)

    if not sentences:
        sentences = [STABLE_MESSAGE]

    return {
        "status": STATUS_LEVELS[severity],
        "color": COLOR_LEVELS[severity],
        "interpretation": sentences,
        "confidence": confidence,
    }


if __name__ == "__main__":
    context = align_context_to_session(load_context())
    sample_driver = context["laps"]["Driver"].iloc[len(context["laps"]) // 2]
    sample_time = context["laps"]["LapStartTimeSeconds"].iloc[len(context["laps"]) // 2]

    current = get_context_at_timestamp(context, sample_time, sample_driver)
    changes = calculate_context_changes(context, sample_time, sample_driver)
    summary = generate_context_summary(current, changes)

    print(f"Driver: {sample_driver}  SessionTimeSeconds: {sample_time:.1f}")
    print("Context:", current)
    print("Changes:", changes)
    print("Summary:", summary)
