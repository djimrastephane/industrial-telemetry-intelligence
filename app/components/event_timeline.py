"""Event timeline HTML component — UI only, no analytics."""

import math

import pandas as pd

_SEVERITY_DOT = {
    "Critical": "critical",
    "Warning":  "warning",
    "Info":     "info",
}

_SEVERITY_ICON = {
    "Critical": "🔴",
    "Warning":  "🟠",
    "Info":     "🔵",
}


def _fmt_time(seconds) -> str:
    if seconds is None:
        return "—"
    try:
        s = float(seconds)
        if math.isnan(s):
            return "—"
        m, sec = divmod(int(s), 60)
        return f"{m}:{sec:02d}"
    except (TypeError, ValueError):
        return "—"


def _safe_str(val) -> str:
    if val is None:
        return ""
    try:
        if isinstance(val, float) and math.isnan(val):
            return ""
    except TypeError:
        pass
    s = str(val)
    return "" if s.lower() == "none" else s


def event_timeline_html(events: pd.DataFrame, max_events: int = 15) -> str:
    if events.empty:
        return '<div class="timeline"><p style="color:#94A3B8;font-size:0.78rem;">No events.</p></div>'

    items = []
    for _, row in events.head(max_events).iterrows():
        severity = _safe_str(row.get("Severity")) or "Info"
        dot_cls  = _SEVERITY_DOT.get(severity, "info")
        icon     = _SEVERITY_ICON.get(severity, "🔵")

        time_str  = _fmt_time(row.get("SessionTimeSeconds"))
        etype     = _safe_str(row.get("EventType"))
        driver    = _safe_str(row.get("Driver"))
        driver_pfx = f"[{driver}] " if driver else ""
        desc      = _safe_str(row.get("Description"))
        action    = _safe_str(row.get("Action"))

        action_html = f'<div class="timeline-action">→ {action}</div>' if action else ""

        items.append(
            f'<div class="timeline-event {dot_cls}">'
            f'<div class="timeline-time">{time_str} · {etype}</div>'
            f'<div class="timeline-desc">{icon} {driver_pfx}{desc}</div>'
            f'{action_html}'
            f"</div>"
        )

    return '<div class="timeline">' + "".join(items) + "</div>"
