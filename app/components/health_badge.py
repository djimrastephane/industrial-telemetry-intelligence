"""Health badge HTML component — UI only, no analytics."""

_BADGE_CLASS = {
    "Healthy":  "badge-healthy",
    "Warning":  "badge-warning",
    "Critical": "badge-critical",
}


def health_badge_html(status: str) -> str:
    cls = _BADGE_CLASS.get(status, "badge-healthy")
    return f'<span class="health-badge {cls}">{status}</span>'
