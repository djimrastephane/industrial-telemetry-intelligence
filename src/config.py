"""Central configuration for the telemetry intelligence pipeline."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"

for directory in (RAW_DIR, PROCESSED_DIR, CACHE_DIR, FIGURES_DIR, REPORTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# Session selection for Phase 1
SEASON_YEAR = 2024
EVENT_NAME = "Bahrain Grand Prix"
SESSION_NAME = "R"  # Race

# Default driver pair pre-selected in the dashboard (3-letter FastF1 codes).
# Telemetry is cached for every driver in the session, not just this pair.
COMPARISON_DRIVERS = ["VER", "LEC"]

# Processed parquet filenames
LAPS_FILE = PROCESSED_DIR / "laps.parquet"
WEATHER_FILE = PROCESSED_DIR / "weather.parquet"
TELEMETRY_FILE = PROCESSED_DIR / "telemetry.parquet"
RACE_CONTROL_FILE = PROCESSED_DIR / "race_control.parquet"

ANOMALY_ZSCORE_THRESHOLD = 2.5

# Phase 3: season monitoring (race rounds only, excludes pre-season testing)
SEASON_ROUNDS = list(range(1, 25))  # 2024 had 24 points-paying rounds
SEASON_LAPS_FILE = PROCESSED_DIR / "season_laps.parquet"

# Phase 4: multi-year fleet monitoring (same race, every year, to compare across time)
FLEET_EVENT_NAME = "Bahrain Grand Prix"
FLEET_YEARS = list(range(2020, 2026))
FLEET_LAPS_FILE = PROCESSED_DIR / "fleet_laps.parquet"

# Phase 6: operational intelligence assistant. Runs against a local Ollama
# server (not a hosted API) so this approach stays usable with confidential
# real ESP/SCADA data that can't be sent to a third-party LLM provider.
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:7b-instruct"
ASSISTANT_EVIDENCE_WINDOW = 5  # laps either side of the lap in question

# Phase 7: decision support. A pit/maintenance window is recommended once the
# linear degradation forecast projects lap time crossing this far above the
# stint's own starting pace (a fixed % threshold, like ANOMALY_ZSCORE_THRESHOLD
# above - simple and configurable, not derived from the data; see README limitations).
DECISION_PIT_THRESHOLD_PCT = 5.0
DECISION_MAX_HORIZON_LAPS = 10

# Phase 8: multi-driver Arcade replay. Default set of drivers shown together
# on track; telemetry is cached for the full grid, so any 3-letter code works.
REPLAY_DRIVERS = ["VER", "LEC", "NOR"]

# Phase 9: operational context engine. How far back (in session-elapsed
# seconds) a race control message still counts as a "recent event" at a
# given moment - a simple fixed window, not derived from the data.
CONTEXT_RECENT_EVENT_WINDOW_SECONDS = 120.0

# Tyre age (laps) above which "lower than expected speed" is attributed to
# tyre degradation rather than flagged as an unexplained anomaly. A
# conservative, explicitly not-fitted constant - see README limitations.
CONTEXT_TYRE_LIFE_HIGH_THRESHOLD = 15
