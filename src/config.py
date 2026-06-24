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

ANOMALY_ZSCORE_THRESHOLD = 2.5

# Phase 3: season monitoring (race rounds only, excludes pre-season testing)
SEASON_ROUNDS = list(range(1, 25))  # 2024 had 24 points-paying rounds
SEASON_LAPS_FILE = PROCESSED_DIR / "season_laps.parquet"
