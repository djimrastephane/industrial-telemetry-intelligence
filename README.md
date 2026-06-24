# Industrial Telemetry Intelligence

[![CI](https://github.com/djimrastephane/industrial-telemetry-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/djimrastephane/industrial-telemetry-intelligence/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An **Industrial Asset Monitoring and Decision Support System**, prototyped end-to-end on
public Formula 1 telemetry data.

**This is not a sports analytics project.** It is an industrial telemetry analytics project,
demonstrated on F1 data because real ESP (electric submersible pump), SCADA, and production
well sensor data is almost never publicly available. F1 telemetry is one of the few public
datasets that shares the same shape as that data: high-frequency, multi-sensor, multi-asset,
and degradation/failure-relevant.

## Screenshots

**Race Detail (Phase 1-2)** - lap time trend, tyre degradation, telemetry traces, baseline
metrics, and the anomaly table for any two drivers in one race.

![Race Detail tab](assets/screenshots/race_detail_tab.png)

**Season Monitoring (Phase 3)** - finishing-position and speed-trap trends across a full
season, plus season KPIs and the `PositionTrendSlope` asset-health indicator.

![Season Monitoring tab](assets/screenshots/season_monitoring_tab.png)

**Fleet Monitoring (Phase 4)** - relative-pace and degradation trends for the same race across
multiple years, with a benchmarking table and year-over-year `Shift` labels.

![Fleet Monitoring tab](assets/screenshots/fleet_monitoring_tab.png)

**Arcade Replay** - standalone "digital control room" window: a car moving around the real
track shape with a live speed/throttle/brake/gear instrument cluster.

![Arcade replay window](assets/screenshots/arcade_replay.png)

## Why FastF1 as a surrogate for industrial telemetry

| F1 telemetry (FastF1) | Industrial equivalent |
|---|---|
| Lap | Duty cycle / operating interval |
| Tyre stint | Run-to-failure interval between maintenance |
| Lap time trending up across a stint | Vibration / temperature / pressure trending up as equipment wears |
| Speed, throttle, brake traces | High-frequency sensor streams (pressure, flow, current, RPM) |
| Tyre compound | Equipment configuration / operating mode |
| Lap time z-score outlier | SCADA threshold alarm |
| Driver-to-driver comparison | Asset-to-asset (well-to-well, pump-to-pump) comparison |

The goal is to build and validate the analytics pipeline — ingestion, cleaning, baselines,
anomaly detection, degradation analysis, and a dashboard — on data that is realistic in
structure and freely available, so the same pipeline can later be pointed at proprietary
ESP/SCADA/production data with minimal rework.

## Project vision

The end state is an **Industrial Telemetry Intelligence Platform** that, on F1 telemetry as
its public surrogate, ultimately supports:

- Multi-season telemetry analysis
- Fleet-level performance monitoring
- Degradation tracking
- Anomaly detection
- Predictive forecasting
- Decision support
- Interactive operational replay
- LLM-assisted explanations

That final architecture is **not** built up front. Per the Karpathy method, each phase below
is built, run on real data, and verified before the next one starts — see
[Architecture rule](#architecture-rule) and [Roadmap](#roadmap).

## Oil and gas analogy

- **ESP sensors**: An electric submersible pump streams motor temperature, vibration, intake
  pressure, and current at high frequency. Degradation shows up as a slow upward (or downward)
  drift before a hard failure — exactly the pattern lap times show across a tyre stint as the
  tyre wears.
- **SCADA**: Supervisory control systems flag readings outside a normal band, usually with
  static or rolling thresholds. The `anomaly_detection.py` z-score flag is a direct, intentionally
  simple analogue.
- **Production monitoring**: Engineers track per-well baselines (average rate, decline curve,
  consistency) the same way `baseline_models.py` tracks per-driver baselines (average lap time,
  consistency score, fastest lap).

## Skills demonstrated

- Real-world API ingestion and local caching (FastF1 → parquet)
- Data cleaning and validation on messy, real time-series data
- Feature engineering or time-series structure (stint-relative lap counters, rolling means)
- Statistical baselining before modelling (means, std dev, linear degradation slope, z-score)
- Interactive visualization (Plotly) and a working Streamlit dashboard
- Test-driven development of data pipelines with pytest
- Clear documentation of assumptions, limitations, and an honest improvement roadmap

## Project structure

```
industrial-telemetry-intelligence/
  src/
    config.py              # paths, session selection, thresholds
    data_ingestion.py       # FastF1 session -> laps/weather/telemetry -> parquet
    data_cleaning.py        # load + basic cleaning of processed parquet
    feature_engineering.py  # stint-lap counter, rolling lap time
    baseline_models.py      # avg lap time, fastest lap, consistency, degradation
    anomaly_detection.py    # per-driver z-score anomaly flag
    degradation_analysis.py # linear lap-time slope per tyre stint
    seasonal_analysis.py    # season-wide position/speed trends and KPIs
    fleet_analysis.py       # multi-year relative pace, degradation, year-over-year shift
    replay_data.py          # pure geometry/interpolation helpers for the Arcade replay
    visualisation.py        # shared Plotly figures
  app/
    streamlit_app.py        # dashboard: Race Detail + Season + Fleet Monitoring tabs
    arcade_replay.py         # standalone Arcade window: car + live HUD on the track
  notebooks/
    01_data_exploration.ipynb
    02_baseline_analysis.ipynb
    03_anomaly_detection.ipynb
  tests/
    test_data_ingestion.py
    test_feature_engineering.py
    test_seasonal_analysis.py
    test_fleet_analysis.py
    test_replay_data.py
  data/                     # raw/processed/cache - gitignored, regenerated locally
  outputs/                  # figures/reports - gitignored, regenerated locally
```

## Setup

Requires Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the pipeline

1. **Download and cache one race session** (2024 Bahrain Grand Prix, Race). This hits the
   FastF1/Ergast APIs once and caches results locally under `data/cache/` — subsequent runs
   are fast and offline.

   ```bash
   python -m src.data_ingestion
   ```

   This writes `data/processed/laps.parquet`, `weather.parquet`, and `telemetry.parquet`.

2. **Inspect cleaning, features, and baselines** (optional, sanity check):

   ```bash
   python -m src.data_cleaning
   python -m src.feature_engineering
   python -m src.baseline_models
   python -m src.degradation_analysis
   python -m src.anomaly_detection
   ```

3. **Download and cache the full season** (2024, all 24 race rounds). This loops over every
   round in `src/config.py:SEASON_ROUNDS`, skipping (with a warning) any round that fails to
   load rather than aborting the whole batch. It can take several minutes the first time.

   ```bash
   python -m src.data_ingestion season
   ```

   This writes `data/processed/season_laps.parquet`.

4. **Download and cache the multi-year fleet dataset** (Bahrain Grand Prix, every year from
   2020-2025). This loops over every year in `src/config.py:FLEET_YEARS`, skipping (with a
   warning) any year that fails to load.

   ```bash
   python -m src.data_ingestion fleet
   ```

   This writes `data/processed/fleet_laps.parquet`.

5. **Run the dashboard**:

   ```bash
   streamlit run app/streamlit_app.py
   ```

   The dashboard has three tabs: **Race Detail** (Phase 1-2, needs step 1's data),
   **Season Monitoring** (Phase 3, needs step 3's data), and **Fleet Monitoring**
   (Phase 4, needs step 4's data).

6. **Run the Arcade replay** (separate native window, needs step 1's data):

   ```bash
   python app/arcade_replay.py --driver VER
   ```

   Replays one driver's cached fastest lap as a car moving around the track, with an
   instrument cluster (speed dial, throttle dial, brake lamp, gear box) above the track and
   a checkered start/finish marker to gauge lap progress.

7. **Run the notebooks** (`notebooks/01_data_exploration.ipynb`, `02_baseline_analysis.ipynb`,
   `03_anomaly_detection.ipynb`) for the same analysis in a more exploratory format.

8. **Run tests**:

   ```bash
   pytest tests/ -v
   ```

   Tests use synthetic fixtures and a temp directory — they never overwrite the real
   downloaded data in `data/processed/`.

## Status: Phases 1-4 done, plus a v1 operational replay

Dataset so far: 2024 Bahrain Grand Prix Race session (all 20 drivers' lap/weather/telemetry
data), the full 2024 season (all 24 race rounds, ~26,600 laps), and Bahrain Grand Prix every
year from 2020-2025 (~6,500 laps).

- End-to-end ingestion of real FastF1 data (laps, weather, telemetry, tyre compound/life,
  sector times), cached locally as parquet. *(Phase 1)*
- Cleaning that drops laps with no recorded time (in/out laps, etc.). *(Phase 1)*
- Feature engineering: stint-relative lap counter, rolling lap time. *(Phase 1)*
- Plotly visualizations: lap time trend, tyre life vs lap time, speed trace, throttle/brake
  trace. *(Phase 1)*
- A working Streamlit dashboard with data-loading status and a driver selector covering all
  20 drivers. *(Phase 1 success criteria: pipeline runs end-to-end, dashboard works)*
- Driver-to-driver comparison (lap time, tyre degradation, telemetry trace overlay) for
  **any** two drivers in the session, not just a fixed pair. *(Phase 2)*
- Baselines across all drivers: average lap time, fastest lap, consistency score (lap time
  std dev), linear lap-time degradation slope per tyre stint. *(Phase 2)*
- Anomaly detection: per-driver lap-time z-score with a configurable threshold, shown as a
  dashboard table. *(Phase 2)*
- Season-wide ingestion across all 24 2024 race rounds, kept lean by extracting only
  `LapTime`, running `Position` (finishing-position proxy), and `SpeedST` (speed-trap
  telemetry-aggregate proxy) per lap instead of full per-race car telemetry. *(Phase 3)*
- Driver and team finishing-position trends across the season, plus a speed-trap trend per
  driver. *(Phase 3)*
- Season KPIs and an asset-health indicator (`PositionTrendSlope`, the season-long analogue
  of the within-stint degradation slope) per driver and per team. *(Phase 3)*
- A "Season Monitoring" dashboard tab with driver multiselect, position/speed trend charts,
  and KPI tables. *(Phase 3 success criteria: season-wide KPIs, asset health indicators)*
- Multi-year ingestion of the same race (Bahrain GP, 2020-2025), using the same lean per-lap
  summary as Phase 3 plus `PitInTime`/`PitOutTime`. *(Phase 4)*
- Cross-year benchmarking, led by `TeammateGapPct` (pace vs. your own teammate that year - same
  chassis/engine, so it cancels out almost all of the regulation/car-development effect), with
  `FieldAverageGapPct` and `FastestLapGapPct` kept as secondary, more noise-sensitive context
  columns. Raw lap times aren't directly comparable across years of changing car regulations,
  so none of these compare absolute lap times across years. *(Phase 4)*
- Reliability and operational-efficiency proxies: `RaceCompletionRatePct` (laps completed vs.
  the most anyone completed that year, standing in for true DNF/mechanical-retirement
  classification, which FastF1 doesn't reliably provide for every season), `PitStopCount`, and
  `AvgPitStopRecoveryLaps` (laps to regain pre-stop position). *(Phase 4)*
- Multi-year tyre degradation trend and `LapTimeConsistencyStd` per driver, and a `Shift` label
  (Declined / Improved / Stable) flagging year-over-year changes in `TeammateGapPct` - the
  long-horizon analogue of Phase 3's `PositionTrendSlope`. *(Phase 4)*
- A "Fleet Monitoring" dashboard tab leading with the Teammate Gap trend, secondary pace
  benchmarks behind an expander, degradation/reliability/consistency context, and a full
  benchmarking table. *(Phase 4 success criteria: multi-year comparisons, benchmarking
  dashboard)*
- A standalone Arcade replay (`app/arcade_replay.py`): a car moving around the real track
  shape for one driver's cached fastest lap, a two-line track outline, a checkered
  start/finish marker, and an instrument-cluster panel above the track (speed dial, throttle
  dial, brake lamp, gear box) - the same layout as an operator's single-asset monitoring
  screen. Pure geometry/interpolation logic (including the gauge-needle angle math) lives in
  `replay_data.py` so it's unit-tested without needing a display.
- pytest coverage for ingestion round-tripping, feature engineering, seasonal analysis, fleet
  analysis, and the replay's geometry/interpolation helpers (38 tests total).

Fastest-lap telemetry (speed/throttle/brake/X/Y position) is cached for every driver in the
Bahrain race session, so the Race Detail tab's driver selector, telemetry comparison, and the
Arcade replay all cover the full grid. `src/config.py:COMPARISON_DRIVERS` only sets the
*default pre-selected pair/driver* shown on load.

## Current limitations

- The z-score anomaly flag treats every lap as independent (i.i.d.) — it does not account for
  pit stops, safety cars, weather changes, or track status, so several "anomalies" are simply
  the first lap on a new tyre compound, not genuine equipment issues.
- The degradation model is a single straight line fit per stint; real wear is often non-linear
  (fast initial drop-off, then a more gradual climb).
- `FinishPosition` is a proxy (last recorded running position), not the official classified
  result — it won't reflect post-race penalties or disqualifications.
- Season-wide anomaly detection isn't built yet — Phase 3 only adds trend KPIs, not flagging.
- `FastestLapGapPct` (kept as a secondary context column) benchmarks against that year's single
  fastest lap, which is itself one noisy data point (e.g. an early red flag can leave an
  artificially slow "fastest" lap) - this is exactly why `TeammateGapPct` is the primary
  benchmark instead.
- `TeammateGapPct` is `NaN` for any driver whose team had no other driver that year (e.g. a
  one-off seat) - there's no fallback benchmark for those rows yet.
- `RaceCompletionRatePct` is a lap-count proxy, not true DNF/mechanical-retirement
  classification - FastF1's results data (which would carry the official reason) isn't
  reliably available for every season (`"Failed to load result data from Ergast"` shows up in
  the ingestion logs for recent seasons).
- Qualifying-session variance isn't ingested yet - only Race sessions are downloaded, so
  consistency metrics are race-lap-time based, not qualifying-lap based.
- `Shift` (year-over-year change) uses a fixed threshold (0.3 percentage points) rather than
  anything statistically derived from the data's own variance.
- `AvgPitStopRecoveryLaps` uses recorded running `Position`, not exact pit-lane stationary
  time, since that requires timing-loop data beyond the per-lap summary.
- The Arcade replay's track width is a fixed visual stand-in (FastF1 doesn't provide real
  track width), and gauge ranges (e.g. 0-350 km/h) are fixed constants, not derived per track.
- No predictive modelling, forecasting, or recommendation logic yet — by design, per the
  Karpathy "dumb baselines first" approach.
- No LLM/RAG explanation layer yet.

## Roadmap

Each phase is built and verified end-to-end on real data before the next one starts.

- **Phase 1 — Single Driver Exploration** ✅ done. Bahrain GP 2024 Race, one driver: load,
  explore, visualize, save processed data, verify data quality. *Success criteria met:
  pipeline runs without errors; basic dashboard works.*
- **Phase 2 — Race Intelligence** ✅ done. Bahrain GP 2024 Race, all drivers: driver
  comparison (lap time and telemetry), tyre degradation analysis, consistency metrics, simple
  anomaly detection. *Success criteria met: dashboard supports driver selection across the
  full grid; comparative analysis available for both lap-time metrics and telemetry traces.*
- **Phase 3 — Seasonal Monitoring** ✅ done. Entire 2024 season (24 rounds): driver trends,
  team trends, performance evolution, telemetry aggregation (speed trap). Industrial analogy:
  monitoring multiple assets over time. *Success criteria met: season-wide KPIs and an
  asset-health indicator (PositionTrendSlope) per driver and team.*
- **Phase 4 — Multi-Year Fleet Monitoring** ✅ done. Bahrain GP 2020–2025: compare operating
  behaviour across years using teammate- and field-relative pace (not raw lap times or a
  single fastest lap), reliability (`RaceCompletionRatePct`), operational efficiency
  (`PitStopCount`, `AvgPitStopRecoveryLaps`), detect long-term performance shifts (`Shift`
  label on `TeammateGapPct`), measure degradation/consistency, build a benchmarking framework.
  Industrial analogy: fleet surveillance across multiple wells, ESPs, turbines, or compressors.
  *Success criteria met: multi-year comparisons and a benchmarking dashboard tab.*
- **Phase 5 — Predictive Analytics** ⏳ planned. Forecast lap times and degradation, estimate
  performance decline, generate risk scores. Models: linear baseline, Random Forest, XGBoost.
  Success criteria: benchmark predictive performance, explain model outputs.
- **Phase 6 — Operational Intelligence Assistant** ⏳ planned. Add RAG, telemetry explanations,
  and the ability to answer operational questions (e.g. *"Why did Driver X lose performance
  after lap 32?"*). The assistant must retrieve telemetry evidence before generating an answer.
  Success criteria: evidence-backed explanations, no hallucinated conclusions.

## Architecture rule

At every phase:

1. Build the simplest version first.
2. Visualize before modelling.
3. Use baseline methods before advanced ML.
4. Validate results against domain knowledge.
5. Do not introduce complexity unless the current phase is working.
6. Keep all code production-quality and portfolio-ready.
