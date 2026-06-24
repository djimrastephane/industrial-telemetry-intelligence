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
    visualisation.py        # shared Plotly figures
  app/
    streamlit_app.py        # Phase 1 dashboard
  notebooks/
    01_data_exploration.ipynb
    02_baseline_analysis.ipynb
    03_anomaly_detection.ipynb
  tests/
    test_data_ingestion.py
    test_feature_engineering.py
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

3. **Run the dashboard**:

   ```bash
   streamlit run app/streamlit_app.py
   ```

4. **Run the notebooks** (`notebooks/01_data_exploration.ipynb`, `02_baseline_analysis.ipynb`,
   `03_anomaly_detection.ipynb`) for the same analysis in a more exploratory format.

5. **Run tests**:

   ```bash
   pytest tests/ -v
   ```

   Tests use synthetic fixtures and a temp directory — they never overwrite the real
   downloaded data in `data/processed/`.

## Status: Phase 1 and Phase 2 done

Dataset so far: 2024 Bahrain Grand Prix, Race session, all 20 drivers' lap/weather/telemetry
data.

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
- pytest coverage for ingestion round-tripping and feature engineering correctness.

Fastest-lap telemetry (speed/throttle/brake) is cached for every driver in the session, so
the dashboard's driver selector and telemetry comparison both cover the full grid.
`src/config.py:COMPARISON_DRIVERS` only sets the *default pre-selected pair* shown on load.

## Current limitations

- The z-score anomaly flag treats every lap as independent (i.i.d.) — it does not account for
  pit stops, safety cars, weather changes, or track status, so several "anomalies" are simply
  the first lap on a new tyre compound, not genuine equipment issues.
- The degradation model is a single straight line fit per stint; real wear is often non-linear
  (fast initial drop-off, then a more gradual climb).
- No predictive modelling, forecasting, or recommendation logic yet — by design, per the
  Karpathy "dumb baselines first" approach.
- No multi-race, multi-season, or multi-year data yet.
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
- **Phase 3 — Seasonal Monitoring** ⏳ planned. Entire 2024 season: driver trends, team trends,
  performance evolution, telemetry aggregation. Industrial analogy: monitoring multiple assets
  over time. Success criteria: season-wide KPIs, asset health indicators.
- **Phase 4 — Multi-Year Fleet Monitoring** ⏳ planned. Bahrain GP 2020–2025: compare operating
  behaviour across years, detect long-term performance shifts, measure degradation patterns,
  build a benchmarking framework. Industrial analogy: fleet surveillance across multiple wells,
  ESPs, turbines, or compressors. Success criteria: multi-year comparisons, benchmarking
  dashboard.
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
