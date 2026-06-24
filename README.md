# Industrial Telemetry Intelligence

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

## What works (Phase 1)

- End-to-end ingestion of real FastF1 data (laps, weather, telemetry, tyre compound/life,
  sector times) for the 2024 Bahrain GP race, cached locally as parquet.
- Cleaning that drops laps with no recorded time (in/out laps, etc.).
- Feature engineering: stint-relative lap counter, rolling lap time.
- Baselines: average lap time per driver, fastest lap per driver, consistency score (lap
  time std dev), linear lap-time degradation slope per tyre stint.
- Anomaly detection: per-driver lap-time z-score with a configurable threshold.
- Plotly visualizations: lap time trend, tyre life vs lap time, speed trace, throttle/brake
  trace, two-driver comparison.
- A working Streamlit dashboard wiring all of the above together, with an explicit section
  translating each metric back to the industrial asset-monitoring framing.
- pytest coverage for ingestion round-tripping and feature engineering correctness.

## Current limitations

- The z-score anomaly flag treats every lap as independent (i.i.d.) — it does not account for
  pit stops, safety cars, weather changes, or track status, so several "anomalies" are simply
  the first lap on a new tyre compound, not genuine equipment issues.
- The degradation model is a single straight line fit per stint; real wear is often non-linear
  (fast initial drop-off, then a more gradual climb).
- Telemetry is only cached for the two drivers in `src/config.py:COMPARISON_DRIVERS` (fastest
  lap each) to keep the parquet files small — extend that list to compare more drivers.
- No predictive modelling, forecasting, or recommendation logic yet — Phase 1 is deliberately
  descriptive only, per the Karpathy "dumb baselines first" approach.
- No LLM/RAG explanation layer yet.

## Roadmap

- **Phase 2** — Better anomaly detection: condition on stint/lap context, track status, and
  weather instead of a flat z-score.
- **Phase 3** — Degradation modelling: non-linear wear curves, per-compound degradation models.
- **Phase 4** — Predictive forecasting: forecast lap time / degradation forward within a stint.
- **Phase 5** — Decision support recommendations: pit-window / maintenance-window suggestions
  derived from the forecasts.
- **Phase 6** — LLM-based explanation assistant: natural-language, evidence-backed explanations
  of detected anomalies and degradation trends.
