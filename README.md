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

## Confidentiality and grounding (Phase 6 assistant)

The operational assistant (Phase 6) is built so it could later run against **real, confidential
ESP/SCADA data** without exposing it:

- **Local LLM only - no external API calls.** The assistant runs against a local
  [Ollama](https://ollama.com) server on your own machine. No telemetry, questions, or answers
  are ever sent to a third-party LLM provider or any service outside your network.
- **Every answer is grounded with citations.** The LLM only ever explains evidence retrieved
  by Python first - it never invents lap times, anomalies, or events. Every factual claim in an
  answer must cite the exact lap number and value it came from (`(Lap N: value)`), and
  `validate_citations()` checks at the code level that those citations actually exist in the
  retrieved evidence, rather than just trusting the model to comply with the prompt.
- **No grounding evidence, no LLM call.** If a question can't be parsed into a driver and lap
  number, or no matching data is found, the assistant says so directly and the LLM is never
  invoked at all.

See [Phase 6 in Status](#status-phases-1-8-done) and
[Current limitations](#current-limitations) below for the honest gaps in these guardrails.

## Screenshots

**Race Detail (Phase 1-2)** - tyre degradation and the speed/throttle/brake telemetry traces:
the closest public analogue to a real ESP/SCADA multi-sensor stream, which is the actual point
of this project (the dashboard also has lap time trend, baseline metrics, and an anomaly table,
not pictured here).

![Race Detail tab](assets/screenshots/race_detail_tab.png)

**Season Monitoring (Phase 3)** - finishing-position and speed-trap trends across a full
season, plus season KPIs and the `PositionTrendSlope` asset-health indicator.

![Season Monitoring tab](assets/screenshots/season_monitoring_tab.png)

**Fleet Monitoring (Phase 4)** - relative-pace and degradation trends for the same race across
multiple years, with a benchmarking table and year-over-year `Shift` labels.

![Fleet Monitoring tab](assets/screenshots/fleet_monitoring_tab.png)

**Predictive Analytics (Phase 5)** - lap-time forecast model comparison, predicted-vs-actual
view, feature importances, an interactive degradation forecast, and risk scores.

![Predictive Analytics tab](assets/screenshots/predictive_analytics_tab.png)

**Operational Assistant (Phase 6)** - asks a real question about the 2024 race, shows the
parsed driver/lap, the retrieved evidence, and the local LLM's grounded answer.

![Operational Assistant tab](assets/screenshots/operational_assistant_tab.png)

**Decision Support (Phase 7)** - recommended pit windows derived from the existing degradation
forecast, sorted by urgency, plus the full recommendations table.

![Decision Support tab](assets/screenshots/decision_support_tab.png)

**Arcade Replay (Phase 8)** - standalone "digital control room" window: multiple cars moving
around the real track shape together, each its own colour with a legend, plus a live
speed/throttle/brake/gear instrument cluster for the focus driver.

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
    predictive_models.py    # lap-time forecast (baselines + Linear/RF/XGBoost), risk scores
    operational_assistant.py # Phase 6: parse question -> retrieve evidence -> local LLM
    decision_support.py     # Phase 7: degradation forecast -> pit/maintenance recommendation
    replay_data.py          # pure geometry/interpolation helpers for the Arcade replay
    visualisation.py        # shared Plotly figures
  app/
    streamlit_app.py        # dashboard: Race Detail + Season + Fleet + Predictive + Assistant + Decision tabs
    arcade_replay.py         # standalone Arcade window: multi-driver cars + live HUD on the track
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
    test_predictive_models.py
    test_operational_assistant.py
    test_degradation_analysis.py
    test_decision_support.py
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

On macOS, `xgboost` (used in Phase 5) needs the OpenMP runtime, which isn't bundled:

```bash
brew install libomp
```

Phase 6 (the operational assistant) needs a local [Ollama](https://ollama.com) server - not a
hosted LLM API - so the same approach works against confidential real ESP/SCADA data later:

```bash
ollama serve                        # start the local server
ollama pull qwen2.5:7b-instruct     # default model (src/config.py:OLLAMA_MODEL)
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

5. **Inspect predictive models** (optional, sanity check - uses step 1's data, no new
   download):

   ```bash
   python -m src.predictive_models
   ```

6. **Ask the operational assistant** (optional, sanity check - needs Ollama running, see Setup):

   ```bash
   python -m src.operational_assistant
   ```

   Prints the retrieved evidence and the local LLM's answer for a grounded example question.

7. **Inspect decision support recommendations** (optional, sanity check - uses step 1's data,
   no new download):

   ```bash
   python -m src.decision_support
   ```

8. **Run the dashboard**:

   ```bash
   streamlit run app/streamlit_app.py
   ```

   The dashboard has six tabs: **Race Detail** (Phase 1-2, needs step 1's data),
   **Season Monitoring** (Phase 3, needs step 3's data), **Fleet Monitoring**
   (Phase 4, needs step 4's data), **Predictive Analytics** (Phase 5, needs step 1's data),
   **Operational Assistant** (Phase 6, needs step 1's data and Ollama running), and
   **Decision Support** (Phase 7, needs step 1's data).

9. **Run the Arcade replay** (separate native window, needs step 1's data):

   ```bash
   python app/arcade_replay.py --drivers VER,LEC,NOR
   python app/arcade_replay.py --drivers VER          # single driver, as in earlier phases
   ```

   Replays each requested driver's cached fastest lap as a colour-coded car moving around the
   track at the same time, each on its own independent lap clock (so it's a pace comparison,
   not a recreation of where cars actually were in the race), with a legend, a checkered
   start/finish marker, and an instrument cluster (speed dial, throttle dial, brake lamp, gear
   box) for the first driver listed (the "focus" driver).

10. **Run the notebooks** (`notebooks/01_data_exploration.ipynb`, `02_baseline_analysis.ipynb`,
    `03_anomaly_detection.ipynb`) for the same analysis in a more exploratory format.

11. **Run tests**:

   ```bash
   pytest tests/ -v
   ```

   Tests use synthetic fixtures and a temp directory — they never overwrite the real
   downloaded data in `data/processed/`.

## Status: Phases 1-8 done

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
- Lap-time forecasting (Phase 5 v1, single race, pooled across all drivers, no driver-identity
  feature): lag-only features (`PrevLapTimeSeconds`, `Rolling3PrevLapTimeSeconds`, computed
  without peeking at the lap being predicted) feed two honest baselines (naive lag-1, mean) and
  three trained models (Linear Regression, Random Forest, XGBoost), all evaluated on the same
  chronological train/test split (test laps happen later in the race than anything trained on).
  **Actual result on this race: the naive lag-1 baseline (MAE 0.43s) beats every trained model**
  (Random Forest MAE 0.53s, XGBoost 0.57s, Linear Regression 1.98s) - lap-to-lap correlation is
  high enough that "next lap ≈ last lap" is a genuinely hard baseline to beat at a 1-lap-ahead
  horizon. This is reported as-is rather than tuned away, per the project's "don't fabricate
  results" rule. *(Phase 5)*
- Model explainability: Linear Regression coefficients and Random Forest feature importances,
  both surfaced in the dashboard. *(Phase 5 success criteria: explain model outputs)*
- Degradation forecasting: extrapolates the existing within-stint linear degradation fit
  forward by a configurable number of laps - a deliberately simple first-order estimate, not a
  precise prediction (see limitations). *(Phase 5)*
- Degradation risk scores: projected lap-time increase over the next few laps per driver/stint,
  with Low/Medium/High categories assigned from this race's own distribution (tertiles) rather
  than a fixed constant. *(Phase 5)*
- A "Predictive Analytics" dashboard tab with the model comparison table/chart, a
  predicted-vs-actual view per model, feature-importance charts, an interactive degradation
  forecast (driver/stint/laps-ahead selectors), and the risk-score table. *(Phase 5 success
  criteria: benchmark predictive performance, explain model outputs)*
- An operational assistant that answers plain-English questions shaped like "Why did VER lose
  performance after lap 32?" by **retrieving real telemetry evidence first**, then asking a
  **local** LLM (Ollama, default `qwen2.5:7b-instruct`) to explain that evidence - never the
  other way around. If a question can't be parsed into a driver + lap number, or no matching
  lap is found, the assistant says so and **never calls the LLM**, so every answer it does give
  is grounded by construction, not by hoping a prompt instruction is obeyed. *(Phase 6)*
- Runs against a local Ollama server rather than a hosted API, specifically so the same
  architecture would work against confidential real ESP/SCADA data that can't be sent to a
  third-party provider. *(Phase 6)*
- The LLM's role is deliberately narrow: it only explains results the Python pipeline already
  computed, never performs its own analysis. The system prompt requires a citation
  (`(Lap N: value)`) for every factual claim and explicitly forbids upgrading a correlation
  (e.g. a tyre change on the same lap as a slow lap) into a diagnosis (e.g. "tyre damage")
  unless that cause is literally in the evidence. *(Phase 6)*
- Two code-level checks verify the model's compliance rather than just trusting the prompt:
  `validate_citations()` extracts every lap number the answer cites and checks it actually
  exists in the retrieved evidence (catches a fabricated lap reference); a heuristic
  `detect_speculative_language()` flags phrases like "likely due to" or "tyre damage" for human
  review. Both are surfaced directly in the dashboard. *(Phase 6)*
- Validated on a real example: asking why BOT lost performance at lap 13 of the actual 2024
  Bahrain race correctly cites the exact lap data (the pit stop onto a fresh HARD tyre,
  `TyreLife` reset to 1, flagged as a z-score anomaly) and explicitly **declines to diagnose** a
  root cause beyond what the evidence shows, rather than guessing. *(Phase 6 success criteria:
  evidence-backed explanations, no hallucinated conclusions)*
- A "Operational Assistant" dashboard tab: example/custom question input, the parsed
  driver/lap, an expandable view of the exact evidence text given to the LLM, the answer, the
  citation-validation result, and any flagged speculative phrasing. *(Phase 6)*
- pytest coverage for ingestion round-tripping, feature engineering, seasonal analysis, fleet
  analysis, predictive models, the operational assistant (citation validation, speculative-
  language detection, and the LLM call mocked/injected so tests don't require Ollama running),
  and the replay's geometry/interpolation helpers.
- Decision support recommendations that turn the existing Phase 5 degradation forecast into an
  action: "Pit now", "Pit within N laps", or "No action needed", by projecting the existing
  linear degradation fit forward to find when it crosses a configurable threshold
  (`DECISION_PIT_THRESHOLD_PCT`, default 5%) above that stint's own starting pace. No new model
  - it reuses `degradation_analysis.degradation_per_stint` and the Phase 5 forecast logic.
  *(Phase 7)*
- Validated against real strategy: for several drivers in the 2024 Bahrain race (STR, ZHO,
  RUS), the model's projected crossing lap lands a few laps *after* their actual pit lap -
  consistent with teams pitting proactively before a hard performance cliff, not reactively
  after hitting one. *(Phase 7)*
- An honestly-surfaced tension: Phase 5's `RiskCategory` (relative to this race's other stints,
  via tertiles) and Phase 7's `RecommendedAction` (an absolute fixed threshold) answer different
  questions and can legitimately disagree - e.g. several SOFT stints rank `High` risk relative
  to their peers while still being many laps from the fixed 5% threshold, so they show "No
  action needed" anyway. Both are shown side by side in the dashboard rather than reconciled
  into one number. *(Phase 7)*
- A "Decision Support" dashboard tab: a bar chart of laps remaining until the recommended pit
  window (most urgent first), and a filterable full recommendations table. *(Phase 7 success
  criteria: turn a forecast into an actionable, explainable recommendation)*
- Fixed a latent bug found while building Phase 7: `degradation_per_stint()` crashed with a
  `KeyError` if every stint in the input had fewer than 2 laps (an edge case no earlier phase's
  data happened to trigger) - it now returns a correctly-columned empty DataFrame instead.
- 80 tests total, including a new dedicated test file for `degradation_analysis.py` (previously
  only covered indirectly through other modules) added alongside the Phase 7 work.
- Multi-driver Arcade replay: any number of drivers' cached fastest laps play together on the
  same track view, each a distinct colour with a legend, no new model - just generalizing the
  Phase-prior single-driver replay to a dict of drivers (`load_multi_driver_lap_telemetry`,
  `multi_track_bounds`, `multi_lap_duration_seconds` in `replay_data.py`). *(Phase 8)*
- Each driver's lap plays on its own independent clock starting at t=0 of *their own* fastest
  lap, not session wall-clock time - so the replay is an honest lap-for-lap pace comparison
  (who's faster), not a recreation of where cars actually were on track relative to each other
  during the race. *(Phase 8)*
- A driver who finishes their lap before the others simply holds at their final telemetry
  sample (reusing the existing single-driver clamping in `get_frame_at_time`) until the shared
  clock loops at the slowest selected driver's lap duration. *(Phase 8)*
- The gauge cluster (speed/throttle/brake/gear) stays single-driver by design - it shows the
  "focus" driver (the first one passed to `--drivers`) while every selected driver's car and
  label are still drawn on the track, the same instrument-panel-plus-overview split as a SCADA
  operator screen that drills into one asset's detail while keeping the others in view.
  *(Phase 8)*
- `src/config.py:REPLAY_DRIVERS` sets the default driver set (`VER, LEC, NOR`); `--drivers` on
  the command line accepts any comma-separated list of cached driver codes, and an unknown code
  is skipped with a warning rather than aborting the whole replay. *(Phase 8)*
- 6 new tests for the multi-driver loading, bounds-union, and longest-duration helpers in
  `replay_data.py`, on top of the existing single-driver replay tests; 84 tests total.
  *(Phase 8)*

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
- Phase 5's forecasting is single-race, pooled-across-drivers, with no driver identity feature
  - by design, to force the model to learn a general degradation pattern rather than memorize
  per-driver pace, but it means the models can't yet account for a specific driver/car's
  characteristics.
- The naive lag-1 baseline beats every trained model on this race (see Status above) - this is
  reported honestly rather than hidden, but it also means none of the trained models are
  currently "production-worthy" for 1-lap-ahead forecasting; a longer forecast horizon or
  richer features (tyre temperature, fuel load, traffic) would be the next thing to try.
- The degradation forecast is a straight-line extrapolation of the existing per-stint linear
  fit; it doesn't know about upcoming pit stops, weather changes, or non-linear wear curves
  (e.g. a tyre cliff), so it should be read as a short-horizon first-order estimate.
- Risk-score thresholds (Low/Medium/High) are tertiles of *this race's own* distribution, so
  they're relative to this specific race, not a fixed, comparable-across-races threshold.
- The assistant's question parsing is regex/keyword matching (one driver code + one "lap N"
  pattern), not an embeddings-based or LLM-based parser - by design, as the smallest version
  that supports the target question shape. Questions that don't mention a 3-letter driver code
  and the word "lap" followed by a number won't be parsed, even if a human would understand them.
- The LLM's *causal reasoning* is only as good as the model, even when the *facts* it's given
  are correct: an early test run had the assistant add a speculative causal gloss (e.g.
  guessing "possible tyre damage") beyond what the evidence established (the real explanation
  is just a normal out-lap on a fresh tyre after a pit stop). The narrower system prompt and
  `detect_speculative_language()` flag reduce and surface this, but don't fully eliminate it -
  `detect_speculative_language()` is a keyword heuristic, so it both under-flags (different
  phrasing for the same speculation) and over-flags (e.g. it flags "damage" even when the model
  uses the word only to say damage is *not* shown by the evidence). Grounding prevents
  fabricated *facts* more reliably than it prevents overconfident *interpretation* of them.
- `validate_citations()` only checks that cited lap numbers exist in the evidence window - it
  cannot verify that the model paired the right lap with the right *value* (e.g. citing a real
  lap number but misquoting its lap time), since that would require re-parsing the answer's
  prose for numbers and matching them positionally, which isn't implemented yet.
- `DECISION_PIT_THRESHOLD_PCT` (default 5%) is a fixed constant, not derived from the data -
  simple and transparent, but it means recommendation timing depends on a number chosen in
  advance rather than something statistically fitted to this race or this tyre compound.
- The pit-window projection is a straight-line extrapolation of the same per-stint linear fit
  used in Phase 5, so it inherits the same caveat: real degradation is often non-linear (e.g. a
  sudden tyre "cliff"), and the recommendation doesn't know about upcoming pit stops, weather
  changes, or track status.
- `RiskCategory` (Phase 5, relative to this race) and `RecommendedAction` (Phase 7, an absolute
  threshold) can disagree, as noted in Status above - the dashboard shows both rather than
  silently picking one, but there's no reconciliation logic to explain *why* they disagree
  beyond the text description.
- Phase 7 recommends a pit *window*, not a pit *lap* tied to strategy (tyre allocation rules,
  rivals' positions, safety car probability, pit lane time loss) - it's a telemetry-only
  trigger, not a race-strategy optimizer.
- The Phase 8 multi-driver replay shows position and a single focus driver's gauges, but no
  per-car gap/delta readout, no tyre-degradation colour-coding of the cars, and no anomaly
  alerts overlaid on the track - it's the simplest version that gets multiple assets on one
  screen at once, not a full multi-asset operator view.
- Because each driver's lap clock starts independently at t=0, the multi-driver replay cannot
  be read as "who was ahead on track at this moment in the race" - only "who is faster, lap for
  lap" - conflating the two would misrepresent actual race positions.
- The reference track centerline/edges in the multi-driver replay are taken from the focus
  driver's racing line only; other drivers' slightly different lines are not used to redraw the
  track outline (drawing once stays simple, and FastF1 has no real track-width data to draw a
  more authoritative outline from anyway).

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
- **Phase 5 — Predictive Analytics** ✅ done. Forecast lap times (naive/mean baselines plus
  Linear Regression, Random Forest, XGBoost on a chronological split) and degradation
  (linear extrapolation), generate risk scores (data-driven Low/Medium/High categories).
  *Success criteria met: benchmark predictive performance (the naive baseline currently wins -
  reported honestly) and explain model outputs (coefficients/feature importances).*
- **Phase 6 — Operational Intelligence Assistant** ✅ done. Answers operational questions (e.g.
  *"Why did Driver X lose performance after lap 32?"*) by retrieving telemetry evidence first,
  then asking a local LLM (Ollama) - in a deliberately narrow role (explain pre-computed
  results only, cite every claim, never diagnose unobserved causes) - to explain it. Chosen
  over a hosted API so the same approach works against confidential real ESP/SCADA data.
  *Success criteria met: evidence-backed explanations (validated on the real BOT lap-13
  anomaly, including an explicit refusal to diagnose beyond the evidence); no hallucinated
  conclusions, checked at the code level via `validate_citations()` and
  `detect_speculative_language()`, not just prompt instructions - though see limitations for
  the remaining gaps in those checks.*
- **Phase 7 — Decision Support Recommendations** ✅ done. Turns the Phase 5 degradation
  forecast into an explicit recommendation ("Pit now" / "Pit within N laps" / "No action
  needed") by projecting the existing linear fit forward to a configurable threshold, rather
  than adding a new model. Industrial analogy: the step that turns a wear-trend reading into an
  actual maintenance work order. *Success criteria met: forecasts become actionable,
  explainable recommendations - validated against real strategy (several drivers' actual pit
  laps preceded the model's projected threshold-crossing lap, consistent with proactive
  pitting) and an honestly-surfaced disagreement between Phase 5's relative risk category and
  Phase 7's absolute threshold, shown side by side rather than papered over.*
- **Phase 8 — Multi-Driver Replay** ✅ done. Generalizes the single-driver Arcade replay to any
  number of drivers' cached fastest laps moving on the same track view at once, each a distinct
  colour with a legend, while keeping the live gauge cluster for one focus driver. Industrial
  analogy: one control-room screen showing every asset's position at a glance, with drill-down
  detail on the one currently selected. *Success criteria met: multiple assets visible
  simultaneously on one replay, reusing the existing single-driver geometry/clamping logic
  rather than rebuilding it, and unit-tested without needing a display.*

## Architecture rule

At every phase:

1. Build the simplest version first.
2. Visualize before modelling.
3. Use baseline methods before advanced ML.
4. Validate results against domain knowledge.
5. Do not introduce complexity unless the current phase is working.
6. Keep all code production-quality and portfolio-ready.
