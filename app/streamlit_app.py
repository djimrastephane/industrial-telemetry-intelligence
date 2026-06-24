"""Industrial Asset Monitoring and Decision Support System dashboard.

FastF1 telemetry is used as a public surrogate for ESP / SCADA / production
sensor data. Run with: streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.anomaly_detection import get_anomaly_table
from src.baseline_models import (
    average_lap_time_by_driver,
    consistency_score,
    degradation_summary,
    fastest_lap_per_driver,
)
from src.data_cleaning import load_and_clean_all, load_and_clean_fleet, load_and_clean_season
from src.fleet_analysis import (
    degradation_by_year,
    field_average_gap_by_year,
    fleet_benchmark_table,
    teammate_gap_by_year,
)
from src.seasonal_analysis import (
    driver_position_trend,
    season_driver_kpis,
    season_team_kpis,
    speed_trap_trend,
)
from src.visualisation import (
    plot_degradation_by_year,
    plot_driver_comparison,
    plot_lap_times,
    plot_pace_gap_trend,
    plot_position_trend,
    plot_speed_trace,
    plot_speed_trap_trend,
    plot_throttle_brake_trace,
    plot_tyre_life_vs_laptime,
)

st.set_page_config(page_title="Industrial Telemetry Intelligence", layout="wide")

st.title("Industrial Asset Monitoring and Decision Support System")
st.caption("Demonstrated on Formula 1 telemetry as a public surrogate for industrial sensor data")

st.markdown(
    """
    **This is not a sports analytics project.** It is an industrial telemetry analytics
    project, demonstrated on Formula 1 data because plant-level ESP, SCADA, and production
    well sensor data is rarely public. F1 telemetry is high-frequency, multi-sensor, and
    failure-relevant in the same way: it lets us prototype anomaly detection and degradation
    analysis end-to-end before connecting it to proprietary industrial data.
    """
)

race_tab, season_tab, fleet_tab = st.tabs(
    ["Race Detail (Phase 1-2)", "Season Monitoring (Phase 3)", "Fleet Monitoring (Phase 4)"]
)

with race_tab:
    st.subheader("Data Loading Status")

    required_files = [config.LAPS_FILE, config.WEATHER_FILE, config.TELEMETRY_FILE]
    missing = [f for f in required_files if not f.exists()]

    if missing:
        st.error(
            "Processed data not found. Run `python -m src.data_ingestion` first to download "
            f"and cache the {config.SEASON_YEAR} {config.EVENT_NAME} ({config.SESSION_NAME}) session."
        )
        st.write("Missing files:", [str(f) for f in missing])
        st.stop()

    laps_df, weather_df, telemetry_df = load_and_clean_all()
    st.success(
        f"Loaded {config.SEASON_YEAR} {config.EVENT_NAME} - Session {config.SESSION_NAME}: "
        f"{laps_df['Driver'].nunique()} drivers, {len(laps_df)} laps, {len(telemetry_df)} telemetry samples."
    )

    st.divider()
    st.subheader("Driver Selection")

    available_drivers = sorted(laps_df["Driver"].unique())
    default_drivers = [d for d in config.COMPARISON_DRIVERS if d in available_drivers] or available_drivers[:2]

    col1, col2 = st.columns(2)
    with col1:
        driver_a = st.selectbox("Driver A", available_drivers, index=available_drivers.index(default_drivers[0]))
    with col2:
        remaining = [d for d in available_drivers if d != driver_a]
        default_b_index = remaining.index(default_drivers[1]) if len(default_drivers) > 1 and default_drivers[1] in remaining else 0
        driver_b = st.selectbox("Driver B", remaining, index=default_b_index)

    st.divider()
    st.subheader("Lap Time Trend")
    st.plotly_chart(plot_lap_times(laps_df[laps_df["Driver"].isin([driver_a, driver_b])]), use_container_width=True)

    st.subheader("Tyre Degradation")
    st.plotly_chart(plot_tyre_life_vs_laptime(laps_df[laps_df["Driver"].isin([driver_a, driver_b])]), use_container_width=True)
    st.dataframe(degradation_summary(laps_df[laps_df["Driver"].isin([driver_a, driver_b])]), use_container_width=True)

    st.subheader("Telemetry Trace (Fastest Lap)")
    telemetry_drivers = set(telemetry_df["Driver"].unique()) if not telemetry_df.empty else set()
    if driver_a in telemetry_drivers:
        st.plotly_chart(plot_speed_trace(telemetry_df, driver_a), use_container_width=True)
        st.plotly_chart(plot_throttle_brake_trace(telemetry_df, driver_a), use_container_width=True)
    if {driver_a, driver_b}.issubset(telemetry_drivers):
        st.plotly_chart(plot_driver_comparison(telemetry_df, driver_a, driver_b), use_container_width=True)
    else:
        st.info(
            f"No cached fastest-lap telemetry for {sorted({driver_a, driver_b} - telemetry_drivers)} "
            "(likely no completed timed lap for this driver in the session). "
            "Re-run `python -m src.data_ingestion` if data/processed is stale."
        )

    st.divider()
    st.subheader("Baseline Metrics")
    b1, b2 = st.columns(2)
    with b1:
        st.write("Average lap time by driver")
        st.dataframe(average_lap_time_by_driver(laps_df), use_container_width=True)
        st.write("Fastest lap per driver")
        st.dataframe(fastest_lap_per_driver(laps_df), use_container_width=True)
    with b2:
        st.write("Consistency score (lap time std dev, lower = more consistent)")
        st.dataframe(consistency_score(laps_df), use_container_width=True)

    st.subheader("Anomaly Table (|z-score| > threshold)")
    st.dataframe(get_anomaly_table(laps_df), use_container_width=True)

    st.divider()
    st.subheader("From Lap Telemetry to Asset Monitoring")
    st.markdown(
        """
        - **Lap = duty cycle.** Each lap is one operating cycle of the asset (e.g. one pump cycle, one ESP run interval).
        - **Tyre stint = run-to-failure interval.** Lap time climbing across a stint is the same shape as vibration,
          temperature, or vibration trending upward as a pump or ESP wears toward failure.
        - **Lap time z-score = SCADA alarm rule.** Flagging a lap because it deviates from that driver's own normal
          range is the simplest possible anomaly detector, identical in form to a sensor threshold alarm.
        - **Speed/throttle/brake trace = multi-sensor stream.** These are the closest public analogue to high-frequency
          pressure, flow, and current readings from a wellhead or ESP controller.

        Phases 1-2 deliberately use only descriptive baselines (averages, std dev, linear degradation slope,
        z-score) so that later phases (seasonal/fleet monitoring, forecasting, decision support, and an LLM
        explanation layer) have a clear, honest improvement to measure against.
        """
    )

with season_tab:
    st.subheader("Season Data Loading Status")

    if not config.SEASON_LAPS_FILE.exists():
        st.error(
            "Season data not found. Run `python -m src.data_ingestion season` first to download "
            f"and cache all {config.SEASON_YEAR} race rounds (this loops over every round in "
            "src/config.py:SEASON_ROUNDS and can take a while)."
        )
        st.stop()

    season_laps_df = load_and_clean_season()
    rounds_loaded = sorted(season_laps_df["RoundNumber"].unique())
    st.success(
        f"Loaded {config.SEASON_YEAR} season: {len(rounds_loaded)} rounds, "
        f"{season_laps_df['Driver'].nunique()} drivers, {len(season_laps_df)} laps."
    )
    if len(rounds_loaded) < len(config.SEASON_ROUNDS):
        st.warning(
            f"Only {len(rounds_loaded)} of {len(config.SEASON_ROUNDS)} configured rounds are cached so far "
            "- season trends below reflect rounds loaded to date."
        )

    st.divider()
    st.subheader("Driver Selection")
    season_drivers = sorted(season_laps_df["Driver"].unique())
    default_season_drivers = [d for d in config.COMPARISON_DRIVERS if d in season_drivers] or season_drivers[:2]
    selected_drivers = st.multiselect(
        "Drivers to plot", season_drivers, default=default_season_drivers
    )

    position_df = driver_position_trend(season_laps_df)
    speed_df = speed_trap_trend(season_laps_df)

    st.subheader("Season Finishing Position Trend")
    st.plotly_chart(
        plot_position_trend(position_df[position_df["Driver"].isin(selected_drivers)], "Driver"),
        use_container_width=True,
    )

    st.subheader("Season Speed Trap Trend")
    st.plotly_chart(
        plot_speed_trap_trend(speed_df[speed_df["Driver"].isin(selected_drivers)]),
        use_container_width=True,
    )

    st.divider()
    st.subheader("Season KPIs and Asset Health Indicators")
    st.write(
        "PositionTrendSlope: positive = finishing position getting worse across the season "
        "(declining asset health); negative = improving."
    )
    k1, k2 = st.columns(2)
    with k1:
        st.write("Driver KPIs")
        st.dataframe(season_driver_kpis(season_laps_df), use_container_width=True)
    with k2:
        st.write("Team KPIs")
        st.dataframe(season_team_kpis(season_laps_df), use_container_width=True)

    st.divider()
    st.subheader("From Race Telemetry to Fleet Monitoring")
    st.markdown(
        """
        - **Race = operating cycle.** Each race is one cycle of the asset (driver/car), the same way each
          well test or compressor run is one operating cycle in production monitoring.
        - **Season = fleet observation window.** Tracking every driver/team across a season is the same
          shape as monitoring a fleet of wells, ESPs, or turbines over the same time horizon.
        - **PositionTrendSlope = long-horizon degradation slope.** Within one race, lap time degrades across a
          tyre stint (Phases 1-2); across a season, finishing position drifting upward is the same kind of
          slow performance decline, just measured cycle-to-cycle instead of lap-to-lap.
        - **Speed trap trend = aggregated sensor KPI.** Rather than every raw telemetry sample, this is a
          single high-frequency reading aggregated per cycle - the same compression engineers apply when
          tracking peak pressure or peak current per well test instead of storing the full waveform.
        """
    )

with fleet_tab:
    st.subheader("Fleet Data Loading Status")

    if not config.FLEET_LAPS_FILE.exists():
        st.error(
            "Multi-year data not found. Run `python -m src.data_ingestion fleet` first to download "
            f"and cache {config.FLEET_EVENT_NAME} for every year in "
            "src/config.py:FLEET_YEARS."
        )
        st.stop()

    fleet_laps_df = load_and_clean_fleet()
    years_loaded = sorted(fleet_laps_df["Year"].unique())
    st.success(
        f"Loaded {config.FLEET_EVENT_NAME}: {len(years_loaded)} years ({min(years_loaded)}-{max(years_loaded)}), "
        f"{fleet_laps_df['Driver'].nunique()} distinct drivers, {len(fleet_laps_df)} laps."
    )
    if len(years_loaded) < len(config.FLEET_YEARS):
        st.warning(
            f"Only {len(years_loaded)} of {len(config.FLEET_YEARS)} configured years are cached so far "
            "- fleet trends below reflect years loaded to date."
        )

    st.divider()
    st.subheader("Driver Selection")
    fleet_drivers = sorted(fleet_laps_df["Driver"].unique())
    default_fleet_drivers = [d for d in config.COMPARISON_DRIVERS if d in fleet_drivers] or fleet_drivers[:2]
    selected_fleet_drivers = st.multiselect(
        "Drivers to plot", fleet_drivers, default=default_fleet_drivers
    )

    teammate_df = teammate_gap_by_year(fleet_laps_df)
    field_df = field_average_gap_by_year(fleet_laps_df)
    degradation_df = degradation_by_year(fleet_laps_df)

    st.subheader("Multi-Year Teammate Gap Trend (primary benchmark)")
    st.write(
        "Pace vs. your own teammate that year - same chassis and engine, so this cancels out "
        "almost all of the regulation/car-development effect, leaving mostly driver/strategy "
        "difference. NaN for a driver whose team had no other driver that year."
    )
    st.plotly_chart(
        plot_pace_gap_trend(
            teammate_df[teammate_df["Driver"].isin(selected_fleet_drivers)],
            "Driver", "TeammateGapPct",
            "Multi-Year Teammate Gap Trend by Driver", "Pace vs Teammate (%)",
        ),
        use_container_width=True,
    )

    with st.expander("Secondary pace benchmarks (Field Average Gap, Fastest Lap Gap)"):
        st.write(
            "Field Average Gap compares against the whole field's average pace that year. "
            "Fastest Lap Gap compares against a single lap, which is one noisy data point "
            "(e.g. an early red flag can leave an artificially slow 'fastest' lap) - kept here "
            "for context, not as the primary benchmark."
        )
        st.plotly_chart(
            plot_pace_gap_trend(
                field_df[field_df["Driver"].isin(selected_fleet_drivers)],
                "Driver", "FieldAverageGapPct",
                "Multi-Year Field Average Gap Trend by Driver", "Pace vs Field Average (%)",
            ),
            use_container_width=True,
        )

    st.subheader("Multi-Year Tyre Degradation Trend")
    st.plotly_chart(
        plot_degradation_by_year(degradation_df[degradation_df["Driver"].isin(selected_fleet_drivers)]),
        use_container_width=True,
    )

    st.divider()
    st.subheader("Reliability and Consistency")
    st.write(
        "RaceCompletionRatePct is a reliability *proxy* (laps completed vs. the most anyone "
        "completed that year) - FastF1's official DNF/retirement-reason data isn't reliably "
        "available for every season, so this stands in for it."
    )

    st.divider()
    st.subheader("Benchmarking Table and Year-over-Year Shift")
    st.write(
        "Shift: Declined / Improved / Stable, comparing each driver's TeammateGapPct this year "
        "against the previous year they appeared in this race (falls back to 'N/A' with no "
        "teammate data)."
    )
    st.dataframe(
        fleet_benchmark_table(fleet_laps_df).query("Driver in @selected_fleet_drivers"),
        use_container_width=True,
    )

    st.divider()
    st.subheader("From Multi-Year Comparison to Fleet Benchmarking")
    st.markdown(
        """
        - **Year = long observation window.** Comparing the same race across years is the same shape as
          comparing a fleet of wells, ESPs, or turbines across multiple years of operation.
        - **Teammate gap = same-asset-class benchmarking.** Comparing a driver to their own teammate (same
          chassis, same engine) is the same idea as comparing two nominally identical pumps on the same
          site - it cancels out generation/regulation effects, leaving the difference that actually matters.
        - **RaceCompletionRatePct = reliability proxy.** Standing in for true DNF/mechanical-retirement
          classification (not reliably available from FastF1 for every season), the same way a production
          system might track "operating hours vs. expected hours" when failure-cause logging is incomplete.
        - **PitStopCount / AvgPitStopRecoveryLaps = operational efficiency.** How often, and how costly,
          an intervention was - directly analogous to maintenance-stop frequency and recovery time for an
          industrial asset.
        - **Year-over-year Shift = structural change detector.** A single bad season is normal variation;
          a sustained shift in TeammateGapPct across consecutive years is the long-horizon equivalent of
          flagging that an asset's baseline performance has genuinely changed, not just had a noisy day.
        """
    )
