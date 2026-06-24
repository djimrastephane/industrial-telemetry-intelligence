"""Arcade replay: one or more drivers' fastest laps, position dot(s) + live
gauges for a single focus driver.

This is the "digital control room" view: car(s) moving around the track on
the bottom of the window, with a gauge cluster (speed dial, throttle dial,
brake lamp, gear box) across the top for the focus driver - the same
instrument-panel layout as a SCADA operator screen monitoring one asset
while still seeing every asset's position on the shared track view.
Tyre-degradation colour, anomaly alerts, pit-stop prompts, and an
asset-health score are deliberately left out of this version - see the
README roadmap for where those land.

Each driver's lap plays on its own independent clock starting at t=0 (their
own fastest lap), not session wall-clock time, so the replay compares pace
lap-for-lap rather than where each car happened to be during the race.

Run with:
    python app/arcade_replay.py --drivers VER,LEC,NOR
    python app/arcade_replay.py --drivers VER          # single driver, as before
"""

import argparse
import sys
from pathlib import Path

import arcade

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import REPLAY_DRIVERS
from src.context_engine import align_context_to_session, get_context_at_timestamp, load_context
from src.data_cleaning import load_and_clean_all
from src.replay_data import (
    checker_line_segments,
    compute_track_edges,
    gauge_needle_point,
    get_frame_at_time,
    load_multi_driver_lap_telemetry,
    multi_lap_duration_seconds,
    multi_track_bounds,
    scale_to_screen,
)

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 850
GAUGE_PANEL_HEIGHT = 260  # top strip reserved for the instrument cluster
TRACK_AREA_HEIGHT = SCREEN_HEIGHT - GAUGE_PANEL_HEIGHT  # track is drawn below the gauges

MARGIN = 60
TRACK_SHIFT_X = 120  # pixels; pushes the track right, away from the left edge
TRACK_HALF_WIDTH = 10  # pixels either side of the centerline (FastF1 has no real track width)

SPEED_MAX_KPH = 350.0
THROTTLE_MAX_PCT = 100.0

GAUGE_CENTER_Y = TRACK_AREA_HEIGHT + GAUGE_PANEL_HEIGHT * 0.42  # vertical center within the panel
SPEED_GAUGE_CENTER = (180, GAUGE_CENTER_Y)
THROTTLE_GAUGE_CENTER = (420, GAUGE_CENTER_Y)
BRAKE_LAMP_CENTER = (650, GAUGE_CENTER_Y)
GEAR_BOX_CENTER = (850, GAUGE_CENTER_Y)
GAUGE_RADIUS = 75
BRAKE_LAMP_RADIUS = 40
GEAR_BOX_HALF_SIZE = 40

# Distinct colours cycled across drivers; the focus driver (gauge panel) is
# always the first one passed on the command line. 20 entries covers the
# full grid without repeating a colour.
CAR_COLORS = [
    arcade.color.RED,
    arcade.color.YELLOW,
    arcade.color.CYAN,
    arcade.color.LIME_GREEN,
    arcade.color.ORANGE,
    arcade.color.VIOLET,
    arcade.color.WHITE,
    arcade.color.PINK,
    arcade.color.SKY_BLUE,
    arcade.color.GOLD,
    arcade.color.SPRING_GREEN,
    arcade.color.SALMON,
    arcade.color.LAVENDER,
    arcade.color.KHAKI,
    arcade.color.TURQUOISE,
    arcade.color.CORAL,
    arcade.color.SILVER,
    arcade.color.YELLOW_GREEN,
    arcade.color.LIGHT_SALMON,
    arcade.color.PALE_GOLDENROD,
]

LEGEND_TOP_Y = TRACK_AREA_HEIGHT - 20  # just below the divider, inside the track view
LEGEND_ROW_HEIGHT = 22
LEGEND_COLUMNS = 5
LEGEND_COLUMN_WIDTH = SCREEN_WIDTH // LEGEND_COLUMNS


class ReplayWindow(arcade.Window):
    def __init__(self, lap_dfs: dict, drivers: list[str], laps_df=None):
        title = "Industrial Telemetry Replay - " + ", ".join(drivers)
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, title)
        arcade.set_background_color(arcade.color.DARK_SLATE_GRAY)
        self.lap_dfs = lap_dfs
        self.drivers = drivers
        self.focus_driver = drivers[0]
        self.colors = {driver: CAR_COLORS[i % len(CAR_COLORS)] for i, driver in enumerate(drivers)}
        self.bounds = multi_track_bounds(lap_dfs)
        self.duration = multi_lap_duration_seconds(lap_dfs)
        self.elapsed = 0.0

        # Phase 9: live Operational Context readout for the focus driver.
        # The replay's own clock is lap-relative (t=0 at the fastest lap's
        # start); the context engine needs absolute session-elapsed time, so
        # `self.operational_context_time_offset` is that lap's LapStartTimeSeconds. If
        # context can't be loaded/matched, the readout is simply skipped.
        self.operational_context = None
        self.operational_context_time_offset = 0.0
        if laps_df is not None:
            try:
                self.operational_context = align_context_to_session(load_context())
                focus_lap_number = lap_dfs[self.focus_driver]["LapNumber"].iloc[0]
                offset_rows = laps_df[
                    (laps_df["Driver"] == self.focus_driver) & (laps_df["LapNumber"] == focus_lap_number)
                ]
                if not offset_rows.empty:
                    self.operational_context_time_offset = float(offset_rows["LapStartTimeSeconds"].iloc[0])
            except (FileNotFoundError, KeyError):
                self.operational_context = None

        reference_lap = lap_dfs[self.focus_driver]
        centerline = [
            self._shift(scale_to_screen(x, y, self.bounds, SCREEN_WIDTH, TRACK_AREA_HEIGHT, MARGIN))
            for x, y in zip(reference_lap["X"], reference_lap["Y"])
        ]
        self.left_edge, self.right_edge = compute_track_edges(centerline, TRACK_HALF_WIDTH)
        self.start_finish_segments = checker_line_segments(
            self.left_edge[0], self.right_edge[0], num_segments=6
        )

    @staticmethod
    def _shift(point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        return x + TRACK_SHIFT_X, y

    def on_update(self, delta_time: float):
        self.elapsed += delta_time
        if self.elapsed > self.duration:
            self.elapsed = 0.0  # loop the replay

    def on_draw(self):
        self.clear()
        frames = {driver: get_frame_at_time(lap_df, self.elapsed) for driver, lap_df in self.lap_dfs.items()}
        self._draw_track(frames)
        self._draw_legend()
        self._draw_gauge_panel(frames[self.focus_driver])

    def _draw_track(self, frames: dict):
        arcade.draw_line_strip(self.left_edge, arcade.color.LIGHT_GRAY, 3)
        arcade.draw_line_strip(self.right_edge, arcade.color.LIGHT_GRAY, 3)

        for (sx, sy), (ex, ey), is_black in self.start_finish_segments:
            color = arcade.color.BLACK if is_black else arcade.color.WHITE
            arcade.draw_line(sx, sy, ex, ey, color, line_width=6)

        for driver, frame in frames.items():
            car_x, car_y = self._shift(
                scale_to_screen(frame["X"], frame["Y"], self.bounds, SCREEN_WIDTH, TRACK_AREA_HEIGHT, MARGIN)
            )
            arcade.draw_circle_filled(car_x, car_y, 8, self.colors[driver])
            arcade.draw_text(
                driver, car_x, car_y + 12, self.colors[driver], 12, anchor_x="center"
            )

    def _draw_legend(self):
        for i, driver in enumerate(self.drivers):
            column, row = i % LEGEND_COLUMNS, i // LEGEND_COLUMNS
            x = MARGIN + column * LEGEND_COLUMN_WIDTH
            y = LEGEND_TOP_Y - row * LEGEND_ROW_HEIGHT
            arcade.draw_circle_filled(x, y, 6, self.colors[driver])
            suffix = " (gauges)" if driver == self.focus_driver else ""
            arcade.draw_text(f"{driver}{suffix}", x + 14, y - 7, arcade.color.WHITE, 12)

    def _draw_gauge_panel(self, frame: dict):
        arcade.draw_lrbt_rectangle_filled(
            0, SCREEN_WIDTH, TRACK_AREA_HEIGHT, SCREEN_HEIGHT, (30, 30, 30)
        )
        arcade.draw_line(0, TRACK_AREA_HEIGHT, SCREEN_WIDTH, TRACK_AREA_HEIGHT, arcade.color.WHITE, 2)

        title = f"Focus: {self.focus_driver}   Lap time: {self.elapsed:5.1f}s / {self.duration:5.1f}s"
        arcade.draw_text(
            title, SCREEN_WIDTH / 2, SCREEN_HEIGHT - 30, arcade.color.WHITE, 16,
            anchor_x="center",
        )

        self._draw_dial(SPEED_GAUGE_CENTER, frame["Speed"], 0, SPEED_MAX_KPH, "SPEED", " km/h")
        self._draw_dial(THROTTLE_GAUGE_CENTER, frame["Throttle"], 0, THROTTLE_MAX_PCT, "THROTTLE", "%")
        self._draw_brake_lamp(BRAKE_LAMP_CENTER, frame["Brake"])
        self._draw_gear_box(GEAR_BOX_CENTER, frame["Gear"])
        self._draw_context_readout()

    def _draw_context_readout(self):
        if self.operational_context is None:
            return
        session_time = self.elapsed + self.operational_context_time_offset
        context_now = get_context_at_timestamp(self.operational_context, session_time, self.focus_driver)

        tyre = f"{context_now.get('Compound', 'Unknown')} ({context_now.get('TyreLife', '?')} laps)"
        track = context_now.get("TrackStatus", "Unknown")
        air_temp = f"{context_now['AirTemp']:.1f}C" if "AirTemp" in context_now else "—"
        track_temp = f"{context_now['TrackTemp']:.1f}C" if "TrackTemp" in context_now else "—"

        readout = f"Tyre: {tyre}   Track: {track}   Air: {air_temp}   Track Temp: {track_temp}"
        arcade.draw_text(
            readout, SCREEN_WIDTH / 2, SCREEN_HEIGHT - 50, arcade.color.LIGHT_GRAY, 12,
            anchor_x="center",
        )

    @staticmethod
    def _draw_dial(center, value, value_min, value_max, label, unit):
        cx, cy = center
        arcade.draw_circle_outline(cx, cy, GAUGE_RADIUS, arcade.color.WHITE, border_width=3)
        needle_x, needle_y = gauge_needle_point(center, GAUGE_RADIUS - 10, value, value_min, value_max)
        arcade.draw_line(cx, cy, needle_x, needle_y, arcade.color.RED, line_width=3)
        arcade.draw_circle_filled(cx, cy, 5, arcade.color.WHITE)
        arcade.draw_text(
            f"{value:.0f}{unit}", cx, cy - GAUGE_RADIUS - 22, arcade.color.WHITE, 14, anchor_x="center"
        )
        arcade.draw_text(
            label, cx, cy + GAUGE_RADIUS + 6, arcade.color.LIGHT_GRAY, 12, anchor_x="center"
        )

    @staticmethod
    def _draw_brake_lamp(center, is_braking: bool):
        cx, cy = center
        lamp_color = arcade.color.RED if is_braking else (60, 60, 60)
        arcade.draw_circle_filled(cx, cy, BRAKE_LAMP_RADIUS, lamp_color)
        arcade.draw_circle_outline(cx, cy, BRAKE_LAMP_RADIUS, arcade.color.WHITE, border_width=2)
        arcade.draw_text(
            "BRAKE", cx, cy - BRAKE_LAMP_RADIUS - 22, arcade.color.WHITE, 12, anchor_x="center"
        )

    @staticmethod
    def _draw_gear_box(center, gear: int):
        cx, cy = center
        arcade.draw_lrbt_rectangle_outline(
            cx - GEAR_BOX_HALF_SIZE, cx + GEAR_BOX_HALF_SIZE,
            cy - GEAR_BOX_HALF_SIZE, cy + GEAR_BOX_HALF_SIZE,
            arcade.color.WHITE, border_width=3,
        )
        arcade.draw_text(
            str(gear), cx, cy, arcade.color.WHITE, 30, anchor_x="center", anchor_y="center"
        )
        arcade.draw_text(
            "GEAR", cx, cy - GEAR_BOX_HALF_SIZE - 22, arcade.color.LIGHT_GRAY, 12, anchor_x="center"
        )


def main():
    parser = argparse.ArgumentParser(description="Arcade telemetry replay (multi-driver)")
    parser.add_argument(
        "--drivers",
        default=",".join(REPLAY_DRIVERS),
        help="Comma-separated 3-letter FastF1 driver codes, e.g. VER,LEC,NOR. "
        "The first one shown gets the gauge panel. Ignored if --all is set.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Replay every driver with cached telemetry (the full grid) instead of --drivers.",
    )
    args = parser.parse_args()

    laps_df, _, telemetry_df = load_and_clean_all()

    if args.all:
        all_drivers = sorted(telemetry_df["Driver"].unique())
        focus_default = REPLAY_DRIVERS[0]
        requested_drivers = (
            [focus_default] + [d for d in all_drivers if d != focus_default]
            if focus_default in all_drivers
            else all_drivers
        )
    else:
        requested_drivers = [d.strip().upper() for d in args.drivers.split(",") if d.strip()]

    lap_dfs = load_multi_driver_lap_telemetry(telemetry_df, requested_drivers)
    missing = [d for d in requested_drivers if d not in lap_dfs]
    if missing:
        print(f"Warning: no cached telemetry for {missing}, skipping.")
    if not lap_dfs:
        raise SystemExit(
            f"No cached telemetry for any of {requested_drivers}. "
            "Run `python -m src.data_ingestion` first, or pick drivers from the cached set."
        )

    drivers = [d for d in requested_drivers if d in lap_dfs]
    ReplayWindow(lap_dfs, drivers, laps_df=laps_df)
    arcade.run()


if __name__ == "__main__":
    main()
