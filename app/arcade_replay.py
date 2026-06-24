"""Arcade replay v1: one driver's fastest lap, position dot + live gauges.

This is the "digital control room" view: a car moving around the track on
the bottom of the window, with a gauge cluster (speed dial, throttle dial,
brake lamp, gear box) across the top - the same instrument-panel layout as
a SCADA operator screen monitoring one asset. Tyre-degradation colour,
anomaly alerts, pit-stop prompts, an asset-health score, and full multi-lap
replay are deliberately left out of this first version - see the README
roadmap for where those land.

Run with: python app/arcade_replay.py --driver VER
"""

import argparse
import sys
from pathlib import Path

import arcade

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_cleaning import load_and_clean_all
from src.replay_data import (
    checker_line_segments,
    compute_track_edges,
    gauge_needle_point,
    get_frame_at_time,
    lap_duration_seconds,
    load_driver_lap_telemetry,
    scale_to_screen,
    track_bounds,
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

GAUGE_CENTER_Y = TRACK_AREA_HEIGHT + GAUGE_PANEL_HEIGHT * 0.48  # vertical center within the panel
SPEED_GAUGE_CENTER = (180, GAUGE_CENTER_Y)
THROTTLE_GAUGE_CENTER = (420, GAUGE_CENTER_Y)
BRAKE_LAMP_CENTER = (650, GAUGE_CENTER_Y)
GEAR_BOX_CENTER = (850, GAUGE_CENTER_Y)
GAUGE_RADIUS = 75
BRAKE_LAMP_RADIUS = 40
GEAR_BOX_HALF_SIZE = 40


class ReplayWindow(arcade.Window):
    def __init__(self, lap_df, driver: str):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, f"Industrial Telemetry Replay - {driver}")
        arcade.set_background_color(arcade.color.DARK_SLATE_GRAY)
        self.lap_df = lap_df
        self.driver = driver
        self.bounds = track_bounds(lap_df)
        self.duration = lap_duration_seconds(lap_df)
        self.elapsed = 0.0
        centerline = [
            self._shift(scale_to_screen(x, y, self.bounds, SCREEN_WIDTH, TRACK_AREA_HEIGHT, MARGIN))
            for x, y in zip(lap_df["X"], lap_df["Y"])
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
        frame = get_frame_at_time(self.lap_df, self.elapsed)
        self._draw_track(frame)
        self._draw_gauge_panel(frame)

    def _draw_track(self, frame: dict):
        arcade.draw_line_strip(self.left_edge, arcade.color.LIGHT_GRAY, 3)
        arcade.draw_line_strip(self.right_edge, arcade.color.LIGHT_GRAY, 3)

        for (sx, sy), (ex, ey), is_black in self.start_finish_segments:
            color = arcade.color.BLACK if is_black else arcade.color.WHITE
            arcade.draw_line(sx, sy, ex, ey, color, line_width=6)

        car_x, car_y = self._shift(
            scale_to_screen(frame["X"], frame["Y"], self.bounds, SCREEN_WIDTH, TRACK_AREA_HEIGHT, MARGIN)
        )
        arcade.draw_circle_filled(car_x, car_y, 8, arcade.color.RED)

    def _draw_gauge_panel(self, frame: dict):
        arcade.draw_lrbt_rectangle_filled(
            0, SCREEN_WIDTH, TRACK_AREA_HEIGHT, SCREEN_HEIGHT, (30, 30, 30)
        )
        arcade.draw_line(0, TRACK_AREA_HEIGHT, SCREEN_WIDTH, TRACK_AREA_HEIGHT, arcade.color.WHITE, 2)

        title = f"Driver: {self.driver}   Lap time: {self.elapsed:5.1f}s / {self.duration:5.1f}s"
        arcade.draw_text(
            title, SCREEN_WIDTH / 2, SCREEN_HEIGHT - 30, arcade.color.WHITE, 16,
            anchor_x="center",
        )

        self._draw_dial(SPEED_GAUGE_CENTER, frame["Speed"], 0, SPEED_MAX_KPH, "SPEED", " km/h")
        self._draw_dial(THROTTLE_GAUGE_CENTER, frame["Throttle"], 0, THROTTLE_MAX_PCT, "THROTTLE", "%")
        self._draw_brake_lamp(BRAKE_LAMP_CENTER, frame["Brake"])
        self._draw_gear_box(GEAR_BOX_CENTER, frame["Gear"])

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
    parser = argparse.ArgumentParser(description="Arcade telemetry replay (v1)")
    parser.add_argument("--driver", default="VER", help="3-letter FastF1 driver code")
    args = parser.parse_args()

    _, _, telemetry_df = load_and_clean_all()
    lap_df = load_driver_lap_telemetry(telemetry_df, args.driver)
    if lap_df.empty:
        raise SystemExit(
            f"No cached telemetry for driver '{args.driver}'. "
            "Run `python -m src.data_ingestion` first, or pick a driver from the cached set."
        )

    ReplayWindow(lap_df, args.driver)
    arcade.run()


if __name__ == "__main__":
    main()
