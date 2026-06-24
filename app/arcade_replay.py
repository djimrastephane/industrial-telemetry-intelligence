"""Arcade replay v1: one driver's fastest lap, position dot + live HUD.

This is the "digital control room" view: a car moving around the track
with a live speed/throttle/brake/gear readout, built on the same cached
fastest-lap telemetry as the Race Detail dashboard tab. Tyre-degradation
colour, anomaly alerts, pit-stop prompts, an asset-health score, and full
multi-lap replay are deliberately left out of this first version - see
the README roadmap for where those land.

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
    get_frame_at_time,
    lap_duration_seconds,
    load_driver_lap_telemetry,
    scale_to_screen,
    track_bounds,
)

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700
MARGIN = 60
TRACK_SHIFT_X = 120  # pixels; pushes the track right, away from the HUD text
TRACK_HALF_WIDTH = 10  # pixels either side of the centerline (FastF1 has no real track width)


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
            self._shift(scale_to_screen(x, y, self.bounds, SCREEN_WIDTH, SCREEN_HEIGHT, MARGIN))
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
        arcade.draw_line_strip(self.left_edge, arcade.color.LIGHT_GRAY, 3)
        arcade.draw_line_strip(self.right_edge, arcade.color.LIGHT_GRAY, 3)

        for (sx, sy), (ex, ey), is_black in self.start_finish_segments:
            color = arcade.color.BLACK if is_black else arcade.color.WHITE
            arcade.draw_line(sx, sy, ex, ey, color, line_width=6)

        frame = get_frame_at_time(self.lap_df, self.elapsed)
        car_x, car_y = self._shift(
            scale_to_screen(frame["X"], frame["Y"], self.bounds, SCREEN_WIDTH, SCREEN_HEIGHT, MARGIN)
        )
        arcade.draw_circle_filled(car_x, car_y, 8, arcade.color.RED)

        hud_lines = [
            f"Driver: {self.driver}",
            f"Lap time: {self.elapsed:5.1f}s / {self.duration:5.1f}s",
            f"Speed: {frame['Speed']:.0f} km/h",
            f"Throttle: {frame['Throttle']:.0f}%",
            f"Brake: {'ON' if frame['Brake'] else 'OFF'}",
            f"Gear: {frame['Gear']}",
        ]
        for i, line in enumerate(hud_lines):
            arcade.draw_text(line, 20, SCREEN_HEIGHT - 30 - i * 22, arcade.color.WHITE, 14)


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
