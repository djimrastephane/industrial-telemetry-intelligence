import pandas as pd
import pytest

from src.replay_data import (
    checker_line_segments,
    compute_track_edges,
    get_frame_at_time,
    lap_duration_seconds,
    load_driver_lap_telemetry,
    scale_to_screen,
    track_bounds,
)


@pytest.fixture
def sample_telemetry_df():
    return pd.DataFrame(
        {
            "Driver": ["VER", "VER", "VER", "LEC"],
            "Time": pd.to_timedelta([0.0, 1.0, 2.0, 0.0], unit="s"),
            "X": [0.0, 100.0, 200.0, 50.0],
            "Y": [0.0, 50.0, 100.0, 25.0],
            "Speed": [200.0, 250.0, 300.0, 210.0],
            "Throttle": [50.0, 100.0, 100.0, 60.0],
            "Brake": [False, False, True, False],
            "nGear": [4, 6, 7, 4],
        }
    )


def test_load_driver_lap_telemetry_filters_and_sorts(sample_telemetry_df):
    lap = load_driver_lap_telemetry(sample_telemetry_df, "VER")
    assert len(lap) == 3
    assert lap["ElapsedSeconds"].tolist() == [0.0, 1.0, 2.0]


def test_track_bounds(sample_telemetry_df):
    lap = load_driver_lap_telemetry(sample_telemetry_df, "VER")
    bounds = track_bounds(lap)
    assert bounds == (0.0, 200.0, 0.0, 100.0)


def test_scale_to_screen_maps_into_margin_box():
    bounds = (0.0, 200.0, 0.0, 100.0)
    sx, sy = scale_to_screen(0.0, 0.0, bounds, screen_width=800, screen_height=600, margin=60)
    assert sx == pytest.approx(60.0)
    assert sy == pytest.approx(60.0)

    sx_max, sy_max = scale_to_screen(200.0, 100.0, bounds, screen_width=800, screen_height=600, margin=60)
    assert sx_max <= 800 - 60 + 1e-6
    assert sy_max <= 600 - 60 + 1e-6


def test_get_frame_at_time_returns_nearest_sample(sample_telemetry_df):
    lap = load_driver_lap_telemetry(sample_telemetry_df, "VER")
    frame = get_frame_at_time(lap, 1.4)
    assert frame["ElapsedSeconds"] == 1.0
    assert frame["Speed"] == 250.0
    assert frame["Gear"] == 6


def test_get_frame_at_time_clamps_to_lap_bounds(sample_telemetry_df):
    lap = load_driver_lap_telemetry(sample_telemetry_df, "VER")
    before_start = get_frame_at_time(lap, -5.0)
    after_end = get_frame_at_time(lap, 999.0)
    assert before_start["ElapsedSeconds"] == 0.0
    assert after_end["ElapsedSeconds"] == 2.0
    assert after_end["Brake"] is True


def test_lap_duration_seconds(sample_telemetry_df):
    lap = load_driver_lap_telemetry(sample_telemetry_df, "VER")
    assert lap_duration_seconds(lap) == pytest.approx(2.0)


def test_compute_track_edges_offsets_perpendicular_to_straight_line():
    # Horizontal line moving in +x: perpendicular offset should be purely vertical.
    points = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]
    left, right = compute_track_edges(points, half_width=5.0)

    assert len(left) == len(right) == 3
    for (lx, ly), (rx, ry), (px, py) in zip(left, right, points):
        assert lx == pytest.approx(px)
        assert rx == pytest.approx(px)
        assert ly == pytest.approx(py + 5.0)
        assert ry == pytest.approx(py - 5.0)


def test_compute_track_edges_empty_input():
    left, right = compute_track_edges([], half_width=5.0)
    assert left == []
    assert right == []


def test_checker_line_segments_covers_full_span_and_alternates():
    segments = checker_line_segments((0.0, 0.0), (12.0, 0.0), num_segments=6)

    assert len(segments) == 6
    assert segments[0][0] == pytest.approx((0.0, 0.0))
    assert segments[-1][1] == pytest.approx((12.0, 0.0))
    # Consecutive segments connect end-to-end.
    for (_, end_prev, _), (start_next, _, _) in zip(segments, segments[1:]):
        assert end_prev == pytest.approx(start_next)
    # Alternates black/white starting with black.
    assert [is_black for _, _, is_black in segments] == [True, False, True, False, True, False]
