import numpy as np
import pandas as pd
import pytest

from src.predictive_models import (
    build_feature_matrix,
    chronological_train_test_split,
    compare_models,
    degradation_risk_scores,
    evaluate,
    forecast_stint_degradation,
    mean_baseline_predict,
    naive_lag1_predict,
    prepare_model_dataset,
)


@pytest.fixture
def sample_laps_df():
    rng = np.random.default_rng(42)
    rows = []
    for driver, base_pace in [("VER", 90.0), ("LEC", 91.0)]:
        for lap in range(1, 21):
            stint = 1 if lap <= 10 else 2
            stint_lap = lap if lap <= 10 else lap - 10
            # Lap time drifts upward within a stint (degradation) plus small noise.
            lap_time = base_pace + 0.1 * stint_lap + rng.normal(0, 0.05)
            rows.append(
                {
                    "Driver": driver, "Team": "Team", "LapNumber": lap, "Stint": stint,
                    "Compound": "MEDIUM" if stint == 1 else "HARD", "TyreLife": stint_lap,
                    "LapTimeSeconds": lap_time,
                }
            )
    return pd.DataFrame(rows)


def test_prepare_model_dataset_drops_first_lap_of_each_stint(sample_laps_df):
    prepared = prepare_model_dataset(sample_laps_df)
    # 2 drivers x 2 stints x 1 first-lap-per-stint dropped = 4 fewer rows than input.
    assert len(prepared) == len(sample_laps_df) - 4
    assert prepared["PrevLapTimeSeconds"].isna().sum() == 0


def test_build_feature_matrix_one_hot_encodes_compound(sample_laps_df):
    prepared = prepare_model_dataset(sample_laps_df)
    matrix = build_feature_matrix(prepared)
    assert "Compound_MEDIUM" in matrix.columns
    assert "Compound_HARD" in matrix.columns
    assert len(matrix) == len(prepared)


def test_chronological_split_keeps_test_laps_later(sample_laps_df):
    prepared = prepare_model_dataset(sample_laps_df)
    train, test = chronological_train_test_split(prepared, test_fraction=0.25)
    assert train["LapNumber"].max() <= test["LapNumber"].min()
    assert len(train) + len(test) == len(prepared)


def test_naive_lag1_predict_equals_previous_lap(sample_laps_df):
    prepared = prepare_model_dataset(sample_laps_df)
    pred = naive_lag1_predict(prepared)
    assert np.allclose(pred, prepared["PrevLapTimeSeconds"].to_numpy())


def test_mean_baseline_predict_is_constant(sample_laps_df):
    prepared = prepare_model_dataset(sample_laps_df)
    train, test = chronological_train_test_split(prepared)
    pred = mean_baseline_predict(train, test)
    assert len(set(pred)) == 1
    assert pred[0] == pytest.approx(train["LapTimeSeconds"].mean())


def test_evaluate_returns_expected_keys():
    metrics = evaluate([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert set(metrics.keys()) == {"MAE", "RMSE", "R2"}
    assert metrics["MAE"] == pytest.approx(0.0)
    assert metrics["R2"] == pytest.approx(1.0)


def test_compare_models_returns_all_models_sorted_by_mae(sample_laps_df):
    results, artifacts = compare_models(sample_laps_df)
    expected_models = {"Naive (lag-1)", "Mean baseline", "Linear Regression", "Random Forest", "XGBoost"}
    assert set(results["Model"]) == expected_models
    assert results["MAE"].is_monotonic_increasing
    assert set(artifacts["predictions"].keys()) == expected_models


def test_forecast_stint_degradation_extends_beyond_observed_laps(sample_laps_df):
    forecast = forecast_stint_degradation(sample_laps_df, "VER", 1, laps_ahead=3)
    assert len(forecast) == 3
    assert forecast["StintLap"].tolist() == [11, 12, 13]
    # Lap time should keep climbing given the upward-drifting fixture data.
    assert forecast["ForecastLapTimeSeconds"].is_monotonic_increasing


def test_forecast_stint_degradation_unknown_driver_returns_empty(sample_laps_df):
    forecast = forecast_stint_degradation(sample_laps_df, "NOBODY", 1)
    assert forecast.empty


def test_degradation_risk_scores_assigns_categories_from_distribution(sample_laps_df):
    risk = degradation_risk_scores(sample_laps_df, laps_ahead=5, min_stint_laps=3)
    assert set(risk["RiskCategory"]).issubset({"Low", "Medium", "High"})
    # Highest projected increase should be sorted first.
    assert risk["ProjectedIncreaseSeconds"].is_monotonic_decreasing
