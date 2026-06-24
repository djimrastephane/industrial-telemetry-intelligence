"""Phase 5: predictive analytics.

Forecasts the next lap's time using only information available *before*
that lap happens (no peeking at the lap's own outcome) - the same shape as
forecasting a sensor's next reading from its recent operating history.
Baselines come first (naive lag-1, mean), then Linear Regression, Random
Forest, and XGBoost, all evaluated on the same chronological train/test
split so the comparison is honest about whether the extra model complexity
actually earns its keep.

Scope (Phase 5 v1): single race (Bahrain 2024), pooled across all drivers,
without driver identity as a feature - the model has to learn the general
tyre-degradation pattern rather than just memorizing each driver's baseline
pace. The first lap of every stint has no lap history yet, so it's excluded
from training/evaluation (see add_forecast_features).
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.degradation_analysis import degradation_per_stint
from src.feature_engineering import add_forecast_features

NUMERIC_FEATURE_COLUMNS = ["StintLap", "TyreLife", "PrevLapTimeSeconds", "Rolling3PrevLapTimeSeconds"]


def prepare_model_dataset(laps_df: pd.DataFrame) -> pd.DataFrame:
    """Adds forecast features and drops rows with no lap history yet."""
    df = add_forecast_features(laps_df)
    return df.dropna(subset=["PrevLapTimeSeconds", "Rolling3PrevLapTimeSeconds", "LapTimeSeconds"])


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric forecast features plus one-hot encoded tyre compound."""
    compound_dummies = pd.get_dummies(df["Compound"], prefix="Compound")
    return pd.concat(
        [df[NUMERIC_FEATURE_COLUMNS].reset_index(drop=True), compound_dummies.reset_index(drop=True)],
        axis=1,
    )


def chronological_train_test_split(df: pd.DataFrame, test_fraction: float = 0.25) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by LapNumber, not randomly: the test set is laps later in the
    race than anything trained on, since this is a forecast, not interpolation."""
    split_lap = df["LapNumber"].quantile(1 - test_fraction)
    train = df[df["LapNumber"] <= split_lap]
    test = df[df["LapNumber"] > split_lap]
    return train, test


def naive_lag1_predict(df: pd.DataFrame) -> np.ndarray:
    """Dumbest forecast: next lap will be the same as the previous lap."""
    return df["PrevLapTimeSeconds"].to_numpy()


def mean_baseline_predict(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    """Even dumber: always predict the training set's average lap time."""
    return np.full(len(test_df), train_df["LapTimeSeconds"].mean())


def evaluate(y_true, y_pred) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "R2": r2_score(y_true, y_pred),
    }


def train_linear_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42) -> RandomForestRegressor:
    model = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def compare_models(laps_df: pd.DataFrame, test_fraction: float = 0.25) -> tuple[pd.DataFrame, dict]:
    """Train every model on the same chronological split; return a
    comparison table (sorted by MAE) plus fitted models/predictions/data for
    further inspection (feature importances, predicted-vs-actual plots)."""
    df = prepare_model_dataset(laps_df)
    train_df, test_df = chronological_train_test_split(df, test_fraction)

    X_train = build_feature_matrix(train_df)
    X_test = build_feature_matrix(test_df).reindex(columns=X_train.columns, fill_value=0)
    y_train = train_df["LapTimeSeconds"]
    y_test = test_df["LapTimeSeconds"]

    predictions = {
        "Naive (lag-1)": naive_lag1_predict(test_df),
        "Mean baseline": mean_baseline_predict(train_df, test_df),
    }
    models = {
        "Linear Regression": train_linear_model(X_train, y_train),
        "Random Forest": train_random_forest(X_train, y_train),
        "XGBoost": train_xgboost(X_train, y_train),
    }
    for name, model in models.items():
        predictions[name] = model.predict(X_test)

    results_df = pd.DataFrame(
        [{"Model": name, **evaluate(y_test, pred)} for name, pred in predictions.items()]
    ).sort_values("MAE").reset_index(drop=True)

    artifacts = {
        "train_df": train_df, "test_df": test_df,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "predictions": predictions, "models": models,
    }
    return results_df, artifacts


def explain_linear_model(model: LinearRegression, feature_names: list[str]) -> pd.DataFrame:
    return (
        pd.DataFrame({"Feature": feature_names, "Coefficient": model.coef_})
        .sort_values("Coefficient", key=abs, ascending=False)
        .reset_index(drop=True)
    )


def explain_tree_model(model, feature_names: list[str]) -> pd.DataFrame:
    return (
        pd.DataFrame({"Feature": feature_names, "Importance": model.feature_importances_})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )


def forecast_stint_degradation(laps_df: pd.DataFrame, driver: str, stint: int, laps_ahead: int = 5) -> pd.DataFrame:
    """Extrapolate the existing within-stint linear degradation fit
    (degradation_analysis.degradation_per_stint) forward by `laps_ahead`
    laps - a deliberately simple forecast: real wear is often non-linear,
    so this should be read as a first-order estimate, not a precise prediction.
    """
    from src.feature_engineering import add_stint_lap_number

    df = add_stint_lap_number(laps_df)
    stint_laps = df[(df["Driver"] == driver) & (df["Stint"] == stint)]
    if stint_laps.empty:
        return pd.DataFrame()
    last_stint_lap = int(stint_laps["StintLap"].max())

    stint_fits = degradation_per_stint(laps_df)
    fit = stint_fits[(stint_fits["Driver"] == driver) & (stint_fits["Stint"] == stint)]
    if fit.empty:
        return pd.DataFrame()
    slope = fit["DegradationSecondsPerLap"].iloc[0]
    intercept = fit["StartingLapTimeEstimate"].iloc[0]

    future_stint_laps = list(range(last_stint_lap + 1, last_stint_lap + 1 + laps_ahead))
    return pd.DataFrame(
        {
            "Driver": driver,
            "Stint": stint,
            "StintLap": future_stint_laps,
            "ForecastLapTimeSeconds": [intercept + slope * sl for sl in future_stint_laps],
        }
    )


def degradation_risk_scores(laps_df: pd.DataFrame, laps_ahead: int = 5, min_stint_laps: int = 3) -> pd.DataFrame:
    """Risk score per driver/stint: projected lap-time increase over the
    next `laps_ahead` laps if the current degradation trend continues.

    Risk category (Low/Medium/High) is assigned from this race's own
    distribution of projected increases (tertiles), not a fixed constant -
    so the threshold adapts to how much degradation is actually happening
    in this race rather than an arbitrary number picked in advance.
    """
    stints = degradation_per_stint(laps_df)
    stints = stints[stints["Laps"] >= min_stint_laps].copy()
    if stints.empty:
        return stints

    stints["ProjectedIncreaseSeconds"] = stints["DegradationSecondsPerLap"] * laps_ahead

    low_cut, high_cut = stints["ProjectedIncreaseSeconds"].quantile([1 / 3, 2 / 3])

    def categorize(value: float) -> str:
        if value <= low_cut:
            return "Low"
        if value <= high_cut:
            return "Medium"
        return "High"

    stints["RiskCategory"] = stints["ProjectedIncreaseSeconds"].apply(categorize)
    columns = ["Driver", "Stint", "Compound", "Laps", "DegradationSecondsPerLap", "ProjectedIncreaseSeconds", "RiskCategory"]
    return stints[columns].sort_values("ProjectedIncreaseSeconds", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    from src.data_cleaning import load_and_clean_all

    laps, _, _ = load_and_clean_all()
    results, artifacts = compare_models(laps)
    print("--- model comparison ---")
    print(results)

    print("\n--- linear model coefficients ---")
    print(explain_linear_model(artifacts["models"]["Linear Regression"], list(artifacts["X_train"].columns)))

    print("\n--- random forest feature importances ---")
    print(explain_tree_model(artifacts["models"]["Random Forest"], list(artifacts["X_train"].columns)))

    print("\n--- degradation risk scores (top 10) ---")
    print(degradation_risk_scores(laps).head(10))
