"""
Time-series and machine learning models module for UAC care load forecasting.

Includes:
- Time-based train/test splitting
- Baseline 7-day Moving Average model
- Auto-ARIMA statistical model
- Auto-SARIMA seasonal statistical model
- Random Forest regressor with lag/rolling features
- Gradient Boosting regressor with lag/rolling features
- Feature importance extraction
"""

from typing import Dict, Any, List, Optional, Tuple
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

try:
    import pmdarima as pm
except ImportError:
    pm = None

# Default ML feature set specified by requirements
ML_FEATURE_COLS = [
    "lag_1",
    "lag_7",
    "lag_14",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_var_7",
    "rolling_var_14",
    "net_intake_pressure",
    "day_of_week",
    "month",
    "is_weekend",
    "is_holiday_proxy",
]


class MovingAverageBaseline:
    """
    Moving Average Baseline model that forecasts using the recent N-day rolling mean.
    """
    def __init__(self, window: int = 7):
        self.window = window
        self.history_: Optional[np.ndarray] = None

    def fit(self, y: pd.Series):
        """Fit the baseline by recording training history."""
        self.history_ = np.asarray(y, dtype=float)
        return self

    def predict(self, n_periods: int, test_actuals: Optional[pd.Series] = None) -> np.ndarray:
        """
        Predict n_periods ahead. If test_actuals is provided, performs a 1-step rolling
        7-day average using available actuals; otherwise performs dynamic autoregressive forecast.
        """
        if self.history_ is None or len(self.history_) == 0:
            raise ValueError("Model must be fit before making predictions.")

        if test_actuals is not None and len(test_actuals) == n_periods:
            # 1-step ahead rolling forecast using ground-truth sequence
            full_series = np.concatenate([self.history_, np.asarray(test_actuals, dtype=float)])
            preds = []
            start_idx = len(self.history_)
            for i in range(start_idx, start_idx + n_periods):
                recent_window = full_series[i - self.window : i]
                preds.append(np.mean(recent_window))
            return np.array(preds)
        else:
            # Autoregressive multi-step projection
            preds = []
            curr_history = list(self.history_)
            for _ in range(n_periods):
                mean_val = np.mean(curr_history[-self.window:])
                preds.append(mean_val)
                curr_history.append(mean_val)
            return np.array(preds)


def train_all_models(
    df: pd.DataFrame,
    target_col: str = "in_hhs_care",
    test_ratio: float = 0.2,
    feature_cols: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Train and evaluate all forecasting models using strict chronological time-based splitting.

    Models trained:
    1. Moving Average Baseline (7-day rolling mean)
    2. ARIMA (Auto-ARIMA via pmdarima)
    3. SARIMA (Seasonal Auto-ARIMA with m=7)
    4. Random Forest Regressor (200 estimators, random_state=42)
    5. Gradient Boosting Regressor (200 estimators, lr=0.1, random_state=42)

    Parameters
    ----------
    df : pd.DataFrame
        Enriched DataFrame containing target_col and feature columns.
    target_col : str, default='in_hhs_care'
        Name of target column.
    test_ratio : float, default=0.2
        Proportion of dataset reserved for testing (last N days).
    feature_cols : Optional[List[str]], default=None
        Columns to use as ML features. If None, defaults to ML_FEATURE_COLS.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary with model names as keys and inner dictionaries containing:
        - 'predictions': np.ndarray of forecasted values for test period
        - 'train_index': pd.Index of training dates
        - 'test_index': pd.Index of test dates
        - 'model': trained model object
        - 'y_test': np.ndarray of ground-truth test values
    """
    if feature_cols is None:
        feature_cols = [c for c in ML_FEATURE_COLS if c in df.columns]

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    total_len = len(df)
    test_size = int(total_len * test_ratio)
    if test_size <= 0:
        raise ValueError("test_ratio resulted in 0 test samples. Provide a larger dataset or ratio.")

    split_idx = total_len - test_size

    # Chronological train / test split
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    train_idx = train_df.index
    test_idx = test_df.index

    y_train = train_df[target_col].values
    y_test = test_df[target_col].values
    n_test = len(y_test)

    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values

    results: Dict[str, Dict[str, Any]] = {}

    # ----------------------------------------------------
    # 1. Moving Average Baseline
    # ----------------------------------------------------
    try:
        ma_model = MovingAverageBaseline(window=7)
        ma_model.fit(train_df[target_col])
        ma_preds = ma_model.predict(n_periods=n_test, test_actuals=test_df[target_col])
        results["Moving Average Baseline"] = {
            "predictions": ma_preds,
            "train_index": train_idx,
            "test_index": test_idx,
            "model": ma_model,
            "y_test": y_test,
        }
    except Exception as exc:
        warnings.warn(f"Failed to fit Moving Average Baseline: {str(exc)}")

    # ----------------------------------------------------
    # 2. ARIMA
    # ----------------------------------------------------
    try:
        if pm is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                arima_model = pm.auto_arima(
                    y_train,
                    seasonal=False,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action="ignore",
                    max_order=6,
                )
            arima_preds = np.asarray(arima_model.predict(n_periods=n_test))
            results["ARIMA"] = {
                "predictions": arima_preds,
                "train_index": train_idx,
                "test_index": test_idx,
                "model": arima_model,
                "y_test": y_test,
            }
        else:
            warnings.warn("pmdarima is not installed; skipping ARIMA.")
    except Exception as exc:
        warnings.warn(f"Failed to fit ARIMA: {str(exc)}")

    # ----------------------------------------------------
    # 3. SARIMA (m=7 for weekly seasonality)
    # ----------------------------------------------------
    try:
        if pm is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sarima_model = pm.auto_arima(
                    y_train,
                    seasonal=True,
                    m=7,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action="ignore",
                    max_p=3,
                    max_q=3,
                    max_P=2,
                    max_Q=2,
                )
            sarima_preds = np.asarray(sarima_model.predict(n_periods=n_test))
            results["SARIMA"] = {
                "predictions": sarima_preds,
                "train_index": train_idx,
                "test_index": test_idx,
                "model": sarima_model,
                "y_test": y_test,
            }
        else:
            warnings.warn("pmdarima is not installed; skipping SARIMA.")
    except Exception as exc:
        warnings.warn(f"Failed to fit SARIMA: {str(exc)}")

    # ----------------------------------------------------
    # 4. Random Forest
    # ----------------------------------------------------
    try:
        rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_preds = rf_model.predict(X_test)
        results["Random Forest"] = {
            "predictions": rf_preds,
            "train_index": train_idx,
            "test_index": test_idx,
            "model": rf_model,
            "y_test": y_test,
        }
    except Exception as exc:
        warnings.warn(f"Failed to fit Random Forest: {str(exc)}")

    # ----------------------------------------------------
    # 5. Gradient Boosting
    # ----------------------------------------------------
    try:
        gb_model = GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.1, random_state=42
        )
        gb_model.fit(X_train, y_train)
        gb_preds = gb_model.predict(X_test)
        results["Gradient Boosting"] = {
            "predictions": gb_preds,
            "train_index": train_idx,
            "test_index": test_idx,
            "model": gb_model,
            "y_test": y_test,
        }
    except Exception as exc:
        warnings.warn(f"Failed to fit Gradient Boosting: {str(exc)}")

    return results


def get_feature_importance(model: Any, feature_names: List[str]) -> Dict[str, float]:
    """
    Extract and return sorted feature importances from a tree-based model (e.g. RF or GB).

    Parameters
    ----------
    model : Any
        Trained model with feature_importances_ attribute.
    feature_names : List[str]
        List of feature column names matching input matrix.

    Returns
    -------
    Dict[str, float]
        Dictionary mapping feature names to importance scores, sorted descending.
    """
    try:
        if not hasattr(model, "feature_importances_"):
            raise ValueError(f"Model of type {type(model).__name__} does not have feature_importances_.")

        importances = model.feature_importances_
        if len(importances) != len(feature_names):
            raise ValueError(
                f"Mismatch: {len(importances)} importances vs {len(feature_names)} feature names."
            )

        importance_dict = {
            name: float(imp) for name, imp in zip(feature_names, importances)
        }
        # Sort descending by importance
        sorted_importance = dict(
            sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)
        )
        return sorted_importance
    except Exception as exc:
        raise RuntimeError(f"Error computing feature importance: {str(exc)}") from exc
