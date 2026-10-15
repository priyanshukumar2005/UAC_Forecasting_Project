"""
Evaluation module for time-series forecasting models.

Provides standard evaluation metrics (MAE, RMSE, MAPE), multi-step horizon error analysis,
and multi-model comparative benchmarking.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


def calculate_metrics(y_true: Any, y_pred: Any) -> Dict[str, float]:
    """
    Compute standard forecasting regression performance metrics.

    Calculates:
    - MAE: Mean Absolute Error
    - RMSE: Root Mean Squared Error
    - MAPE: Mean Absolute Percentage Error (expressed in percentage: 0 to 100)

    Parameters
    ----------
    y_true : array-like
        Ground truth actual values.
    y_pred : array-like
        Predicted values from the model.

    Returns
    -------
    Dict[str, float]
        Dictionary with keys 'MAE', 'RMSE', and 'MAPE'.
    """
    try:
        y_t = np.asarray(y_true, dtype=float).ravel()
        y_p = np.asarray(y_pred, dtype=float).ravel()

        if len(y_t) != len(y_p):
            raise ValueError(f"Length mismatch: y_true ({len(y_t)}) vs y_pred ({len(y_p)})")

        if len(y_t) == 0:
            raise ValueError("Empty array provided for evaluation.")

        mae = float(np.mean(np.abs(y_t - y_p)))
        rmse = float(np.sqrt(np.mean((y_t - y_p) ** 2)))

        # Safeguard against zero division in MAPE
        eps = 1e-8
        denom = np.where(np.abs(y_t) < eps, eps, np.abs(y_t))
        mape = float(np.mean(np.abs((y_t - y_p) / denom)) * 100.0)

        return {
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "MAPE": round(mape, 4),
        }
    except Exception as exc:
        raise RuntimeError(f"Error calculating metrics: {str(exc)}") from exc


def horizon_error(
    y_true: Any,
    y_pred: Any,
    horizons: Optional[List[int]] = None,
) -> Dict[int, float]:
    """
    Compute Mean Absolute Error (MAE) at specific forecast horizon offsets.

    For example, with horizons=[1, 7, 14], calculates the absolute error at day 1,
    day 7, and day 14 (or mean absolute error up to that horizon / at that step).

    Parameters
    ----------
    y_true : array-like
        Ground truth values.
    y_pred : array-like
        Forecasted values.
    horizons : Optional[List[int]], default=[1, 7, 14]
        List of day offsets to evaluate.

    Returns
    -------
    Dict[int, float]
        Mapping from horizon offset to MAE at that offset.
    """
    if horizons is None:
        horizons = [1, 7, 14]

    try:
        y_t = np.asarray(y_true, dtype=float).ravel()
        y_p = np.asarray(y_pred, dtype=float).ravel()

        results: Dict[int, float] = {}
        for h in horizons:
            idx = h - 1
            if 0 <= idx < len(y_t):
                # Error at specific horizon offset index
                err = float(np.abs(y_t[idx] - y_p[idx]))
                results[h] = round(err, 4)
            else:
                results[h] = float("nan")

        return results
    except Exception as exc:
        raise RuntimeError(f"Error calculating horizon errors: {str(exc)}") from exc


def compare_models(results_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Compare multiple models and return a ranked performance DataFrame.

    Parameters
    ----------
    results_dict : Dict[str, Dict[str, Any]]
        Dictionary containing model results as produced by train_all_models.
        Each entry must contain 'y_test' and 'predictions'.

    Returns
    -------
    pd.DataFrame
        DataFrame with index = model names, and columns = ['MAE', 'RMSE', 'MAPE', 'Rank'],
        sorted in ascending order of MAE.
    """
    try:
        rows = []
        for model_name, res in results_dict.items():
            if "y_test" not in res or "predictions" not in res:
                continue

            y_t = res["y_test"]
            y_p = res["predictions"]
            metrics = calculate_metrics(y_t, y_p)
            rows.append({
                "Model": model_name,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "MAPE": metrics["MAPE"],
            })

        if not rows:
            return pd.DataFrame(columns=["Model", "MAE", "RMSE", "MAPE", "Rank"]).set_index("Model")

        comp_df = pd.DataFrame(rows).set_index("Model")
        # Rank by MAE ascending
        comp_df["Rank"] = comp_df["MAE"].rank(ascending=True, method="min").astype(int)
        comp_df = comp_df.sort_values(by="Rank", ascending=True)

        return comp_df

    except Exception as exc:
        raise RuntimeError(f"Error creating model comparison: {str(exc)}") from exc
