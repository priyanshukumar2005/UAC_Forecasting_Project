"""
Risk analysis and operational KPI module for UAC care load forecasting.

Provides capacity breach risk modeling, surge lead time estimation,
accuracy scores, and executive risk summary generation.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from scipy.stats import norm


def forecast_accuracy(mape: float) -> float:
    """
    Calculate forecast accuracy percentage from MAPE.

    Accuracy = max(0.0, 100.0 - MAPE).

    Parameters
    ----------
    mape : float
        Mean Absolute Percentage Error (0-100 scale).

    Returns
    -------
    float
        Accuracy score in percentage (bounded between 0 and 100).
    """
    try:
        acc = 100.0 - float(mape)
        return round(float(np.clip(acc, 0.0, 100.0)), 2)
    except Exception as exc:
        raise ValueError(f"Error calculating forecast accuracy: {str(exc)}") from exc


def capacity_breach_probability(
    predictions: Any,
    threshold: float = 7000.0,
    std_estimate: Optional[float] = None,
) -> float:
    """
    Estimate the probability of breaching a given operational capacity threshold
    using normal cumulative distribution function (CDF) on forecast residuals/distribution.

    P(Breach) = 1 - Phi((threshold - peak_prediction) / std_estimate)

    Parameters
    ----------
    predictions : array-like
        Predicted values for future period.
    threshold : float, default=7000.0
        Capacity limit threshold.
    std_estimate : Optional[float], default=None
        Forecast standard deviation / residual standard error.
        If None, estimated from variation of predictions or conservative empirical bound.

    Returns
    -------
    float
        Probability value between 0.0 and 1.0 (or percentage 0 to 100 depending on downstream use).
        Returns float between 0.0 and 1.0.
    """
    try:
        preds = np.asarray(predictions, dtype=float).ravel()
        if len(preds) == 0:
            return 0.0

        # Maximum predicted load in forecast window
        max_pred = float(np.max(preds))

        if std_estimate is None or std_estimate <= 0:
            # Estimate residual standard deviation from series dispersion or default 5% of max
            std_estimate = float(np.std(preds))
            if std_estimate <= 1e-4:
                std_estimate = max(1.0, 0.05 * max_pred)

        # Standard normal z-score
        z = (threshold - max_pred) / std_estimate

        # P(Load > threshold) = 1 - norm.cdf(z)
        breach_prob = float(1.0 - norm.cdf(z))
        return round(float(np.clip(breach_prob, 0.0, 1.0)), 4)
    except Exception as exc:
        raise RuntimeError(f"Error calculating breach probability: {str(exc)}") from exc


def surge_lead_time(
    df: pd.DataFrame,
    predictions: Any,
    threshold: float = 7000.0,
) -> int:
    """
    Determine the days of advance warning before the forecast exceeds the capacity threshold.

    If the forecast exceeds the threshold, returns the number of days until the first breach.
    If the current actual load is already above threshold, returns 0.
    If no breach is predicted during the forecast horizon, returns the length of the horizon.

    Parameters
    ----------
    df : pd.DataFrame
        Historical preprocessed DataFrame with 'in_hhs_care'.
    predictions : array-like
        Predicted values over the test / future horizon.
    threshold : float, default=7000.0
        Capacity threshold limit.

    Returns
    -------
    int
        Days of lead time before threshold breach.
    """
    try:
        preds = np.asarray(predictions, dtype=float).ravel()

        # Check current actual value if df is provided
        if "in_hhs_care" in df.columns and len(df) > 0:
            current_val = float(df["in_hhs_care"].iloc[-1])
            if current_val >= threshold:
                return 0

        # Find first index where prediction exceeds threshold
        breach_indices = np.where(preds >= threshold)[0]
        if len(breach_indices) > 0:
            # First breach day offset (1-indexed days ahead)
            return int(breach_indices[0] + 1)
        else:
            # No breach during predicted period
            return int(len(preds))
    except Exception as exc:
        raise RuntimeError(f"Error calculating surge lead time: {str(exc)}") from exc


def calculate_all_kpis(
    df: pd.DataFrame,
    results_dict: Dict[str, Dict[str, Any]],
    threshold: float = 7000.0,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute operational KPIs for all trained models.

    KPIs include:
    - MAE: Mean absolute error
    - RMSE: Root mean squared error
    - MAPE: Mean absolute percentage error
    - Accuracy: 100 - MAPE
    - Max Predicted Load: Highest predicted bed requirement
    - Breach Probability: Probability of exceeding threshold
    - Surge Lead Time: Days until predicted breach
    - Breach Detected: Boolean indicating whether breach occurs

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame.
    results_dict : Dict[str, Dict[str, Any]]
        Results dictionary from train_all_models.
    threshold : float, default=7000.0
        Capacity breach threshold.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Nested dictionary with model names as keys and computed KPIs as values.
    """
    kpis: Dict[str, Dict[str, Any]] = {}
    try:
        for model_name, res in results_dict.items():
            if "predictions" not in res or "y_test" not in res:
                continue

            preds = np.asarray(res["predictions"], dtype=float)
            y_test = np.asarray(res["y_test"], dtype=float)

            # Error metrics
            mae = float(np.mean(np.abs(y_test - preds)))
            rmse = float(np.sqrt(np.mean((y_test - preds) ** 2)))
            eps = 1e-8
            denom = np.where(np.abs(y_test) < eps, eps, np.abs(y_test))
            mape = float(np.mean(np.abs((y_test - preds) / denom)) * 100.0)

            accuracy = forecast_accuracy(mape)

            # Residual standard deviation
            residuals = y_test - preds
            res_std = float(np.std(residuals))

            breach_prob = capacity_breach_probability(preds, threshold=threshold, std_estimate=res_std)
            lead_time = surge_lead_time(df, preds, threshold=threshold)
            max_pred = float(np.max(preds)) if len(preds) > 0 else 0.0

            kpis[model_name] = {
                "MAE": round(mae, 2),
                "RMSE": round(rmse, 2),
                "MAPE": round(mape, 2),
                "Accuracy (%)": accuracy,
                "Max Predicted Load": round(max_pred, 0),
                "Breach Probability": breach_prob,
                "Breach Probability (%)": round(breach_prob * 100, 2),
                "Surge Lead Time (Days)": lead_time,
                "Capacity Breach Predicted": bool(max_pred >= threshold),
            }

        return kpis
    except Exception as exc:
        raise RuntimeError(f"Error calculating all KPIs: {str(exc)}") from exc


def generate_risk_summary(kpis_dict: Dict[str, Dict[str, Any]]) -> str:
    """
    Format operational KPIs into an executive risk summary string.

    Parameters
    ----------
    kpis_dict : Dict[str, Dict[str, Any]]
        Dictionary output from calculate_all_kpis.

    Returns
    -------
    str
        Formatted executive markdown summary.
    """
    try:
        if not kpis_dict:
            return "No KPI data available to generate risk summary."

        lines = [
            "=" * 60,
            "UAC CARE LOAD FORECASTING - EXECUTIVE RISK SUMMARY",
            "=" * 60,
            "",
        ]

        # Find best model by lowest MAE
        best_model = min(kpis_dict.keys(), key=lambda m: kpis_dict[m].get("MAE", float("inf")))
        best_kpis = kpis_dict[best_model]

        lines.append(f"Top Performing Model: {best_model}")
        lines.append(f"- Forecast Accuracy: {best_kpis.get('Accuracy (%)', 'N/A')}% (MAPE: {best_kpis.get('MAPE', 'N/A')}%)")
        lines.append(f"- Mean Absolute Error (MAE): {best_kpis.get('MAE', 'N/A')} children")
        lines.append(f"- Root Mean Squared Error (RMSE): {best_kpis.get('RMSE', 'N/A')} children")
        lines.append(f"- Max Projected Care Load: {best_kpis.get('Max Predicted Load', 'N/A'):,.0f} children")
        lines.append(f"- Capacity Breach Probability: {best_kpis.get('Breach Probability (%)', 'N/A')}%")
        lines.append(f"- Surge Lead Time: {best_kpis.get('Surge Lead Time (Days)', 'N/A')} days")
        lines.append("")
        lines.append("Model Performance & Risk Matrix:")
        lines.append("-" * 60)
        lines.append(f"{'Model':<25} | {'MAE':<8} | {'Accuracy':<9} | {'Breach Prob':<11} | {'Lead Time'}")
        lines.append("-" * 60)

        for model_name, m_kpis in kpis_dict.items():
            mae_str = f"{m_kpis.get('MAE', 0):.1f}"
            acc_str = f"{m_kpis.get('Accuracy (%)', 0):.1f}%"
            bp_str = f"{m_kpis.get('Breach Probability (%)', 0):.1f}%"
            lt_str = f"{m_kpis.get('Surge Lead Time (Days)', 0)}d"
            lines.append(f"{model_name:<25} | {mae_str:<8} | {acc_str:<9} | {bp_str:<11} | {lt_str}")

        lines.append("-" * 60)
        lines.append("")
        lines.append("Operational Recommendation:")
        if best_kpis.get("Capacity Breach Predicted", False):
            lines.append(
                f"[WARNING] Capacity breach projected within {best_kpis.get('Surge Lead Time (Days)')} days. "
                "Initiate emergency bed surge mobilization protocols immediately."
            )
        else:
            lines.append(
                "[NORMAL] Current forecasts indicate care load remains within safe operating thresholds. "
                "Maintain standard intake monitoring and bed placement procedures."
            )

        return "\n".join(lines)
    except Exception as exc:
        return f"Error generating risk summary: {str(exc)}"
