"""
Feature engineering module for UAC (Unaccompanied Alien Children) care load forecasting.

Generates temporal lags, rolling statistical windows, net intake pressure flow signals,
and calendar/holiday indicators from the preprocessed time-series data.
"""

from typing import Optional
import pandas as pd
import numpy as np

try:
    import holidays
except ImportError:
    holidays = None


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate engineered features for forecasting 'in_hhs_care'.

    Features added:
    - Lag features:
        - lag_1: Care count 1 day prior
        - lag_7: Care count 7 days prior
        - lag_14: Care count 14 days prior
    - Rolling metrics:
        - rolling_mean_7: 7-day rolling average
        - rolling_mean_14: 14-day rolling average
        - rolling_var_7: 7-day rolling variance
        - rolling_var_14: 14-day rolling variance
    - Flow signal:
        - net_intake_pressure = transfers - discharges
    - Calendar features:
        - day_of_week: Day of week integer (0 = Monday, 6 = Sunday)
        - month: Month integer (1 = January, 12 = December)
        - is_weekend: Boolean flag for Saturday/Sunday
        - is_holiday_proxy: Boolean flag for US federal holidays

    Drops any initial rows with NaN values resulting from rolling/lag operations.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned daily DataFrame containing at least 'in_hhs_care', 'transfers', and 'discharges'
        with a DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame with engineered feature columns and NaNs removed.

    Raises
    ------
    ValueError
        If required columns or DatetimeIndex are missing.
    """
    try:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Input DataFrame must have a DatetimeIndex.")

        required_cols = ["in_hhs_care", "transfers", "discharges"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Input DataFrame missing required columns: {missing_cols}")

        feat_df = df.copy()

        # 1. Lag features (of in_hhs_care)
        feat_df["lag_1"] = feat_df["in_hhs_care"].shift(1)
        feat_df["lag_7"] = feat_df["in_hhs_care"].shift(7)
        feat_df["lag_14"] = feat_df["in_hhs_care"].shift(14)

        # 2. Rolling metrics (of in_hhs_care, closed on current day or lag window)
        # Using rolling window over the series
        feat_df["rolling_mean_7"] = feat_df["in_hhs_care"].rolling(window=7).mean()
        feat_df["rolling_mean_14"] = feat_df["in_hhs_care"].rolling(window=14).mean()
        feat_df["rolling_var_7"] = feat_df["in_hhs_care"].rolling(window=7).var()
        feat_df["rolling_var_14"] = feat_df["in_hhs_care"].rolling(window=14).var()

        # 3. Flow signal
        feat_df["net_intake_pressure"] = feat_df["transfers"] - feat_df["discharges"]

        # 4. Calendar features
        feat_df["day_of_week"] = feat_df.index.dayofweek
        feat_df["month"] = feat_df.index.month
        feat_df["is_weekend"] = feat_df["day_of_week"].isin([5, 6]).astype(int)

        # Holiday proxy
        if holidays is not None:
            # Instantiate US federal holiday calendar covering the years in dataset
            years = list(feat_df.index.year.unique())
            us_holidays = holidays.US(years=years)
            feat_df["is_holiday_proxy"] = feat_df.index.map(lambda d: 1 if d in us_holidays else 0).astype(int)
        else:
            # Fallback if holidays package not installed
            feat_df["is_holiday_proxy"] = 0

        # Drop NaN rows resulting from 14-day lag and rolling calculations
        feat_df = feat_df.dropna()

        return feat_df

    except Exception as exc:
        raise RuntimeError(f"Error occurred during feature engineering: {str(exc)}") from exc
