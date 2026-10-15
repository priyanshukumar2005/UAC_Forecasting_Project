"""
Data preprocessing module for UAC (Unaccompanied Alien Children) forecasting.

Handles data loading, date formatting, chronological sorting, numeric cleaning,
column renaming, frequency regularization, and missing-day linear interpolation.
"""

from typing import Optional
import pandas as pd
import numpy as np


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load raw UAC CSV dataset from disk.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Raw pandas DataFrame loaded directly from the CSV.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at filepath.
    ValueError
        If the loaded file is empty or corrupted.
    """
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            raise ValueError(f"Dataset at '{filepath}' is empty.")
        return df
    except FileNotFoundError as fnf_err:
        raise FileNotFoundError(f"File not found at path: {filepath}") from fnf_err
    except Exception as exc:
        raise RuntimeError(f"Failed to load dataset from '{filepath}': {str(exc)}") from exc


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and regularize the raw UAC dataset.

    Steps:
    1. Parse 'Date' column to datetime with dayfirst=True.
    2. Sort chronologically ascending.
    3. Set 'Date' as DatetimeIndex.
    4. Remove commas from numeric strings and convert columns to integers/floats.
    5. Rename columns to standard short names:
       'apprehended', 'in_cbp_custody', 'transfers', 'in_hhs_care', 'discharges'.
    6. Regularize daily frequency (asfreq('D')) and fill missing days using linear interpolation.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input DataFrame.

    Returns
    -------
    pd.DataFrame
        Cleaned daily time-series DataFrame with regular DatetimeIndex and interpolated values.
    """
    try:
        df_clean = df.copy()

        # Identify date column
        date_col = None
        for col in df_clean.columns:
            if col.strip().lower() == "date":
                date_col = col
                break

        if date_col is None:
            raise ValueError("Required 'Date' column not found in input DataFrame.")

        # Convert Date to datetime with dayfirst=True
        df_clean[date_col] = pd.to_datetime(df_clean[date_col], dayfirst=True)

        # Sort chronologically ascending
        df_clean = df_clean.sort_values(by=date_col, ascending=True)

        # Set Date as DatetimeIndex
        df_clean = df_clean.set_index(date_col)
        df_clean.index.name = "date"

        # Rename columns to standardized short names
        # Expected mapping based on raw dataset header
        col_rename_map = {}
        for col in df_clean.columns:
            norm_col = col.strip().lower()
            if "apprehended" in norm_col:
                col_rename_map[col] = "apprehended"
            elif "in cbp custody" in norm_col:
                col_rename_map[col] = "in_cbp_custody"
            elif "transferred out of cbp" in norm_col:
                col_rename_map[col] = "transfers"
            elif "in hhs care" in norm_col:
                col_rename_map[col] = "in_hhs_care"
            elif "discharged from hhs" in norm_col:
                col_rename_map[col] = "discharges"

        df_clean = df_clean.rename(columns=col_rename_map)

        expected_cols = ["apprehended", "in_cbp_custody", "transfers", "in_hhs_care", "discharges"]
        missing_expected = [c for c in expected_cols if c not in df_clean.columns]
        if missing_expected:
            raise ValueError(f"Failed to identify columns: {missing_expected}. Existing columns: {list(df_clean.columns)}")

        # Keep relevant columns
        df_clean = df_clean[expected_cols]

        # Clean numeric columns: remove commas, whitespace, convert to numeric
        for col in expected_cols:
            if df_clean[col].dtype == object:
                # Remove quotes, commas, spaces
                df_clean[col] = (
                    df_clean[col]
                    .astype(str)
                    .str.replace(",", "", regex=False)
                    .str.strip()
                )
                df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
            else:
                df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

        # In case of duplicate dates, aggregate by mean
        if df_clean.index.has_duplicates:
            df_clean = df_clean.groupby(df_clean.index).mean()

        # Fill missing calendar days with asfreq('D') and linear interpolation
        df_clean = df_clean.asfreq("D")
        df_clean = df_clean.interpolate(method="linear")

        # In case of leading NaNs at start after interpolation, backward fill
        df_clean = df_clean.bfill()

        # Convert count values to int (or rounded int)
        for col in expected_cols:
            df_clean[col] = df_clean[col].round().astype(int)

        return df_clean

    except Exception as exc:
        raise RuntimeError(f"Error occurred during preprocessing: {str(exc)}") from exc


def load_and_preprocess(filepath: str) -> pd.DataFrame:
    """
    Convenience function that loads raw CSV and applies full preprocessing pipeline.

    Parameters
    ----------
    filepath : str
        Path to the raw CSV dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned, daily-indexed, interpolated DataFrame.
    """
    raw_df = load_data(filepath)
    return preprocess(raw_df)
