#!/usr/bin/env python3
"""
Expert-Level Data Validation & Reporting Script

This script demonstrates how to load, validate, and report on a time-series dataset
prior to plotting and backtesting. It checks for:

1. Missing values
2. Duplicates
3. Outliers (via IQR and Z-scores)
4. Timestamp continuity

It then generates a textual summary and sample visual reports to assist in
understanding data quality. Modify as needed for your particular workflow.
"""

import os
import logging
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# -------------------------------------------------------------------
# Logging configuration
# -------------------------------------------------------------------
LOG_FORMAT = "%(asctime)s — %(name)s — %(levelname)s — %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Data Loading & Parsing
# -------------------------------------------------------------------
def load_data(
    file_path: str,
    date_col: str = "timestamp",
    price_col: str = "last_trade_price",
    parse_dates: bool = True
) -> pd.DataFrame:
    """
    Load the CSV file into a Pandas DataFrame with optional date parsing.
    
    :param file_path: Path to the CSV file.
    :param date_col: Name of the column representing the timestamp.
    :param price_col: Name of the column representing the main numeric metric (price).
    :param parse_dates: Whether to parse the date column as a datetime object.
    :return: DataFrame with parsed datetime index (if parse_dates=True).
    """
    logger.info(f"Loading data from {file_path}")

    if parse_dates:
        # Let pandas parse the 'timestamp' column automatically, no date_parser argument
        df = pd.read_csv(file_path, parse_dates=[date_col])
    else:
        df = pd.read_csv(file_path)

    # Rename columns so the script can standardize them
    df.rename(columns={date_col: "Timestamp", price_col: "Price"}, inplace=True)

    # Sort by timestamp to ensure chronological order
    df.sort_values("Timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Optionally set 'Timestamp' as index
    df.set_index("Timestamp", inplace=True, drop=True)

    logger.info(f"Data loaded: {len(df)} rows")
    return df


# -------------------------------------------------------------------
# Data Checks
# -------------------------------------------------------------------
def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check and report missing values in each column.
    
    :param df: Input DataFrame.
    :return: DataFrame with columns ['Column', 'MissingValues', 'PctMissing'].
    """
    missing_report = []
    for col in df.columns:
        n_missing = df[col].isnull().sum()
        pct_missing = (n_missing / len(df)) * 100
        missing_report.append([col, n_missing, pct_missing])
    
    report_df = pd.DataFrame(missing_report, columns=["Column", "MissingValues", "PctMissing"])
    return report_df


def check_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check and report duplicate rows.
    
    :param df: Input DataFrame.
    :return: DataFrame containing the duplicate rows.
    """
    duplicates = df[df.duplicated(keep=False)]
    return duplicates


def detect_outliers_iqr(df: pd.DataFrame, column: str, iqr_factor: float = 1.5) -> pd.DataFrame:
    """
    Detect outliers using the IQR (Interquartile Range) method.
    
    :param df: Input DataFrame.
    :param column: Column name to check for outliers.
    :param iqr_factor: The multiplier for the IQR range (default=1.5).
    :return: DataFrame containing rows flagged as outliers.
    """
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - iqr_factor * iqr
    upper_bound = q3 + iqr_factor * iqr

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers


def detect_outliers_zscore(df: pd.DataFrame, column: str, threshold: float = 3.0) -> pd.DataFrame:
    """
    Detect outliers using z-score.
    
    :param df: Input DataFrame.
    :param column: Column name to check for outliers.
    :param threshold: The z-score cutoff (3.0 by default).
    :return: DataFrame containing rows flagged as outliers.
    """
    # stats.zscore() excludes NaN by default, so dropna() ensures valid array
    z_scores = stats.zscore(df[column].dropna())
    abs_z_scores = np.abs(z_scores)
    outliers = df.loc[abs_z_scores > threshold]
    return outliers


def check_timestamp_continuity(df: pd.DataFrame, max_gap: timedelta = timedelta(days=1)) -> pd.DataFrame:
    """
    Check for missing time intervals or abnormal gaps in the timestamp index.
    
    :param df: Input DataFrame (with 'Timestamp' as index).
    :param max_gap: Maximum allowed gap between consecutive rows.
    :return: DataFrame with potential gaps.
    """
    # Compute the difference between consecutive timestamps
    time_diffs = df.index.to_series().diff()
    
    # Flag rows where the gap is larger than max_gap
    gap_indices = time_diffs[time_diffs > max_gap].index
    
    # Create a summary DataFrame indicating the gap size
    gap_report = pd.DataFrame({
        "GapStart": gap_indices,
        "GapSize": time_diffs[gap_indices].values
    })
    gap_report.reset_index(drop=True, inplace=True)
    return gap_report


# -------------------------------------------------------------------
# Visualization & Reporting
# -------------------------------------------------------------------
def plot_price(df: pd.DataFrame, price_col: str = "Price", title: str = "Price Over Time") -> None:
    """
    Basic time-series plot of the price column.
    
    :param df: Input DataFrame.
    :param price_col: Column name for price.
    :param title: Plot title.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df[price_col], label=price_col)
    plt.title(title)
    plt.xlabel("Timestamp")
    plt.ylabel("Price")
    plt.legend()
    plt.show()


def plot_outliers(df: pd.DataFrame, original_df: pd.DataFrame, price_col: str = "Price", method_name: str = "IQR") -> None:
    """
    Plot time-series data highlighting outlier points in red.
    
    :param df: DataFrame containing outlier rows.
    :param original_df: The full DataFrame for reference.
    :param price_col: Column name for price.
    :param method_name: The name of the outlier detection method (for labeling).
    """
    plt.figure(figsize=(12, 6))
    plt.plot(original_df.index, original_df[price_col], label="Original Data", color="blue")
    plt.scatter(df.index, df[price_col], label=f"Outliers ({method_name})", color="red")
    plt.title(f"Outliers Detected Using {method_name}")
    plt.xlabel("Timestamp")
    plt.ylabel("Price")
    plt.legend()
    plt.show()


def generate_textual_report(df: pd.DataFrame,
                            missing_df: pd.DataFrame,
                            duplicates_df: pd.DataFrame,
                            outliers_iqr_df: pd.DataFrame,
                            outliers_zscore_df: pd.DataFrame,
                            gap_report: pd.DataFrame) -> str:
    """
    Create a textual summary of key validation metrics.
    
    :param df: Original DataFrame.
    :param missing_df: Missing values report.
    :param duplicates_df: Duplicate rows report.
    :param outliers_iqr_df: IQR-based outliers.
    :param outliers_zscore_df: Z-score based outliers.
    :param gap_report: Timestamp continuity gaps.
    :return: String containing the summarized report.
    """
    lines = []
    lines.append("DATA VALIDATION REPORT\n")
    lines.append("====================================\n")
    
    # Basic info
    lines.append(f"Total rows in dataset: {len(df)}\n")
    lines.append(f"Data start: {df.index.min()}, Data end: {df.index.max()}\n")
    lines.append("-" * 40 + "\n")
    
    # Missing values
    lines.append("MISSING VALUES:\n")
    lines.append(missing_df.to_string(index=False))
    lines.append("\n" + "-" * 40 + "\n")
    
    # Duplicates
    lines.append("DUPLICATES:\n")
    duplicates_count = len(duplicates_df)
    lines.append(f"Number of duplicate rows: {duplicates_count}")
    if duplicates_count > 0:
        lines.append("\nSample duplicates:\n" + duplicates_df.head().to_string())
    lines.append("\n" + "-" * 40 + "\n")
    
    # Outliers (IQR)
    lines.append("IQR-BASED OUTLIERS:\n")
    outliers_iqr_count = len(outliers_iqr_df)
    lines.append(f"Number of outliers: {outliers_iqr_count}")
    if outliers_iqr_count > 0:
        lines.append("\nSample outliers:\n" + outliers_iqr_df.head().to_string())
    lines.append("\n" + "-" * 40 + "\n")
    
    # Outliers (Z-score)
    lines.append("Z-SCORE-BASED OUTLIERS:\n")
    outliers_zscore_count = len(outliers_zscore_df)
    lines.append(f"Number of outliers: {outliers_zscore_count}")
    if outliers_zscore_count > 0:
        lines.append("\nSample outliers:\n" + outliers_zscore_df.head().to_string())
    lines.append("\n" + "-" * 40 + "\n")
    
    # Timestamp gaps
    lines.append("TIMESTAMP GAPS:\n")
    gap_count = len(gap_report)
    lines.append(f"Number of gaps: {gap_count}")
    if gap_count > 0:
        lines.append("\nSample gaps:\n" + gap_report.head().to_string())
    lines.append("\n" + "-" * 40 + "\n")

    return "\n".join(lines)


# -------------------------------------------------------------------
# Main Reporting Function
# -------------------------------------------------------------------
def generate_data_quality_report(file_path: str) -> None:
    """
    Main function that orchestrates the data checks and reporting.
    
    :param file_path: Path to the CSV file containing time-series data.
    """
    # 1) Load Data
    df = load_data(file_path, date_col="timestamp", price_col="last_trade_price", parse_dates=True)
    
    # 2) Data Checks
    missing_df = check_missing_values(df)
    duplicates_df = check_duplicate_rows(df)
    outliers_iqr_df = detect_outliers_iqr(df, "Price")
    outliers_zscore_df = detect_outliers_zscore(df, "Price")
    gap_report = check_timestamp_continuity(df, max_gap=timedelta(days=1))
    
    # 3) Reporting
    # 3a) Textual report
    report_text = generate_textual_report(
        df,
        missing_df,
        duplicates_df,
        outliers_iqr_df,
        outliers_zscore_df,
        gap_report
    )
    
    logger.info("\n" + report_text)

    # 3b) Basic Plots
    plot_price(df, price_col="Price", title="Raw Price Series")
    if not outliers_iqr_df.empty:
        plot_outliers(outliers_iqr_df, df, price_col="Price", method_name="IQR")
    if not outliers_zscore_df.empty:
        plot_outliers(outliers_zscore_df, df, price_col="Price", method_name="Z-score")


# -------------------------------------------------------------------
# Script Entry
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Example usage: relative path to keep it portable
    data_file_path = os.path.join("data", "processed", "back-test-480_cleaned_no_duplicates.csv")

    logger.info(f"Attempting to load data from: {data_file_path}")
    
    # Check if the file exists
    if not os.path.exists(data_file_path):
        logger.error(f"File {data_file_path} does not exist.")
    else:
        generate_data_quality_report(data_file_path)
