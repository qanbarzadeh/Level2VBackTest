"""
data_loader.py

A utility module for loading and validating processed Level 2 data from CSV files.

Example Usage:
    from utils.data_loader import DataLoader

    if __name__ == '__main__':
        loader = DataLoader(levels=10)  # e.g., support 10 levels
        df = loader.load_csv('data/processed/back-test-480_cleaned.csv')
        print(df.head())

Author: Your Name
Date: 2025-01-05
"""

import os
import logging
import pandas as pd
from typing import Union, List


# Configure logging
LOG_FILE_PATH = os.path.join('logs', 'data_loader.log')
logging.basicConfig(
    level=logging.INFO,  # Set the default logging level
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),  # Log to file
        logging.StreamHandler()             # Log to console
    ]
)

class DataLoader:
    """A class to load and validate Level 2 order book data from a CSV file.

    This class provides functionality to:
      - Read CSV files containing up to N levels of bid and ask data.
      - Validate columns, formats, and sorting.
      - Return the data in a pandas DataFrame for fast access.

    Attributes:
        levels (int): The number of price levels for bids and asks.
        required_columns (List[str]): A list of columns that must be present in the CSV.
    """

    def __init__(self, levels: int = 10) -> None:
        """Initialize the DataLoader with a configurable number of levels.

        Args:
            levels (int): Number of bid/ask price levels in the dataset.
                          Defaults to 24 if not specified.
        """
        self.levels = levels
        self.required_columns = self._generate_required_columns()
        logging.info(f"DataLoader initialized with {self.levels} levels and required columns.")

    def _generate_required_columns(self) -> List[str]:
        """Generate a list of required columns for bid/ask prices and sizes up to `self.levels`.

        Returns:
            List[str]: The list of required column names.
        """
        columns = ["timestamp"]
        # Add bid columns
        for i in range(1, self.levels + 1):
            columns.append(f"bid_price_{i}")
            columns.append(f"bid_size_{i}")
        # Add ask columns
        for i in range(1, self.levels + 1):
            columns.append(f"ask_price_{i}")
            columns.append(f"ask_size_{i}")
        # Add last_trade_price
        columns.append("last_trade_price")
        return columns

    def load_csv(self, file_path: str) -> pd.DataFrame:
        """Load Level 2 data from a CSV file and return it as a pandas DataFrame.

        Args:
            file_path (str): Path to the CSV file (e.g., 'data/processed/my_data.csv').

        Returns:
            pd.DataFrame: A DataFrame containing the validated Level 2 data.

        Raises:
            FileNotFoundError: If the specified CSV file does not exist.
            ValueError: If required columns are missing, data is unsorted, or contains invalid values.
        """
        logging.info(f"Attempting to load CSV file from: {file_path}")

        # Check if file exists
        if not os.path.isfile(file_path):
            error_msg = f"CSV file not found at: {file_path}"
            logging.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Load CSV using pandas
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            error_msg = f"Failed to read CSV file '{file_path}': {e}"
            logging.error(error_msg)
            raise ValueError(error_msg)

        # Validate columns
        self._validate_columns(df)

        # Parse timestamps
        self._parse_timestamps(df)

        # Validate numeric and non-negative columns
        self._validate_numeric(df)

        # Ensure data is sorted by timestamp
        self._validate_sorting(df)

        # Final logging before return
        logging.info("Data successfully loaded and validated.")
        return df

    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Check if all required columns are present in the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to validate.

        Raises:
            ValueError: If any required column is missing.
        """
        missing_cols = [col for col in self.required_columns if col not in df.columns]
        if missing_cols:
            error_msg = f"Missing required columns: {missing_cols}"
            logging.error(error_msg)
            raise ValueError(error_msg)
        logging.info("All required columns are present.")

    def _parse_timestamps(self, df: pd.DataFrame) -> None:
        """Convert the 'timestamp' column to pandas datetime format.

        Args:
            df (pd.DataFrame): The DataFrame with a 'timestamp' column.

        Raises:
            ValueError: If the 'timestamp' column contains invalid date formats.
        """
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, infer_datetime_format=True)
        except Exception as e:
            error_msg = f"Invalid timestamp format in 'timestamp' column: {e}"
            logging.error(error_msg)
            raise ValueError(error_msg)
        logging.info("Timestamp column successfully parsed to datetime.")

    def _validate_numeric(self, df: pd.DataFrame) -> None:
        """Validate that price and size columns (and last_trade_price) are numeric and non-negative.

        Args:
            df (pd.DataFrame): The DataFrame containing numeric columns.

        Raises:
            ValueError: If any numeric column contains negative values or non-numeric data.
        """
        numeric_cols = [col for col in self.required_columns if col != "timestamp"]
        for col in numeric_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                error_msg = f"Column '{col}' must be numeric."
                logging.error(error_msg)
                raise ValueError(error_msg)

            # Check non-negative
            if (df[col] < 0).any():
                error_msg = f"Column '{col}' contains negative values."
                logging.error(error_msg)
                raise ValueError(error_msg)

        logging.info("All numeric columns validated as non-negative and numeric.")

    def _validate_sorting(self, df: pd.DataFrame) -> None:
        """Check if the DataFrame is sorted by ascending timestamp.

        Args:
            df (pd.DataFrame): The DataFrame to check.

        Raises:
            ValueError: If 'timestamp' is not sorted in ascending order.
        """
        if not df['timestamp'].is_monotonic_increasing:
            error_msg = "Data is not sorted in ascending order by timestamp."
            logging.error(error_msg)
            raise ValueError(error_msg)
        logging.info("Data is sorted in ascending order by timestamp.")


if __name__ == '__main__':
    """Demonstration script for the DataLoader."""
    # Example usage of DataLoader with configurable levels
    loader = DataLoader(levels=10)  # Support for 10 levels of bid/ask
    try:
        sample_csv_path = 'data/processed/back-test-480_cleaned.csv'
        data = loader.load_csv(sample_csv_path)
        print(data.head())
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading data: {e}")
