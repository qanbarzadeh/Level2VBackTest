#!/usr/bin/env python3

"""
remove_duplicates.py

A script to remove duplicate rows from the dataset and save the cleaned file.

Usage:
    python -m src.utils.remove_duplicates
"""

import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

# Input and output paths
INPUT_FILE = "data/processed/back-test-480_cleaned.csv"
OUTPUT_FILE = "data/processed/back-test-480_cleaned_no_duplicates.csv"

def remove_duplicates(file_path: str, output_path: str) -> None:
    """
    Load the dataset, remove duplicate rows, and save the cleaned file.

    Args:
        file_path (str): Path to the input CSV file.
        output_path (str): Path to save the cleaned CSV file.
    """
    if not os.path.exists(file_path):
        logger.error(f"Input file does not exist: {file_path}")
        return

    logger.info(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Initial number of rows: {len(df)}")

    # Remove duplicates
    df_cleaned = df.drop_duplicates()
    logger.info(f"Number of rows after removing duplicates: {len(df_cleaned)}")

    # Save the cleaned dataset
    df_cleaned.to_csv(output_path, index=False)
    logger.info(f"Cleaned data saved to: {output_path}")

if __name__ == "__main__":
    remove_duplicates(INPUT_FILE, OUTPUT_FILE)
