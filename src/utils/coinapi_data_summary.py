import os
import pandas as pd

# -------------------------------------------------------------------
# Helper to dynamically locate the latest file
# -------------------------------------------------------------------
def get_latest_file(directory: str, prefix: str) -> str:
    """
    Find the latest file in the specified directory with a given prefix.
    
    Args:
        directory (str): The directory to search in.
        prefix (str): The prefix to match file names.

    Returns:
        str: The full path to the latest matching file.
    """
    files = [f for f in os.listdir(directory) if f.startswith(prefix)]
    if not files:
        raise FileNotFoundError(f"No files with prefix '{prefix}' found in {directory}")
    latest_file = max(files, key=lambda f: os.path.getctime(os.path.join(directory, f)))
    return os.path.join(directory, latest_file)

# -------------------------------------------------------------------
# Main script
# -------------------------------------------------------------------
try:
    # Define directory and file prefix
    directory = "data/processed"
    file_prefix = "BITSTAMP_SPOT_BTC_USD_2025-01-11"

    # Locate the latest file
    file_path = get_latest_file(directory, file_prefix)

    # Load the data
    data_sample = pd.read_csv(file_path)

    # Display basic information about the dataset
    summary = {
        "Number of Rows": len(data_sample),
        "Number of Columns": len(data_sample.columns),
        "Columns": data_sample.columns.tolist(),
        "Start Timestamp": data_sample["time_exchange"].min(),
        "End Timestamp": data_sample["time_exchange"].max(),
    }

    print("Data Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")

except FileNotFoundError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
