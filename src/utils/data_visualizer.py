import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List

# ----------------------------------------------------------------------
# (Assuming DataLoader is optional; we can integrate its logic here or
# adapt it. If you're using src.utils.data_loader, update that similarly.)
# ----------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

RESULTS_DIR = os.path.join("results", "visualizations")
try:
    os.makedirs(RESULTS_DIR, exist_ok=True)
except OSError as e:
    print(f"Error creating directory {RESULTS_DIR}: {e}")

# ----------------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------------
def rename_timestamp_column(data: pd.DataFrame) -> pd.DataFrame:
    """
    If 'timestamp' doesn't exist, but 'time_exchange' or 'time_coinapi' do,
    rename one of them to 'timestamp' for consistency.
    Priority is 'time_exchange' -> 'time_coinapi' -> fallback remains as is.
    """
    if 'timestamp' not in data.columns:
        if 'time_exchange' in data.columns:
            data.rename(columns={'time_exchange': 'timestamp'}, inplace=True)
            logging.info("Renamed 'time_exchange' to 'timestamp'.")
        elif 'time_coinapi' in data.columns:
            data.rename(columns={'time_coinapi': 'timestamp'}, inplace=True)
            logging.info("Renamed 'time_coinapi' to 'timestamp'.")
        else:
            logging.warning(
                "No 'timestamp', 'time_exchange', or 'time_coinapi' found. "
                "Plots might not be time-based."
            )
    return data

def detect_level_count(data: pd.DataFrame) -> int:
    """
    Detect the maximum level of bid/ask in the dataset by scanning
    for columns named 'bid_price_X' or 'ask_price_X'.
    """
    level = 0
    # For example, if columns are bid_price_1 ... bid_price_5, we get 5
    while True:
        level += 1
        test_bid_col = f"bid_price_{level}"
        test_ask_col = f"ask_price_{level}"
        if test_bid_col not in data.columns and test_ask_col not in data.columns:
            # If NEITHER exist, we've reached the last level
            return level - 1

def _prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Common pre-processing:
      - Rename a suitable column to 'timestamp' if missing
      - Sort by 'timestamp' if available
      - Convert numeric columns
      - Drop NaN rows in critical columns (if present)
    """
    data = rename_timestamp_column(data)

    # Sort if we have 'timestamp' now
    if 'timestamp' in data.columns:
        data = data.sort_values('timestamp').reset_index(drop=True)

    level_count = detect_level_count(data)

    # Build up the list of numeric columns that actually exist
    numeric_cols = []
    for prefix in ['bid_price_', 'ask_price_', 'bid_size_', 'ask_size_']:
        for lvl in range(1, level_count + 1):
            col_name = f"{prefix}{lvl}"
            if col_name in data.columns:
                numeric_cols.append(col_name)

    # If last_trade_price exists, convert it too
    if 'last_trade_price' in data.columns:
        numeric_cols.append('last_trade_price')

    # Convert all numeric columns
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # We only *require* at least one level, e.g. bid_price_1 and ask_price_1,
    # but we won't treat them as a hard error if missing—just log a warning.
    critical_cols = ['bid_price_1', 'ask_price_1']
    for c in critical_cols:
        if c not in data.columns:
            logging.warning(f"Column '{c}' is missing in the dataset.")
            continue

        # Drop rows where these critical columns are NaN
        original_len = len(data)
        data = data.dropna(subset=[c])
        if len(data) < original_len:
            logging.info(f"Dropped {original_len - len(data)} rows due to NaN in '{c}'.")

    return data

# ----------------------------------------------------------------------
# PLOT FUNCTIONS
# ----------------------------------------------------------------------
def plot_bid_ask_spread(data: pd.DataFrame) -> None:
    """
    Plot the spread using 'ask_price_1' - 'bid_price_1' if they exist.
    """
    data = _prepare_data(data)
    if 'ask_price_1' not in data.columns or 'bid_price_1' not in data.columns:
        logging.error("Cannot plot bid-ask spread: missing ask_price_1 or bid_price_1.")
        return

    data['bid_ask_spread'] = data['ask_price_1'] - data['bid_price_1']

    # Optionally save debug CSV
    debug_file = os.path.join(RESULTS_DIR, "bid_ask_spread_data.csv")
    if 'timestamp' in data.columns:
        data[['timestamp', 'bid_ask_spread']].to_csv(debug_file, index=False)
    else:
        # If no time-based column, just save the entire data
        data.to_csv(debug_file, index=False)
    logging.info(f"Bid-Ask Spread data saved to: {debug_file}")

    plt.figure(figsize=(12, 6))
    if 'timestamp' in data.columns:
        plt.plot(data['timestamp'], data['bid_ask_spread'], label='Bid-Ask Spread', color='blue')
        plt.xlabel('Timestamp')
    else:
        plt.plot(data['bid_ask_spread'], label='Bid-Ask Spread', color='blue')
        plt.xlabel('Row Index')

    plt.title('Bid-Ask Spread Over Time')
    plt.ylabel('Spread')
    plt.legend()
    plt.tight_layout()

    out_file = os.path.join(RESULTS_DIR, "bid_ask_spread.png")
    plt.savefig(out_file)
    plt.close()
    logging.info(f"Bid-Ask Spread plot saved to: {out_file}")

def plot_order_flow_imbalance(data: pd.DataFrame) -> None:
    """
    Plot the order flow imbalance (OFI) across existing levels.
    """
    data = _prepare_data(data)
    level_count = detect_level_count(data)

    # Collect the actual bid_size_N / ask_size_N columns that exist
    bid_cols = [f"bid_size_{i}" for i in range(1, level_count + 1) if f"bid_size_{i}" in data.columns]
    ask_cols = [f"ask_size_{i}" for i in range(1, level_count + 1) if f"ask_size_{i}" in data.columns]

    if not bid_cols or not ask_cols:
        logging.error("No valid bid_size_ or ask_size_ columns found; cannot plot OFI.")
        return

    # Sum across the levels that do exist
    data['total_bid_volume'] = data[bid_cols].sum(axis=1, skipna=True)
    data['total_ask_volume'] = data[ask_cols].sum(axis=1, skipna=True)

    # OFI = (bid_vol - ask_vol) / (bid_vol + ask_vol)
    data['ofi_denominator'] = data['total_bid_volume'] + data['total_ask_volume']
    data['ofi_denominator'] = data['ofi_denominator'].replace({0: float('nan')})
    data['ofi'] = (data['total_bid_volume'] - data['total_ask_volume']) / data['ofi_denominator']

    debug_file = os.path.join(RESULTS_DIR, "order_flow_imbalance_data.csv")
    save_cols = ['total_bid_volume', 'total_ask_volume', 'ofi']
    if 'timestamp' in data.columns:
        save_cols.insert(0, 'timestamp')
    data[save_cols].to_csv(debug_file, index=False)
    logging.info(f"Order Flow Imbalance data saved to: {debug_file}")

    plt.figure(figsize=(12, 6))
    if 'timestamp' in data.columns:
        plt.plot(data['timestamp'], data['ofi'], label='Order Flow Imbalance', color='red')
        plt.xlabel('Timestamp')
    else:
        plt.plot(data['ofi'], label='Order Flow Imbalance', color='red')
        plt.xlabel('Row Index')

    plt.title(f'Order Flow Imbalance (Detected Levels = {level_count})')
    plt.ylabel('OFI')
    plt.legend()
    plt.tight_layout()

    out_file = os.path.join(RESULTS_DIR, "order_flow_imbalance.png")
    plt.savefig(out_file)
    plt.close()
    logging.info(f"Order Flow Imbalance plot saved to: {out_file}")

def plot_liquidity_heatmap(data: pd.DataFrame) -> None:
    """
    Create a heatmap of volumes across existing levels and timestamps.
    """
    data = _prepare_data(data)
    level_count = detect_level_count(data)

    # We only care about size columns for the heatmap
    bid_size_cols = [col for col in data.columns if col.startswith("bid_size_")]
    ask_size_cols = [col for col in data.columns if col.startswith("ask_size_")]

    if not bid_size_cols and not ask_size_cols:
        logging.error("No bid_size_ or ask_size_ columns found; cannot plot heatmap.")
        return

    heatmap_data = []
    # Convert each row into a flattened form: (timestamp, "Bid_i"/"Ask_i", volume)
    for idx, row in data.iterrows():
        # We'll pick time-based label if available
        time_label = row['timestamp'] if 'timestamp' in data.columns else idx

        # Bids
        for col in bid_size_cols:
            level_str = col.split("_")[-1]  # e.g., "3"
            heatmap_data.append([time_label, f"Bid_{level_str}", row[col]])

        # Asks
        for col in ask_size_cols:
            level_str = col.split("_")[-1]
            heatmap_data.append([time_label, f"Ask_{level_str}", row[col]])

    heatmap_df = pd.DataFrame(heatmap_data, columns=['time_label', 'level', 'volume'])

    # Pivot so that rows = level, columns = time_label
    pivot_df = heatmap_df.pivot(index='level', columns='time_label', values='volume')

    debug_file = os.path.join(RESULTS_DIR, "liquidity_heatmap_data.csv")
    pivot_df.to_csv(debug_file)
    logging.info(f"Liquidity Heatmap pivot data saved to: {debug_file}")

    plt.figure(figsize=(14, 8))
    sns.heatmap(pivot_df, cmap="viridis")
    plt.title("Liquidity Heatmap (Bid & Ask Volumes Across Levels)")
    plt.xlabel("Time" if 'timestamp' in data.columns else "Row Index")
    plt.ylabel("Level")
    plt.tight_layout()

    out_file = os.path.join(RESULTS_DIR, "liquidity_heatmap.png")
    plt.savefig(out_file)
    plt.close()
    logging.info(f"Liquidity heatmap saved to: {out_file}")

def plot_price_trend(data: pd.DataFrame) -> None:
    """
    Plot a simple price trend over time.
    If 'last_trade_price' is missing, it will plot the mid-price from bid/ask_1.
    """
    data = _prepare_data(data)  # use the same pre-processing

    if 'last_trade_price' in data.columns:
        # We have a last_trade_price column
        price_col = 'last_trade_price'
        title = 'Last Trade Price Over Time'
    elif all(col in data.columns for col in ['bid_price_1', 'ask_price_1']):
        # No last_trade_price, but can compute mid-price
        data['mid_price'] = (data['bid_price_1'] + data['ask_price_1']) / 2.0
        price_col = 'mid_price'
        title = 'Mid-Price (Bid/Ask) Over Time'
    else:
        logging.error("No suitable price columns found for a price trend plot.")
        return

    # Optional: remove NaNs
    data = data.dropna(subset=[price_col])

    plt.figure(figsize=(12, 6))
    if 'timestamp' in data.columns:
        plt.plot(data['timestamp'], data[price_col], color='green', label=price_col)
        plt.xlabel('Timestamp')
    else:
        # Fallback if no time column
        plt.plot(data[price_col], color='green', label=price_col)
        plt.xlabel('Row Index')

    plt.ylabel('Price')
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    # Save figure
    out_file = os.path.join(RESULTS_DIR, "price_trend.png")
    plt.savefig(out_file)
    plt.close()
    logging.info(f"Price trend plot saved to: {out_file}")


# ----------------------------------------------------------------------
# OPTIONAL MAIN for Testing
# ----------------------------------------------------------------------
if __name__ == '__main__':
    # Example usage
    sample_csv_path = 'data/processed/BITSTAMP_SPOT_BTC_USD_2025-01-11_08-27-12.csv'
    if not os.path.exists(sample_csv_path):
        logging.error(f"Sample file does not exist: {sample_csv_path}")
    else:
        df = pd.read_csv(sample_csv_path)
        plot_bid_ask_spread(df)
        plot_order_flow_imbalance(df)
        plot_liquidity_heatmap(df)
        plot_price_trend(df)

