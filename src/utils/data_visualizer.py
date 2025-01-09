"""
data_visualizer.py

A utility module for visualizing Level 2 order book data. This module
provides several plotting functions to help analyze key metrics such
as bid-ask spread, order flow imbalance, liquidity distribution, and
overall price trends.
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Optional
from src.utils.data_loader import DataLoader

# Ensure the results directory for visualizations exists
RESULTS_DIR = os.path.join("results", "visualizations")
try:
    os.makedirs(RESULTS_DIR, exist_ok=True)
except OSError as e:
    print(f"Error creating directory {RESULTS_DIR}: {e}")


def plot_bid_ask_spread(data: pd.DataFrame) -> None:
    """
    Plot the bid-ask spread (ask_price_1 - bid_price_1) over time and save as a PNG.

    Args:
        data (pd.DataFrame): The loaded Level 2 data as a pandas DataFrame.
    """
    # Calculate bid-ask spread
    data['bid_ask_spread'] = data['ask_price_1'] - data['bid_price_1']

    plt.figure(figsize=(12, 6))
    plt.plot(data['timestamp'], data['bid_ask_spread'], label='Bid-Ask Spread', color='blue')
    plt.title('Bid-Ask Spread Over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Spread')
    plt.legend()
    plt.tight_layout()

    # Save plot
    out_file = os.path.join(RESULTS_DIR, "bid_ask_spread.png")
    plt.savefig(out_file)
    plt.close()
    print(f"Bid-Ask Spread plot saved to: {out_file}")


def plot_order_flow_imbalance(data: pd.DataFrame, levels: int = 10) -> None:
    """
    Calculate and plot the Order Flow Imbalance (OFI) over time:

        OFI = (Total Bid Volume - Total Ask Volume) / (Total Bid Volume + Total Ask Volume)

    Args:
        data (pd.DataFrame): The loaded Level 2 data as a pandas DataFrame.
        levels (int): Number of bid/ask levels to use in the OFI calculation.
                      Default is 10. Adjust as needed.
    """
    # Calculate total bid and ask volumes for the specified levels
    bid_cols = [f"bid_size_{i}" for i in range(1, levels + 1) if f"bid_size_{i}" in data.columns]
    ask_cols = [f"ask_size_{i}" for i in range(1, levels + 1) if f"ask_size_{i}" in data.columns]

    data['total_bid_volume'] = data[bid_cols].sum(axis=1)
    data['total_ask_volume'] = data[ask_cols].sum(axis=1)

    # Avoid zero division
    data['ofi_denominator'] = data['total_bid_volume'] + data['total_ask_volume']
    data['ofi_denominator'] = data['ofi_denominator'].replace({0: float('nan')})

    data['ofi'] = (data['total_bid_volume'] - data['total_ask_volume']) / data['ofi_denominator']

    plt.figure(figsize=(12, 6))
    plt.plot(data['timestamp'], data['ofi'], label='Order Flow Imbalance', color='red')
    plt.title(f'Order Flow Imbalance (Levels = {levels})')
    plt.xlabel('Timestamp')
    plt.ylabel('OFI')
    plt.legend()
    plt.tight_layout()

    # Save plot
    out_file = os.path.join(RESULTS_DIR, "order_flow_imbalance.png")
    plt.savefig(out_file)
    plt.close()
    print(f"Order Flow Imbalance plot saved to: {out_file}")


def plot_liquidity_heatmap(data: pd.DataFrame) -> None:
    """
    Create and save a heatmap of bid/ask volumes across all levels over time.

    Args:
        data (pd.DataFrame): The loaded Level 2 data as a pandas DataFrame.
    """
    # Identify all bid_size_x and ask_size_x columns
    bid_size_cols = [col for col in data.columns if col.startswith("bid_size_")]
    ask_size_cols = [col for col in data.columns if col.startswith("ask_size_")]

    # Combine bid/ask volumes in a single DataFrame for heatmap
    heatmap_data = []

    for col in bid_size_cols:
        level_str = col.split("_")[-1]
        for idx, row in data.iterrows():
            heatmap_data.append([row['timestamp'], f"Bid_{level_str}", row[col]])

    for col in ask_size_cols:
        level_str = col.split("_")[-1]
        for idx, row in data.iterrows():
            heatmap_data.append([row['timestamp'], f"Ask_{level_str}", row[col]])

    heatmap_df = pd.DataFrame(heatmap_data, columns=['timestamp', 'level', 'volume'])
    pivot_df = heatmap_df.pivot(index='level', columns='timestamp', values='volume')

    plt.figure(figsize=(14, 8))
    sns.heatmap(pivot_df, cmap="viridis", norm=None)
    plt.title("Liquidity Heatmap (Bid & Ask Volumes Across Levels)")
    plt.xlabel("Timestamp")
    plt.ylabel("Level")
    plt.tight_layout()

    # Save plot
    out_file = os.path.join(RESULTS_DIR, "liquidity_heatmap.png")
    plt.savefig(out_file)
    plt.close()
    print(f"Liquidity heatmap saved to: {out_file}")


def plot_price_trends(data: pd.DataFrame) -> None:
    """
    Plot the last_trade_price over time, and overlay with average bid/ask prices.

    Args:
        data (pd.DataFrame): The loaded Level 2 data as a pandas DataFrame.
    """
    bid_price_cols = [col for col in data.columns if col.startswith("bid_price_")]
    ask_price_cols = [col for col in data.columns if col.startswith("ask_price_")]

    data['avg_bid_price'] = data[bid_price_cols].mean(axis=1)
    data['avg_ask_price'] = data[ask_price_cols].mean(axis=1)

    plt.figure(figsize=(12, 6))
    plt.plot(data['timestamp'], data['last_trade_price'], label='Last Trade Price', color='green')
    plt.plot(data['timestamp'], data['avg_bid_price'], label='Avg Bid Price', color='blue')
    plt.plot(data['timestamp'], data['avg_ask_price'], label='Avg Ask Price', color='red')
    plt.title('Price Trends Over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()

    # Save plot
    out_file = os.path.join(RESULTS_DIR, "price_trends.png")
    plt.savefig(out_file)
    plt.close()
    print(f"Price trends plot saved to: {out_file}")


def plot_volume_concentration(data: pd.DataFrame) -> None:
    """
    Visualize total bid and ask volumes at each timestamp.

    Args:
        data (pd.DataFrame): The loaded Level 2 data as a pandas DataFrame.
    """
    bid_size_cols = [col for col in data.columns if col.startswith("bid_size_")]
    ask_size_cols = [col for col in data.columns if col.startswith("ask_size_")]

    data['total_bid_volume'] = data[bid_size_cols].sum(axis=1)
    data['total_ask_volume'] = data[ask_size_cols].sum(axis=1)

    plt.figure(figsize=(12, 6))
    plt.plot(data['timestamp'], data['total_bid_volume'], label='Total Bid Volume', color='blue')
    plt.plot(data['timestamp'], data['total_ask_volume'], label='Total Ask Volume', color='red')
    plt.title('Volume Concentration Over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Volume')
    plt.legend()
    plt.tight_layout()

    # Save plot
    out_file = os.path.join(RESULTS_DIR, "volume_concentration.png")
    plt.savefig(out_file)
    plt.close()
    print(f"Volume concentration plot saved to: {out_file}")


if __name__ == '__main__':
    """
    Demonstration script for data visualization. Assumes that DataLoader
    and a sample CSV file are available in your environment.
    """
    loader = DataLoader()
    sample_csv_path = 'data/processed/back-test-480_cleaned.csv'

    try:
        df = loader.load_csv(sample_csv_path)

        # Generate plots
        plot_bid_ask_spread(df)
        plot_order_flow_imbalance(df, levels=10)
        plot_liquidity_heatmap(df)
        plot_price_trends(df)
        plot_volume_concentration(df)

    except FileNotFoundError:
        print(f"Could not find file: {sample_csv_path}")
    except ValueError as e:
        print(f"Data validation error: {e}")
