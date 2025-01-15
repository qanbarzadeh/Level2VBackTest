import os
import pandas as pd
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter
from openpyxl.styles import PatternFill

# Checklist data
# Checklist data
checklist_data = {
    "Category": [
        "Backtesting Basics", "Backtesting Basics", "Backtesting Basics", "Backtesting Basics", "Backtesting Basics", "Backtesting Basics", "Backtesting Basics",
        "Market Data Handling", "Market Data Handling", "Market Data Handling", "Market Data Handling",
        "Strategy Design", "Strategy Design", "Strategy Design", "Strategy Design", "Strategy Design",
        "Execution Considerations",
        "Risk Management",
        "Validation", "Validation", "Validation", "Validation", "Validation",
        "Performance Metrics"
    ],
    "Identifier": [
        "Backtesting/1", "Backtesting/2", "Backtesting/3", "Backtesting/4", "Backtesting/5", "Backtesting/6", "Backtesting/7",
        "Market/1", "Market/2", "Market/3", "Market/4",
        "Strategy/1", "Strategy/2", "Strategy/3", "Strategy/4", "Strategy/5",
        "Execution/1",
        "Risk/1",
        "Validation/1", "Validation/2", "Validation/3", "Validation/4", "Validation/5",
        "Metrics/1"
    ],
    "Consideration": [
        "Avoid Look-Ahead Bias", "Address Data-Snooping Bias", "Use Realistic Transaction Costs", "Evaluate Statistical Significance", "Address Randomness in Finite Sample Sizes", "Avoid Backtesting Low Sharpe Ratio Strategies", "Compare to Appropriate Benchmarks",
        "Adjust for Venue-Specific Quotes", "Handle Missing or Outlier Data", "Avoid Survivorship Bias", "Use Survivorship-Bias-Free Datasets",
        "Simplify Models", "Avoid Overfitting in Model Design", "Consider Benchmark Appropriateness", "Use Appropriate Statistical Measures for Strategy Validation", "Avoid Overfitted Models",
        "Account for Latency and Slippage",
        "Add Risk Controls",
        "Use Walk-Forward Testing", "Test for Robustness Using Monte Carlo Simulation", "Use Randomized Trade Testing", "Test for Overfitting with Statistical Significance", "Evaluate High-Frequency Strategies Critically",
        "Track Statistical Significance"
    ],
    "Description/Action": [
        "Ensure no future data is used to generate signals for the current period.",
        "Use out-of-sample testing, cross-validation, and keep models simple with fewer parameters.",
        "Account for slippage, spreads, and fees in the backtesting framework.",
        "Use hypothesis testing to determine whether observed backtest results are statistically significant and not due to randomness. Consider p-values and critical thresholds.",
        "Use statistical significance testing to verify that observed strategy performance is not due to chance. Compute p-values to assess reliability.",
        "Do not backtest strategies with high returns but low Sharpe ratios (<1.0) or long drawdowns, as they are likely inconsistent and prone to data-snooping bias.",
        "Always benchmark strategy performance against simple alternatives like buy-and-hold returns or an information ratio for long-only strategies.",
        "Use historical data from the same venue where trades will execute (e.g., exchange-specific bid-ask spreads).",
        "Clean the dataset to remove erroneous or missing values that could skew results.",
        "Ensure that backtests use a survivorship-bias-free dataset that includes delisted stocks to avoid inflated returns.",
        "Verify datasets include delisted stocks or other excluded entities to avoid inflated backtest results.",
        "Use linear models with minimal parameters to reduce overfitting risks.",
        "Avoid models with excessive parameters (e.g., neural networks with too many nodes) as they may fit historical data but lack predictive power.",
        "Always compare strategies to appropriate benchmarks. For long-only strategies, use buy-and-hold returns and information ratios rather than Sharpe ratios.",
        "Apply statistical measures like Sharpe ratio, Sortino ratio, and profit factor to validate strategy robustness.",
        "Avoid strategies that overfit historical data (e.g., overly complex neural networks with many parameters) as they lack predictive power in live trading.",
        "Simulate execution delays and price impacts due to order size and market conditions.",
        "Implement stop-loss, take-profit, and position-sizing mechanisms in backtesting and live trading.",
        "Sequentially test strategies on unseen data to validate out-of-sample performance.",
        "Use Monte Carlo simulations to generate simulated price data with the same moments as observed data. Run the strategy on this simulated data to test its robustness.",
        "Randomize trade entry dates while keeping the number of trades and holding periods fixed. Evaluate whether strategy performance persists under randomized conditions.",
        "Validate strategies with statistical significance tests to detect overfitting. Ensure robustness through out-of-sample testing.",
        "Be cautious when backtesting high-frequency strategies, as success depends heavily on market microstructure and live execution dynamics. Simulations may fail to capture these effects.",
        "Evaluate Sharpe ratio, Sortino ratio, max drawdown, and profit factor to validate performance."
    ],
    "Status": [
        "⚠️ Pending", "⚠️ Pending", "⚠️ Pending", "⚠️ Pending", "⚠️ Pending", "⚠️ Pending", "⚠️ Pending",
        "⚠️ Pending", "⚠️ Pending", "⚠️ Pending", "⚠️ Pending",
        "⚠️ Pending", "⚠️ Pending", "⚠️ Pending", "⚠️ Pending", "⚠️ Pending",
        "⚠️ Pending",
        "⚠️ Pending",
        "⚠️ Pending", "⚠️ Pending", "⚠️ Pending", "⚠️ Pending", "⚠️ Pending",
        "⚠️ Pending"
    ]
}



# Define file paths
output_dir = "./docs/checklists"
output_file = "algorithmic_trading_checklist.xlsx"
os.makedirs(output_dir, exist_ok=True)
file_path = os.path.join(output_dir, output_file)

# Create DataFrame
checklist_df = pd.DataFrame(checklist_data)

# Save to Excel with auto-adjusted column widths and color-coding
with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
    checklist_df.to_excel(writer, index=False, sheet_name='Checklist')
    worksheet = writer.sheets['Checklist']
    
    # Define category colors
    category_colors = {
        "Backtesting Basics": "B4C7E7",  # Light Blue
        "Market Data Handling": "A9D08E",  # Light Green
        "Strategy Design": "FFE699",  # Light Yellow
        "Execution Considerations": "F4B084",  # Light Orange
        "Risk Management": "D9E2F3",  # Light Purple
        "Validation": "FABF8F",  # Peach
        "Performance Metrics": "FFD966"  # Yellow
    }

    # Apply colors
    for row_idx, category in enumerate(checklist_df['Category'], start=2):
        fill_color = category_colors.get(category, "FFFFFF")  # Default white
        for col_idx in range(1, len(checklist_df.columns) + 1):
            worksheet.cell(row=row_idx, column=col_idx).fill = PatternFill(start_color=fill_color, end_color=fill_color, fill_type="solid")

    # Adjust column widths
    for col_num, column in enumerate(checklist_df.columns, 1):
        col_letter = get_column_letter(col_num)
        max_length = max(checklist_df[column].astype(str).apply(len).max(), len(column)) + 2
        worksheet.column_dimensions[col_letter].width = max_length

print(f"Checklist successfully updated with enhanced readability and saved to {file_path}")
