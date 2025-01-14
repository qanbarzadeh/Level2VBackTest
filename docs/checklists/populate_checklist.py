import os
import pandas as pd
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter

# Checklist data
checklist_data = {
    "Category": [
        "Backtesting Basics", "Backtesting Basics", "Backtesting Basics",
        "Market Data Handling", "Market Data Handling",
        "Strategy Design", "Execution Considerations", "Risk Management",
        "Validation", "Performance Metrics"
    ],
    "Consideration": [
        "Avoid Look-Ahead Bias", "Address Data-Snooping Bias", "Use Realistic Transaction Costs",
        "Adjust for Venue-Specific Quotes", "Handle Missing or Outlier Data",
        "Simplify Models", "Account for Latency and Slippage",
        "Add Risk Controls", "Use Walk-Forward Testing", "Track Statistical Significance"
    ],
    "Description/Action": [
        "Ensure no future data is used to generate signals for the current period.",
        "Use out-of-sample testing, cross-validation, and keep models simple with fewer parameters.",
        "Account for slippage, spreads, and fees in the backtesting framework.",
        "Use historical data from the same venue where trades will execute (e.g., exchange-specific bid-ask spreads).",
        "Clean the dataset to remove erroneous or missing values that could skew results.",
        "Use linear models with minimal parameters to reduce overfitting risks.",
        "Simulate execution delays and price impacts due to order size and market conditions.",
        "Implement stop-loss, take-profit, and position-sizing mechanisms in backtesting and live trading.",
        "Sequentially test strategies on unseen data to validate out-of-sample performance.",
        "Evaluate Sharpe ratio, Sortino ratio, max drawdown, and profit factor to validate performance."
    ],
    "Status": ["Pending"] * 10
}

# Define file paths
output_dir = "./docs/checklists"
output_file = "algorithmic_trading_checklist.xlsx"
os.makedirs(output_dir, exist_ok=True)
file_path = os.path.join(output_dir, output_file)

# Create DataFrame
checklist_df = pd.DataFrame(checklist_data)

# Save to Excel with auto-adjusted column widths
with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
    checklist_df.to_excel(writer, index=False, sheet_name='Checklist')
    worksheet = writer.sheets['Checklist']
    
    # Adjust column widths
    for col_num, column in enumerate(checklist_df.columns, 1):
        col_letter = get_column_letter(col_num)
        max_length = max(checklist_df[column].astype(str).apply(len).max(), len(column)) + 2
        worksheet.column_dimensions[col_letter].width = max_length

print(f"Checklist successfully saved to {file_path}")
