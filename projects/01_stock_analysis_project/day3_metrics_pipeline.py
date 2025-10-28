import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from modules.metrics import simple_return, sharpe_ratio

# ----------------------------------------------------------
# 1. Download stock data
# ----------------------------------------------------------
tickers = ["AAPL", "MSFT", "GOOGL"]
data = yf.download(tickers, start="2023-01-01", end="2024-01-01")
print("Columns returned from yfinance:\n", data.columns)

# Handle both MultiIndex and single-index structures
if isinstance(data.columns, pd.MultiIndex):
    if "Close" in data.columns.get_level_values(0):
        close_prices = data["Close"]
    elif "Adj Close" in data.columns.get_level_values(0):
        close_prices = data["Adj Close"]
    else:
        raise ValueError("No valid price column found in yfinance output.")
else:
    close_prices = data["Close"] if "Close" in data.columns else data["Adj Close"]

# ----------------------------------------------------------
# 2. Compute returns
# ----------------------------------------------------------
daily_returns = close_prices.pct_change().dropna()

# ----------------------------------------------------------
# 3. Compute cumulative returns
# ----------------------------------------------------------
cumulative_returns = (1 + daily_returns).cumprod() - 1

# ----------------------------------------------------------
# 4. Compute Sharpe ratios
# ----------------------------------------------------------
sharpe_ratios = {ticker: sharpe_ratio(daily_returns[ticker]) for ticker in daily_returns.columns}

# ----------------------------------------------------------
# 5. Visualizations (Save Instead of Show)
# ----------------------------------------------------------
plt.figure(figsize=(10, 6))
sns.lineplot(data=cumulative_returns)
plt.title("Cumulative Returns (2023)")
plt.ylabel("Cumulative Return")
plt.xlabel("Date")
plt.legend(cumulative_returns.columns)
plt.grid(True)
plt.tight_layout()

# Save plot instead of showing
fig_path = os.path.join(os.path.dirname(__file__), "../../data/processed/cumulative_returns.png")
plt.savefig(fig_path)

# ----------------------------------------------------------
# 6. Save Sharpe ratios
# ----------------------------------------------------------
sharpe_path = os.path.join(os.path.dirname(__file__), "../../data/processed/sharpe_ratios_day4.csv")
pd.Series(sharpe_ratios).to_csv(sharpe_path)

# ----------------------------------------------------------
# 7. Confirm completion
# ----------------------------------------------------------
print("\n✅ Visualization complete.")
print(f"✅ Chart saved at: {fig_path}")
print(f"✅ Sharpe ratios saved at: {sharpe_path}")
