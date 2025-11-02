import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from modules.metrics import simple_return, log_return, sharpe_ratio

# ----------------------------------------------------------
# Download data
# ----------------------------------------------------------
tickers = ["AAPL", "MSFT", "GOOGL"]
data = yf.download(tickers, start="2023-01-01", end="2024-01-01")

# Extract close prices correctly
if isinstance(data.columns, pd.MultiIndex):
    close_prices = data["Close"]
else:
    close_prices = data["Adj Close"] if "Adj Close" in data.columns else data["Close"]

# ----------------------------------------------------------
# Calculate daily returns and cumulative returns
# ----------------------------------------------------------
daily_returns = close_prices.pct_change().dropna()
cumulative_returns = (1 + daily_returns).cumprod()

# ----------------------------------------------------------
# Plot cumulative returns
# ----------------------------------------------------------
plt.figure(figsize=(10,6))
for ticker in cumulative_returns.columns:
    plt.plot(cumulative_returns.index, cumulative_returns[ticker], label=ticker)

plt.title("Cumulative Returns (2023–2024)")
plt.xlabel("Date")
plt.ylabel("Growth of $1 investment")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save plot
output_path = os.path.join(os.path.dirname(__file__), "../../data/processed/cumulative_returns.png")
plt.savefig(output_path)
print(f"\n✅ Saved cumulative returns chart to {output_path}")
plt.show()