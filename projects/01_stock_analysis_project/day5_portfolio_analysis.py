import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.returns import simple_return, log_return

# ----------------------------------------------------------
# 1️⃣ Download stock data
# ----------------------------------------------------------
tickers = ["AAPL", "MSFT", "GOOGL"]
data = yf.download(tickers, start="2023-01-01", end="2024-01-01")

# Handle multi-index columns
if isinstance(data.columns, pd.MultiIndex):
    close_prices = data["Close"]
else:
    close_prices = data["Adj Close"] if "Adj Close" in data.columns else data["Close"]

# ----------------------------------------------------------
# 2️⃣ Calculate daily returns
# ----------------------------------------------------------
daily_returns = simple_return(close_prices)

# ----------------------------------------------------------
# 3️⃣ Portfolio setup (equal weights)
# ----------------------------------------------------------
weights = np.array([1/len(tickers)] * len(tickers))  # Equal weights for each stock
portfolio_daily_returns = daily_returns.dot(weights)

# ----------------------------------------------------------
# 4️⃣ Portfolio metrics
# ----------------------------------------------------------
mean_return = portfolio_daily_returns.mean() * 252  # Annualized
volatility = portfolio_daily_returns.std() * np.sqrt(252)  # Annualized
sharpe_ratio = mean_return / volatility

print("\n📊 Portfolio Performance Summary:")
print(f"Annualized Return: {mean_return:.2%}")
print(f"Annualized Volatility: {volatility:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

# ----------------------------------------------------------
# 5️⃣ Plot cumulative portfolio growth
# ----------------------------------------------------------
cumulative_portfolio = (1 + portfolio_daily_returns).cumprod()

plt.figure(figsize=(10,6))
plt.plot(cumulative_portfolio.index, cumulative_portfolio, label="Portfolio", color='purple', linewidth=2)
plt.title("Portfolio Cumulative Growth (2023–2024)")
plt.xlabel("Date")
plt.ylabel("Growth of $1 Investment")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save and show
output_path = os.path.join(os.path.dirname(__file__), "../../data/processed/portfolio_growth.png")
plt.savefig(output_path)
print(f"\n✅ Saved portfolio chart to {output_path}")
plt.show()
