# =========================================
# Day 6 – Portfolio Optimization (Markowitz)
# =========================================
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.returns import simple_return
from modules.optimizer import optimize_portfolio

# 1️⃣ Download stock data
tickers = ["AAPL", "MSFT", "GOOGL"]
print(f"📥 Downloading data for {tickers}...")
data = yf.download(tickers, start="2023-01-01", end="2024-01-01", group_by='ticker')

# 2️⃣ Extract Close prices
close_prices = pd.concat([data[t]['Close'] for t in tickers], axis=1)
close_prices.columns = tickers

# 3️⃣ Compute daily returns
returns = simple_return(close_prices)

# 4️⃣ Optimize portfolio
weights, mean_returns, cov_matrix = optimize_portfolio(returns)

print("\n📊 Optimal Portfolio Weights:")
for t, w in zip(tickers, weights):
    print(f"{t}: {w:.3f}")

# 5️⃣ Save optimized weights
os.makedirs("data", exist_ok=True)
output_path = "data/optimized_weights.csv"
pd.DataFrame({"Ticker": tickers, "Weight": weights}).to_csv(output_path, index=False)
print(f"\n✅ Saved optimized weights to {output_path}")

# 6️⃣ Efficient Frontier Visualization
print("\n📈 Generating Efficient Frontier...")
port_returns, port_vols = [], []
for _ in range(1000):
    w = np.random.dirichlet(np.ones(len(mean_returns)))
    r = np.dot(w, mean_returns)
    v = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
    port_returns.append(r)
    port_vols.append(v)

plt.figure(figsize=(8,6))
plt.scatter(port_vols, port_returns, c=(np.array(port_returns)/np.array(port_vols)),
            cmap='viridis', s=10)
plt.title("Efficient Frontier")
plt.xlabel("Volatility (Std Dev)")
plt.ylabel("Expected Return")
plt.colorbar(label='Sharpe Ratio')
plt.show()

print("\n🎯 Day 6 completed successfully!")
