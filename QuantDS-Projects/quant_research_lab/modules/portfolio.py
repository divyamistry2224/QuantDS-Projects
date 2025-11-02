# =========================================
# modules/portfolio.py
# Portfolio Optimization (Mean-Variance)
# =========================================

import pandas as pd
import numpy as np
from scipy.optimize import minimize

# ---------------------------
# 1️⃣ Portfolio Performance
# ---------------------------
def portfolio_performance(weights, mean_returns, cov_matrix):
    """Return portfolio return and volatility."""
    port_return = np.dot(weights, mean_returns)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return port_return, port_vol

# ---------------------------
# 2️⃣ Objective Function (Negative Sharpe)
# ---------------------------
def neg_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=0.0):
    ret, vol = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(ret - risk_free_rate) / vol

# ---------------------------
# 3️⃣ Optimize Portfolio
# ---------------------------
def optimize_portfolio(returns):
    """Optimize portfolio using mean-variance (maximize Sharpe ratio)."""
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_assets = len(mean_returns)
    init_weights = np.ones(num_assets) / num_assets

    bounds = tuple((0, 1) for _ in range(num_assets))
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

    result = minimize(
        neg_sharpe,
        init_weights,
        args=(mean_returns, cov_matrix),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
    )

    optimal_weights = result.x
    tickers = returns.columns
    optimized_df = pd.DataFrame({'Ticker': tickers, 'Weight': optimal_weights})
    optimized_df.to_csv('data/processed/optimized_weights.csv', index=False)
    print("✅ Optimized portfolio saved to data/processed/optimized_weights.csv")

    return optimized_df
