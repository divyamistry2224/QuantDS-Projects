# =========================================
# modules/optimizer.py
# Portfolio Optimization (Markowitz Framework)
# =========================================

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ---- Portfolio math ----
def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate=0.0):
    weights = np.array(weights)
    port_return = np.dot(weights, mean_returns)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = (port_return - risk_free_rate) / port_vol if port_vol != 0 else 0
    return port_return, port_vol, sharpe

# ---- Objective functions ----
def min_variance(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))

    result = minimize(
        lambda w: portfolio_performance(w, mean_returns, cov_matrix)[1],
        num_assets * [1/num_assets],
        bounds=bounds,
        constraints=constraints
    )
    return result.x

def max_sharpe(mean_returns, cov_matrix, risk_free_rate=0.0):
    num_assets = len(mean_returns)
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))

    def neg_sharpe(w):
        return -portfolio_performance(w, mean_returns, cov_matrix, risk_free_rate)[2]

    result = minimize(
        neg_sharpe,
        num_assets * [1/num_assets],
        bounds=bounds,
        constraints=constraints
    )
    return result.x

# ---- Efficient frontier ----
def efficient_frontier(mean_returns, cov_matrix, return_targets):
    results = {'Return': [], 'Volatility': [], 'Sharpe': []}
    for target in return_targets:
        num_assets = len(mean_returns)
        constraints = (
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: np.dot(w, mean_returns) - target}
        )
        bounds = tuple((0, 1) for _ in range(num_assets))
        result = minimize(
            lambda w: np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))),
            num_assets * [1/num_assets],
            bounds=bounds,
            constraints=constraints
        )
        if result.success:
            r, s, sh = portfolio_performance(result.x, mean_returns, cov_matrix)
            results['Return'].append(r)
            results['Volatility'].append(s)
            results['Sharpe'].append(sh)
    return pd.DataFrame(results)

# ---- Plotting helper ----
def plot_efficient_frontier(frontier_df):
    plt.figure(figsize=(8,5))
    plt.plot(frontier_df['Volatility'], frontier_df['Return'], 'o-', label='Efficient Frontier')
    plt.xlabel('Volatility (σ)')
    plt.ylabel('Expected Return (μ)')
    plt.title('Efficient Frontier')
    plt.legend()
    plt.grid(True)
    plt.show()

# ---- Wrapper for main pipeline ----
def optimize_portfolio(returns, risk_free_rate=0.0):
    """
    Wrapper to optimize portfolio using the Max Sharpe Ratio method.
    Compatible with main.py pipeline.
    """
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    weights = max_sharpe(mean_returns, cov_matrix, risk_free_rate)
    return weights, mean_returns, cov_matrix
