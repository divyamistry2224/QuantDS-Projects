# =========================================
# modules/portfolio.py
# Portfolio Optimization Module
# =========================================

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ---------------------------------------------------
# Function: neg_sharpe
# ---------------------------------------------------
def neg_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=0.0):
    """
    Compute negative Sharpe ratio for optimization.
    """
    port_return = np.dot(weights, mean_returns)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return -(port_return - risk_free_rate) / port_vol

# ---------------------------------------------------
# Function: optimize_portfolio
# ---------------------------------------------------
def optimize_portfolio(returns):
    """
    Optimize portfolio weights using the Sharpe ratio.
    
    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame of daily returns (rows = dates, columns = tickers)

    Returns
    -------
    pd.DataFrame
        DataFrame of optimized weights with columns ['Ticker', 'Weight']
    """
    print("⚙️ Starting fast & robust portfolio optimization...")

    # -------------------------
    # 1️⃣ Clean Data
    # -------------------------
    returns = returns.dropna(how="all", axis=0)  # drop empty rows
    returns = returns.dropna(how="all", axis=1)  # drop empty columns

    if returns.empty:
        raise ValueError("❌ Returns DataFrame is empty after cleaning.")

    # -------------------------
    # 2️⃣ Prepare Mean & Covariance
    # -------------------------
    # If returns have tickers as columns, use that directly
    mean_returns = returns.mean()
    cov_matrix = returns.cov() + np.eye(len(mean_returns)) * 1e-6  # regularization

    num_assets = len(mean_returns)
    initial_weights = np.ones(num_assets) / num_assets
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    # -------------------------
    # 3️⃣ Run Optimization
    # -------------------------
    try:
        result = minimize(
            neg_sharpe,
            initial_weights,
            args=(mean_returns, cov_matrix),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"disp": False, "maxiter": 100},
        )

        if not result.success:
            raise ValueError("Optimization failed.")

        weights = result.x

    except Exception as e:
        print(f"⚠️ Optimization fallback due to error: {e}")
        weights = initial_weights  # fallback to equal weights

    # -------------------------
    # 4️⃣ Return Weights DataFrame
    # -------------------------
    weights_df = pd.DataFrame({
        "Ticker": mean_returns.index,
        "Weight": weights
    })

    print("✅ Portfolio optimization complete.")
    return weights_df
