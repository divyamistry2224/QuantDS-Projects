# =========================================
# modules/metrics.py
# Handles financial metrics & return calcs
# =========================================
import numpy as np
import pandas as pd

# -----------------------------
# Compute log returns
# -----------------------------
def log_return(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """
    Calculate log returns: ln(Pt / Pt-1)
    Works for a Series (single asset) or DataFrame (multiple assets)
    """
    return np.log(prices / prices.shift(1)).dropna()

# -----------------------------
# Compute annualized mean return & volatility
# -----------------------------
def annualized_metrics(returns: pd.DataFrame, periods_per_year: int = 252):
    """
    Returns:
        mean_ann: annualized mean return
        vol_ann:  annualized volatility
    """
    mean_ann = returns.mean() * periods_per_year
    vol_ann = returns.std() * np.sqrt(periods_per_year)
    return mean_ann, vol_ann

# -----------------------------
# Compute Sharpe Ratio
# -----------------------------
def sharpe_ratio(returns: pd.DataFrame, risk_free_rate: float = 0.02, periods_per_year: int = 252):
    """
    Calculate annualized Sharpe ratio
    """
    mean_ann, vol_ann = annualized_metrics(returns, periods_per_year)
    excess = mean_ann - risk_free_rate
    return excess / vol_ann
