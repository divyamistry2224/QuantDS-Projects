import pandas as pd
import numpy as np

def simple_return(prices):
    """Calculate simple daily returns for a DataFrame or Series of prices."""
    return prices.pct_change().dropna()

def log_return(prices):
    """Calculate log returns for a DataFrame or Series of prices."""
    return np.log(prices / prices.shift(1)).dropna()

