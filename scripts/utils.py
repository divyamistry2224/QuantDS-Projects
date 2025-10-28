import pandas as pd

def simple_return(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()
