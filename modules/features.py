# =========================================
# modules/features.py
# Process and Save Financial Features
# =========================================

import os
import pandas as pd
import numpy as np

# ---------------------------------------------------
# Function: process_and_save_features
# ---------------------------------------------------
def process_and_save_features(data, folder_path="data/processed"):
    """
    Process raw market data into useful financial features.
    Specifically computes daily log returns and saves them to CSV.

    Parameters
    ----------
    data : pd.DataFrame
        Market data with tickers as columns and date as index.
    folder_path : str
        Directory where processed features will be saved.

    Returns
    -------
    pd.DataFrame
        DataFrame of daily log returns.
    """

    # Ensure output directory exists
    os.makedirs(folder_path, exist_ok=True)

    # Validate input
    if data is None or data.empty:
        raise ValueError("❌ Input market data is empty. Please fetch data first.")

    # ---------------------------------------------------
    # 0️⃣ Select only numeric columns (avoid strings like 'Ticker')
    # ---------------------------------------------------
    price_data = data.select_dtypes(include=['float', 'int'])
    if price_data.empty:
        raise ValueError("❌ No numeric columns found for return calculation.")

    # ---------------------------------------------------
    # 1️⃣ Compute Daily Log Returns
    # ---------------------------------------------------
    print("⚙️ Computing daily log returns...")
    returns = np.log(price_data / price_data.shift(1))
    returns = returns.dropna(how="all", axis=1)

    # ---------------------------------------------------
    # 2️⃣ Save Returns to CSV
    # ---------------------------------------------------
    returns_path = os.path.join(folder_path, "daily_returns.csv")
    returns.to_csv(returns_path)
    print(f"✅ Processed features saved to {returns_path}")

    # ---------------------------------------------------
    # 3️⃣ Return DataFrame for Next Steps
    # ---------------------------------------------------
    return returns
