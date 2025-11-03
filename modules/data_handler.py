# =========================================
# modules/data_handler.py
# Handles data fetching and database saving
# =========================================

import yfinance as yf
import pandas as pd
import sqlite3
import os

# -----------------------------
# Fetch data from Yahoo Finance
# -----------------------------
def fetch_yfinance(tickers, start, end):
    """
    Fetch adjusted close prices for multiple tickers from Yahoo Finance.
    
    Returns:
        pd.DataFrame: index=Date, columns=tickers, values=Adj Close prices
    """
    print(f"📥 Fetching adjusted close prices for {tickers} from Yahoo Finance...")
    data = yf.download(tickers, start=start, end=end, group_by='ticker', auto_adjust=True)
    
    all_close = {}
    for t in tickers:
        try:
            # Some tickers may fail, so handle individually
            all_close[t] = data[t]['Close']
        except Exception as e:
            print(f"⚠️ Could not fetch {t}: {e}")

    df_close = pd.DataFrame(all_close)
    df_close.index = pd.to_datetime(df_close.index)
    print(f"✅ Downloaded adjusted close prices for {len(df_close)} dates.")
    return df_close

# -----------------------------
# Save dataframe to SQLite database
# -----------------------------
def save_to_db(df, db_path="data/market_data.db"):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    df.to_sql("historical_prices", conn, if_exists="replace", index=True)
    conn.close()
    print(f"💾 Data saved successfully to {db_path}")
