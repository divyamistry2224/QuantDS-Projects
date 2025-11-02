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
    print(f"📥 Fetching data for {tickers} from Yahoo Finance...")
    data = yf.download(tickers, start=start, end=end, group_by='ticker', auto_adjust=True)

    all_data = []
    for t in tickers:
        df = data[t].copy()
        df["Ticker"] = t
        all_data.append(df)

    df = pd.concat(all_data).reset_index()
    print(f"✅ Downloaded {len(df)} rows for {len(tickers)} tickers.")
    return df


# -----------------------------
# Save dataframe to SQLite database
# -----------------------------
def save_to_db(df, db_path="data/market_data.db"):
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(db_path)
    df.to_sql("historical_prices", conn, if_exists="replace", index=False)
    conn.close()
    print(f"💾 Data saved successfully to {db_path}")
