# fetch_data.py
"""
Fetches daily stock data using yfinance and saves it as CSV in the data folder.
"""

import yfinance as yf
import os
from datetime import datetime

# Create 'data' directory if it doesn't exist
os.makedirs("../data", exist_ok=True)

# Choose your stock ticker
TICKER = "AAPL"  # You can change to 'GOOG', 'MSFT', 'TSLA', etc.

# Fetch historical data
print(f"Fetching data for {TICKER}...")
data = yf.download(TICKER, start="2020-01-01", end=datetime.today().strftime("%Y-%m-%d"), auto_adjust=True)

# Save to CSV
file_path = f"../data/{TICKER}_data.csv"
data.to_csv(file_path)
print(f"✅ Data saved successfully at: {file_path}")
