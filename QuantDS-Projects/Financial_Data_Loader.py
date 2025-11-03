import pandas as pd          # For handling data tables (DataFrames)
import yfinance as yf        # To download stock data from Yahoo Finance
import os                    # To work with file paths (like folder + file name)

class FinancialDataLoader:
    def __init__(self, data_dir=None):
        # 'data_dir' is the folder where your CSV files are stored
        self.data_dir = data_dir

    # Function to load data from a local CSV file
    def load_from_csv(self, filename):
        """Get data from a local CSV file"""
        # If a folder is given, join it with the file name
        path = os.path.join(self.data_dir, filename) if self.data_dir else filename
        try:
            # Read CSV with pandas
            data = pd.read_csv(path, parse_dates=True, index_col=0)
            print(f"Loaded data from {filename}")
            return data
        except FileNotFoundError:
            print("CSV file not found.")
            return None

    # Function to load data from Yahoo Finance
    def load_from_yfinance(self, ticker, start, end):
        """Get data from Yahoo Finance"""
        try:
            # Download stock data (like SPY)
            data = yf.download(ticker, start=start, end=end)
            print(f"Downloaded {ticker} data from Yahoo Finance.")
            return data
        except Exception as e:
            print("Error fetching data:", e)
            return None

# The part below runs only when this file is run directly
if __name__ == "__main__":
    # Create the loader (you can keep your CSVs in a folder named 'data')
    loader = FinancialDataLoader(data_dir="data")

    # Example 1: Try loading from a local CSV file (optional)
    csv_data = loader.load_from_csv("SPY.csv")

    # Example 2: Download SPY data from Yahoo Finance
    spy_data = loader.load_from_yfinance("SPY", "2023-01-01", "2025-01-01")

    # Show first few rows of data
    print(spy_data.head())

    # Save downloaded data to CSV for later use
    spy_data.to_csv("SPY_data.csv")
    print("SPY data saved as SPY_data.csv")
