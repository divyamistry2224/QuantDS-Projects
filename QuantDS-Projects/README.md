# QuantDS-Projects

A collection of Python-based projects for Quantitative Data Science, combining object-oriented programming, finance APIs, and data analysis.

Project 1: Financial Data Loader

This project demonstrates Object-Oriented Programming (OOP) in Python by building a reusable class for loading financial data from multiple sources.

🔹 Features
- Load data from local CSV files.
- Fetch historical stock data using **Yahoo Finance (yfinance API)**.
- Includes **error handling** for missing files or invalid ticker symbols.
- Simple and modular — can be reused in larger Quantitative Research workflows.

Example 

```python
from Financial_Data_Loader import FinancialDataLoader

# Initialize the data loader
loader = FinancialDataLoader(data_dir="data")

# Load local CSV
csv_data = loader.load_from_csv("SPY.csv")

# Fetch SPY data from Yahoo Finance
spy_data = loader.load_from_yfinance("SPY", "2023-01-01", "2025-01-01")

# Save data for future analysis
spy_data.to_csv("SPY_data.csv")
