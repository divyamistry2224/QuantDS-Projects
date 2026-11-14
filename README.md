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

```


## Market Direction Prediction Using Sentiment and Economic Indicators

An end-to-end financial machine learning research project combining public market data, economic-news sentiment and macroeconomic indicators. Includes CatBoost, Random Forest and additional model comparisons, time-aligned features, chronological validation, transaction-cost-aware backtests, and an interactive dashboard.

**[Open the live dashboard](https://divyamistry2224.github.io/QuantDS-Projects/market-direction/)** · **[Project source & detailed README](market-direction-prediction/)** · [Research note (PDF)](market-direction-prediction/reports/market_research_note.pdf) · [Executed notebook](market-direction-prediction/notebooks/research_walkthrough.ipynb)

Measured outcomes include **+12.15% historical net return** for the validation-selected original strategy over 2022–2025, and **53.53% balanced accuracy** for the enhanced CatBoost on the fresh 2026 sample. Periods, baselines, costs and limitations are documented; historical profitability is distinct from proof of future returns.
