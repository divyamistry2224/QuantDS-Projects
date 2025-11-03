# =========================================
# main.py
# Quant Research Pipeline Orchestrator
# =========================================

import os
import pandas as pd
import numpy as np

# ===== Import Core Modules =====
from modules.data_handler import fetch_yfinance, save_to_db
from modules.features import process_and_save_features
from modules.portfolio import optimize_portfolio
from modules.time_series import fit_arima, forecast_arima
from modules.volatility import fit_garch, forecast_vol
from modules.evaluation import evaluate_forecast
from modules.report import generate_report

# ==============================
# Setup directories
# ==============================
DATA_DIR = "data/processed"
os.makedirs(DATA_DIR, exist_ok=True)

# ==============================
# Step 1: Fetch & Save Market Data
# ==============================
print("\n📥 Fetching and saving market data...")
tickers = ["AAPL", "MSFT", "GOOGL"]
raw_data = fetch_yfinance(tickers, start="2020-01-01", end="2024-12-31")
save_to_db(raw_data)

# ==============================
# Step 2: Process & Save Features
# ==============================
print("\n⚙️ Processing features (log returns)...")
returns = process_and_save_features(raw_data)

# Ensure we have tickers as columns (transpose if needed)
if returns.shape[0] < returns.shape[1]:
    returns = returns.T

# ==============================
# Step 3: Portfolio Optimization
# ==============================
print("\n💼 Running portfolio optimization...")
# Drop completely empty columns (tickers with all NaNs)
returns_clean = returns.dropna(how="all", axis=1)
weights_df = optimize_portfolio(returns_clean)
weights_path = os.path.join(DATA_DIR, "optimized_weights.csv")
weights_df.to_csv(weights_path, index=False)
print(f"✅ Optimized weights saved to {weights_path}")

# ==============================
# Step 4: ARIMA Forecast
# ==============================
print("\n📈 Running ARIMA forecast for AAPL...")
if "AAPL" in returns_clean.columns:
    arima_model = fit_arima(returns_clean["AAPL"].dropna(), order=(1,1,1))
    arima_forecast = forecast_arima(arima_model, steps=10)
    arima_forecast_path = os.path.join(DATA_DIR, "arima_forecast_aapl.csv")
    arima_forecast.to_csv(arima_forecast_path, index=False)
    print(f"✅ ARIMA forecast saved to {arima_forecast_path}")
else:
    print("⚠️ AAPL not found in returns. Skipping ARIMA forecast.")

# ==============================
# Step 5: GARCH Volatility Forecast
# ==============================
print("\n📊 Estimating volatility using GARCH(1,1)...")
if "AAPL" in returns_clean.columns:
    garch_res = fit_garch(returns_clean["AAPL"].dropna())
    vol_forecast = forecast_vol(garch_res, steps=10)
    print("✅ Forecasted volatility (next 10 days):", vol_forecast)
else:
    print("⚠️ AAPL not found in returns. Skipping GARCH volatility forecast.")

# ==============================
# Step 6: Forecast Evaluation (Example)
# ==============================
print("\n🧮 Evaluating forecast performance...")
true = np.random.normal(0, 0.01, 10)
pred = np.random.normal(0, 0.01, 10)
metrics = evaluate_forecast(true, pred)
print("✅ Forecast evaluation metrics:", metrics)

# ==============================
# Step 7: Generate Report
# ==============================
print("\n🧾 Generating final report...")
if "AAPL" in returns_clean.columns:
    report_path = generate_report(weights_path, returns_clean["AAPL"])
    print(f"✅ Report saved at: {report_path}")
else:
    print("⚠️ AAPL not found. Skipping report generation.")

print("\n✅ All tasks completed successfully!")
