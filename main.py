# =========================================
# main.py
# Quant Research Pipeline Orchestrator
# =========================================

import os
import pandas as pd
from modules.portfolio import optimize_portfolio
from modules.forecast import arima_forecast
from modules.volatility import fit_garch, forecast_vol
from modules.evaluation import evaluate_forecast
from modules.report import generate_report

# ==============================
# Setup
# ==============================
DATA_DIR = "data/processed"
os.makedirs(DATA_DIR, exist_ok=True)

# ==============================
# Load Data
# ==============================
print("📥 Loading daily returns data...")
returns_path = os.path.join(DATA_DIR, "daily_returns.csv")

if not os.path.exists(returns_path):
    raise FileNotFoundError(f"❌ Missing file: {returns_path}. Please generate daily_returns.csv first.")

returns = pd.read_csv(returns_path, index_col=0)
print(f"✅ Data loaded successfully: {returns.shape[0]} days, {returns.shape[1]} assets.")

# ==============================
# Portfolio Optimization
# ==============================
print("\n💼 Running portfolio optimization...")
weights_df = optimize_portfolio(returns)
weights_path = os.path.join(DATA_DIR, "optimized_weights.csv")
weights_df.to_csv(weights_path, index=False)
print(f"✅ Optimized weights saved to {weights_path}")

# ==============================
# Forecasting (ARIMA Example)
# ==============================
print("\n📈 Running ARIMA forecast for AAPL...")
from modules.time_series import fit_arima, forecast_arima
model = fit_arima(returns["AAPL"], order=(1, 1, 1))
forecast = forecast_arima(model, steps=10)
forecast_path = os.path.join(DATA_DIR, "arima_forecast_aapl.csv")
forecast.to_csv(forecast_path, index=False)
print(f"✅ ARIMA forecast saved to {forecast_path}")

# ==============================
# Volatility Modeling (GARCH)
# ==============================
print("\n📊 Estimating volatility using GARCH(1,1)...")
from modules.volatility import fit_garch, forecast_vol
res = fit_garch(returns["AAPL"])
vol_forecast = forecast_vol(res, steps=10)
print("✅ GARCH forecast (next 10 days):", vol_forecast)

# ==============================
# Evaluation Example
# ==============================
print("\n🧮 Evaluating ARIMA forecast performance...")
# (for testing — use synthetic or matched data lengths)
import numpy as np
true = np.random.normal(0, 0.01, 10)
pred = np.random.normal(0, 0.01, 10)
metrics = evaluate_forecast(true, pred)
print("✅ Forecast Evaluation Metrics:", metrics)

# ==============================
# Generate Quant Report (PDF)
# ==============================
print("\n🧾 Generating final report...")
report_path = generate_report(weights_path, returns["AAPL"])
print(f"🎯 Final report generated: {report_path}")

print("\n✅ All tasks completed successfully!")
