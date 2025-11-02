# =========================================
# app.py
# Streamlit Live Quant Research Dashboard (Full Version)
# =========================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io, zipfile

from modules.data_handler import fetch_yfinance, save_to_db
from modules.features import process_and_save_features
from modules.portfolio import optimize_portfolio
from modules.time_series import fit_arima, forecast_arima
from modules.volatility import fit_garch, forecast_vol
from modules.evaluation import evaluate_forecast

st.set_page_config(page_title="Quant Research Dashboard", layout="wide")
st.title("📊 Quantitative Research Live Dashboard")

# -----------------------------
# 1️⃣ Sidebar Inputs
# -----------------------------
tickers_input = st.sidebar.text_input("Tickers (comma-separated)", "AAPL,MSFT,GOOGL")
tickers = [t.strip().upper() for t in tickers_input.split(",")]

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-12-31"))
forecast_steps = st.sidebar.number_input("Forecast horizon (days)", min_value=1, max_value=30, value=10)

st.sidebar.write("---")
fetch_button = st.sidebar.button("Fetch & Process Data")
portfolio_button = st.sidebar.button("Run Portfolio Optimization")
arima_button = st.sidebar.button("Run ARIMA Forecasts")
garch_button = st.sidebar.button("Run GARCH Volatility Forecasts")

# -----------------------------
# 2️⃣ Fetch Market Data
# -----------------------------
if fetch_button:
    st.header("📥 Fetching Market Data")
    with st.spinner("Fetching data from Yahoo Finance..."):
        raw_data = fetch_yfinance(tickers, start=start_date.strftime("%Y-%m-%d"),
                                  end=end_date.strftime("%Y-%m-%d"))
        save_to_db(raw_data)
        returns = process_and_save_features(raw_data)
        returns_clean = returns.dropna(how="all", axis=1)
    st.success("✅ Data fetched and processed!")
    st.subheader("Raw Market Data Preview")
    st.dataframe(raw_data.head())
    st.subheader("Daily Log Returns Preview")
    st.dataframe(returns_clean.tail())

# Ensure we have returns if user skipped fetch button
if 'returns' in locals():
    returns_clean = returns.dropna(how="all", axis=1)

# -----------------------------
# 3️⃣ Portfolio Optimization
# -----------------------------
if portfolio_button:
    if 'returns_clean' in locals():
        st.header("💼 Portfolio Optimization")
        weights_df = optimize_portfolio(returns_clean)
        st.success("✅ Portfolio optimization complete.")
        st.subheader("Optimized Portfolio Weights")
        st.dataframe(weights_df)

        # Pie chart
        st.subheader("Portfolio Allocation")
        fig, ax = plt.subplots()
        ax.pie(weights_df["Weight"], labels=weights_df["Ticker"], autopct='%1.1f%%', startangle=140)
        ax.axis('equal')
        st.pyplot(fig)

        # Download CSV
        st.download_button("Download Portfolio Weights CSV", weights_df.to_csv(index=False), "optimized_weights.csv")
    else:
        st.warning("⚠️ Please fetch data first!")

# -----------------------------
# Helper functions for ARIMA/GARCH
# -----------------------------
def get_arima_forecast(series, steps):
    model = fit_arima(series)
    forecast = forecast_arima(model, steps)
    return forecast

def get_garch_forecast(series, steps):
    res = fit_garch(series)
    vol_forecast = forecast_vol(res, steps)
    return vol_forecast

# -----------------------------
# 4️⃣ ARIMA Forecasts
# -----------------------------
if arima_button:
    if 'returns_clean' in locals():
        st.header("📈 ARIMA Forecasts (Multi-Ticker)")

        combined_forecast = pd.DataFrame(index=range(forecast_steps))
        for t in tickers:
            st.subheader(f"{t} ARIMA Forecast")
            try:
                forecast = get_arima_forecast(returns_clean[t], forecast_steps)
                combined_forecast[t] = forecast["Forecast"].values
                st.line_chart(forecast)
            except Exception as e:
                st.error(f"❌ ARIMA forecast failed for {t}: {e}")

        # Download all ARIMA forecasts as ZIP
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zf:
            for t in combined_forecast.columns:
                csv_bytes = combined_forecast[[t]].to_csv(index=False).encode()
                zf.writestr(f"{t}_arima_forecast.csv", csv_bytes)
        st.download_button("Download All ARIMA Forecasts as ZIP", zip_buffer.getvalue(), "arima_forecasts.zip")
    else:
        st.warning("⚠️ Please fetch data first!")

# -----------------------------
# 5️⃣ GARCH Volatility Forecasts
# -----------------------------
if garch_button:
    if 'returns_clean' in locals():
        st.header("📊 GARCH Volatility Forecasts (Multi-Ticker)")

        combined_vol = pd.DataFrame(index=range(forecast_steps))
        for t in tickers:
            st.subheader(f"{t} GARCH Volatility")
            try:
                vol_forecast = get_garch_forecast(returns_clean[t], forecast_steps)
                combined_vol[t] = vol_forecast
                st.line_chart(pd.DataFrame(vol_forecast, columns=["Volatility"]))
            except Exception as e:
                st.error(f"❌ GARCH forecast failed for {t}: {e}")

        # Download all GARCH forecasts as ZIP
        zip_buffer_vol = io.BytesIO()
        with zipfile.ZipFile(zip_buffer_vol, "a", zipfile.ZIP_DEFLATED) as zf:
            for t in combined_vol.columns:
                csv_bytes = combined_vol[[t]].to_csv(index=False).encode()
                zf.writestr(f"{t}_garch_vol.csv", csv_bytes)
        st.download_button("Download All GARCH Volatilities as ZIP", zip_buffer_vol.getvalue(), "garch_volatilities.zip")
    else:
        st.warning("⚠️ Please fetch data first!")

# -----------------------------
# 6️⃣ Footer
# -----------------------------
st.write("---")
st.write("⚡ This dashboard runs your live Quant Research pipeline interactively. Data is fetched, processed, and forecasts are displayed in real-time.")
