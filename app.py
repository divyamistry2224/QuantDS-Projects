# =========================================
# app.py
# Streamlit Live Quant Research Dashboard with Session State
# =========================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from modules.data_handler import fetch_yfinance, save_to_db
from modules.features import process_and_save_features
from modules.portfolio import optimize_portfolio
from modules.time_series import fit_arima, forecast_arima
from modules.volatility import fit_garch, forecast_vol
from modules.evaluation import evaluate_forecast

# -----------------------------
# Page Config
# -----------------------------
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

# -----------------------------
# Initialize Session State
# -----------------------------
if "raw_data" not in st.session_state:
    st.session_state.raw_data = None
if "returns" not in st.session_state:
    st.session_state.returns = None
if "weights_df" not in st.session_state:
    st.session_state.weights_df = None

# -----------------------------
# 2️⃣ Fetch & Process Market Data
# -----------------------------
if st.sidebar.button("Fetch & Process Data"):
    st.info("Fetching market data...")
    st.session_state.raw_data = fetch_yfinance(tickers, start=start_date.strftime("%Y-%m-%d"),
                                               end=end_date.strftime("%Y-%m-%d"))
    save_to_db(st.session_state.raw_data)
    st.session_state.returns = process_and_save_features(st.session_state.raw_data)
    st.success("✅ Data fetched and processed!")

# Show preview if available
if st.session_state.raw_data is not None:
    st.subheader("Raw Market Data Preview")
    st.dataframe(st.session_state.raw_data.head())

if st.session_state.returns is not None:
    st.subheader("Daily Log Returns Preview")
    st.dataframe(st.session_state.returns.tail())

# -----------------------------
# 3️⃣ Portfolio Optimization
# -----------------------------
if st.sidebar.button("Run Portfolio Optimization"):
    if st.session_state.returns is not None:
        st.session_state.weights_df = optimize_portfolio(st.session_state.returns)
        st.success("✅ Portfolio optimization complete!")

        st.subheader("Optimized Portfolio Weights")
        st.dataframe(st.session_state.weights_df)

        st.subheader("Portfolio Allocation")
        fig, ax = plt.subplots()
        ax.pie(st.session_state.weights_df["Weight"], labels=st.session_state.weights_df["Ticker"],
               autopct='%1.1f%%', startangle=140)
        ax.axis('equal')
        st.pyplot(fig)

        st.download_button("Download Weights CSV",
                           st.session_state.weights_df.to_csv(index=False),
                           "optimized_weights.csv")
    else:
        st.warning("⚠️ Please fetch data first!")

# -----------------------------
# 4️⃣ ARIMA Forecast
# -----------------------------
if st.sidebar.button("Run ARIMA Forecast"):
    if st.session_state.returns is not None:
        st.subheader("ARIMA Forecasts")
        for t in tickers:
            try:
                model = fit_arima(st.session_state.returns[t])
                forecast = forecast_arima(model, steps=forecast_steps)
                st.write(f"ARIMA Forecast for {t}")
                st.line_chart(forecast)
                st.download_button(f"Download {t} Forecast CSV",
                                   forecast.to_csv(index=False),
                                   f"{t}_forecast.csv")
            except Exception as e:
                st.error(f"❌ ARIMA forecast failed for {t}: {e}")
    else:
        st.warning("⚠️ Please fetch data first!")

# -----------------------------
# 5️⃣ GARCH Volatility Forecast
# -----------------------------
if st.sidebar.button("Run GARCH Volatility"):
    if st.session_state.returns is not None:
        st.subheader("GARCH Volatility Forecasts")
        for t in tickers:
            try:
                res = fit_garch(st.session_state.returns[t])
                vol_forecast = forecast_vol(res, steps=forecast_steps)
                st.write(f"GARCH Volatility Forecast for {t}")
                st.line_chart(vol_forecast)
            except Exception as e:
                st.error(f"❌ GARCH forecast failed for {t}: {e}")
    else:
        st.warning("⚠️ Please fetch data first!")

# -----------------------------
# 6️⃣ Footer / Notes
# -----------------------------
st.write("---")
st.write("⚡ This dashboard runs your live Quant Research pipeline interactively. "
         "Data is fetched, processed, and forecasts are displayed in real-time.")
