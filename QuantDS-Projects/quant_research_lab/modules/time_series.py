# =========================================
# modules/time_series.py
# Stationarity Tests + ARIMA Helpers
# =========================================

import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA

# ---------------------------
# 1️⃣ Stationarity Tests
# ---------------------------
def adf_test(series):
    """Perform Augmented Dickey-Fuller test."""
    result = adfuller(series.dropna())
    return {'ADF Statistic': result[0], 'p-value': result[1]}

def kpss_test(series):
    """Perform KPSS test."""
    result = kpss(series.dropna(), regression='c')
    return {'KPSS Statistic': result[0], 'p-value': result[1]}

# ---------------------------
# 2️⃣ Stationarize Time Series
# ---------------------------
def make_stationary(series):
    """Convert a time series to stationary form by differencing."""
    return series.diff().dropna()

# ---------------------------
# 3️⃣ ARIMA Fit + Forecast
# ---------------------------
def fit_arima(series, order=(1, 1, 1)):
    """Fit an ARIMA model to a single time series."""
    model = ARIMA(series.dropna(), order=order)
    fitted = model.fit()
    return fitted

def forecast_arima(model, steps=5):
    """Forecast using a fitted ARIMA model."""
    forecast = model.forecast(steps=steps)
    return forecast
