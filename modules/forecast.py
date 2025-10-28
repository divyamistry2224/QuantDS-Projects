# =========================================
# modules/forecast.py
# Time Series Forecasting (ARIMA, VAR)
# =========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR
from modules.time_series import make_stationary

# ---------------------------
# 1️⃣ Univariate Forecast (ARIMA)
# ---------------------------

def arima_forecast(series, order=(1, 1, 1), steps=5, plot=True):
    """Fit an ARIMA model and forecast future values."""
    series = series.dropna()
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)

    if plot:
        plt.figure(figsize=(8, 4))
        plt.plot(series.index[-50:], series.values[-50:], label='Historical')
        future_index = pd.date_range(start=series.index[-1], periods=steps + 1, freq='B')[1:]
        plt.plot(future_index, forecast, color='red', label='Forecast')
        plt.title('ARIMA Forecast')
        plt.legend()
        plt.tight_layout()
        plt.savefig('data/arima_forecast.png')
        plt.close()

    return forecast


# ---------------------------
# 2️⃣ Multivariate Forecast (VAR)
# ---------------------------

def var_forecast(df, lags=1, steps=5, plot=True):
    """Fit a VAR model to multiple time series and forecast jointly."""
    df = df.dropna()
    stationary_df = make_stationary(df)
    model = VAR(stationary_df)
    model_fit = model.fit(lags)
    forecast = model_fit.forecast(stationary_df.values[-lags:], steps=steps)
    forecast_df = pd.DataFrame(forecast, columns=stationary_df.columns)

    if plot:
        plt.figure(figsize=(8, 4))
        for col in forecast_df.columns[:3]:  # plot first 3 assets for clarity
            plt.plot(forecast_df[col], label=f'{col} Forecast')
        plt.title('VAR Forecast (first 3 assets)')
        plt.legend()
        plt.tight_layout()
        plt.savefig('data/var_forecast.png')
        plt.close()

    return forecast_df
