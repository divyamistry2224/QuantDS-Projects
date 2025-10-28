# =========================================
# modules/volatility.py
# GARCH Volatility Modeling
# =========================================

from arch import arch_model
import pandas as pd
import matplotlib.pyplot as plt

def fit_garch(returns):
    """Fit a basic GARCH(1,1) model to return series."""
    model = arch_model(returns.dropna(), vol='Garch', p=1, q=1)
    res = model.fit(disp='off')
    return res

def forecast_vol(res, steps=5):
    """Forecast volatility (variance) over given steps."""
    forecast = res.forecast(horizon=steps)
    return forecast.variance.values[-1, :]


# Test helper (optional)
if __name__ == "__main__":
    rets = pd.read_csv('data/processed/daily_returns.csv', index_col=0)
    res = fit_garch(rets['AAPL'])
    forecast = forecast_vol(res, steps=10)
    print("Forecasted Variance:", forecast)

    # Compare visually
    rolling_vol = rets['AAPL'].rolling(window=20).std()
    plt.figure(figsize=(10, 5))
    plt.plot(rolling_vol, label='Rolling Std (20D)')
    plt.plot(res.conditional_volatility, label='GARCH Volatility')
    plt.legend()
    plt.title("AAPL Volatility: GARCH vs Rolling Std")
    plt.tight_layout()
    plt.savefig('data/processed/volatility_comparison.png')
    plt.close()
    print("✅ Saved comparison chart: data/processed/volatility_comparison.png")
