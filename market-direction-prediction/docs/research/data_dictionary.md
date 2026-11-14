# Data dictionary

## Snapshots

Original snapshots are under `data/raw`. Extension snapshots are under `experiments/v2/data/raw`. Each has a `provenance.json` with source URLs, retrieval times, response hashes, normalized-file hashes and row counts. Snapshots include warm-up history from 2004; original feature rows start in 2006. Cross-asset inception dates and rolling features determine extension row availability.

`prices.csv`: `date`, `open`, `high`, `low`, `close`, `volume`, `adj_close`. Prices are USD per SPY share; volume is reported shares. Yahoo timestamps are normalized to America/New_York session dates. `adj_close` is the provider's current adjustment for corporate actions/distributions.

`macro.csv`: `observation_date`, `value`, `series`. DFF and T10Y2Y are percent/percentage-point series, VIXCLS is volatility index points, UNRATE is percent, CPIAUCSL is an index level. A zero or negative yield-curve spread is valid.

`sentiment.csv`: `observation_date`, `news_sentiment`. Unitless published daily economic-news sentiment from FRBSF.

`cross_assets.csv` (extension only): `date`, `symbol`, `adj_close`. Symbols: QQQ, IWM, TLT, HYG, GLD, UUP. Five-session adjusted returns are used; no future or cross-sectional target is joined.

## Original predictors

| Feature | Definition / unit |
|---|---|
| `return_1d`, `return_5d`, `return_20d` | Trailing adjusted-close fractional returns |
| `volatility_20d` | Rolling sample standard deviation of daily adjusted returns × sqrt(252) |
| `distance_sma50`, `distance_sma200` | Adjusted close / trailing simple moving average −1 |
| `intraday_return` | Current session close / open −1 |
| `overnight_return` | Current open / previous raw close −1 |
| `high_low_range` | High / low −1 |
| `volume_z20` | Current log(1+volume), standardized over trailing 20 sessions |
| `fed_funds`, `yield_curve`, `vix`, `unemployment` | Last available values under the documented lag and staleness rules |
| `inflation_yoy` | CPIAUCSL percentage change over 12 observations ×100, then availability-lagged |
| `fed_funds_change20`, `yield_curve_change20` | Twenty-session difference in lagged levels |
| `vix_change5` | Five-session difference in lagged VIX |
| `unemployment_change60` | Sixty-session difference in lagged unemployment |
| `news_sentiment` | Last available lagged sentiment value |
| `sentiment_mean5`, `sentiment_mean20` | Trailing session averages of available sentiment |
| `sentiment_change5` | Five-session sentiment difference |
| `sentiment_z60` | Sentiment standardized over trailing 60 sessions |

## Enhanced predictors

| Feature | Definition |
|---|---|
| `rsi2`, `rsi14` | RSI from exponentially averaged gains/losses, alpha=1/n; initialization differs from vendor implementations |
| `close_location` | (Close − low) / (high − low) |
| `vol_ratio` | Five-session return standard deviation / twenty-session standard deviation |
| `downside_vol20` | Root mean square of negative daily returns over 20 sessions |
| `return_lag2`, `return_lag3`, `return_lag4` | Previous-session daily returns at lags 1, 2, 3 relative to current return |
| `weekday` | Signal-date weekday, Monday=0 |
| `month_end_distance` | Calendar days from signal date to month end |
| `*_return5` | Five-session adjusted return of each of the six context ETFs |
| `smallcap_relative5` | IWM five-session return − SPY five-session return |
| `credit_relative5` | HYG five-session return − TLT five-session return |
| `sentiment_x_trend` | Available news sentiment × SPY distance from 200-session SMA; full extension feature set only |

All predictors are derived causally through the signal date. Full-data normalization is not used. Logistic scaling is fit on each training fold only. No future observation is used to fill missing values.

## Prediction outputs

`reports/predictions.csv` includes one row per original model and trade session:

- `signal_date`: session when features are observed after the close.
- `date`: following session when the strategy would enter and exit.
- `model`: configuration identifier.
- `probability_up`: fitted model probability; not a calibrated confidence guarantee.
- `actual_up`: next-session open-to-close label.
- `market_return`: next-session open-to-close fractional return.
- `buy_hold_return`: next adjusted close / signal adjusted close −1.
- `position`: 1 for long, 0 for cash, using the study's fixed trading threshold.
- `net_return`: position × market return − position ×2×cost_per_side/10,000.

Extension outputs are in `experiments/v2/reports`, including `development_selection.json` written before scoring the fresh period. `reports/exploratory_predictions.csv` contains additional historical model comparisons; its file name preserves their research status even though the dashboard uses the display label “Historical backtest.”
