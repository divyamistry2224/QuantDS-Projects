# Model card and research controls

## Intended use

Historical financial signal research and demonstration of model-validation skills. A researcher can inspect whether predictive signals generalize, how implementation costs change returns, and why classifications and trading outcomes differ. No API in this repository submits orders or provides current investment forecasts.

## Target and execution

`target_up = 1` when the next SPY session closes above its open. Zero returns count as not-up. Features are calculated through the signal session's close. A selected signal enters the following open and exits that close. Positions are long or cash, without leverage or shorting. Session rows come from observed market dates, not a generated weekday calendar.

The original trading threshold is 0.55. The enhanced model's classification threshold is 0.58 and its trading threshold is 0.61, selected/fixed under the extension protocol. Every active day pays 0.0002 on entry and 0.0002 on exit. These are assumed combined spread, slippage and fees, not observed execution prices. The strategy is flat every night, including between consecutive active days.

## Original model study

- Training history begins in 2006 after feature warm-up.
- Validation years: 2019, 2020, 2021, equally weighted annual scores.
- Each fold uses only earlier labels and removes five sessions before validation.
- CatBoost: 250 iterations, depth 4, learning rate 0.04, L2 8.
- Random Forest: 300 trees, depth 5, minimum leaf 40.
- Logistic baseline: standardized predictors, C 0.1.
- Market-only, market + macro, market + sentiment and full-feature ablations for CatBoost and Random Forest; full-feature Logistic and a majority baseline.
- Selection: mean validation balanced accuracy; AUC tie-breaker. No holdout-based early stopping.
- Final training ends 2021-12-23, then evaluates 2022-01-03 through 2025-12-31.
- Original winner: Logistic / Full. Its positive 12.15% net cumulative return coexists with weak 48.60% balanced accuracy. Neither statistic should substitute for the other.

## Enhanced study

The [extension protocol](extension_protocol.md), recorded in commit `5f15846`, precedes its model evaluation. New RSI, close-location, reversal, calendar and cross-asset features supplement the original variables. Thirty-two configurations use 2019–2025 development validation and threshold candidates 0.45–0.60 in increments of 0.01. This is 512 model/threshold combinations, so the validation winner is subject to selection optimism.

Only the development winner is scored on January–October 2026. The unchanged original Logistic specification is also refit through pre-2026 data as a reference, without retuning. The selected CatBoost uses enhanced market features and a rolling five-year training window. Macro/sentiment features are not in that winner. The model is not retrained during the new evaluation period.

Fresh balanced accuracy is 53.53%, versus 50% for always-up; raw accuracy is 53.11%, versus 50.72% always-up. Its block-bootstrap interval includes 50%, and its fixed trading overlay returns −0.44% net. These are observed improvements in direction, not a proven trading edge.

## Additional historical comparisons

Extra Trees and histogram gradient boosting were added after observing the original sample. They use fixed specifications and the original validation years. Their 2022–2025 results reuse an observed period and must not be represented as fresh confirmation. The dashboard calls these historical backtests and records this testing history. The profit-focused selection rule ranks mean annual validation net Sharpe, not maximum test return.

## Data timing and revisions

All source data are latest-vintage downloads. No ALFRED vintages, exact historical release calendars or news-publishing timestamps are available in the committed snapshot. The implementation therefore uses explicit availability assumptions and records them as assumptions:

| Source | Assumed available date | Maximum permitted age after assumed availability |
|---|---|---|
| DFF, T10Y2Y, VIXCLS | Observation +2 business days | 14 calendar days |
| UNRATE, CPIAUCSL | Observation month end +45 calendar days | 100 calendar days |
| FRBSF daily news sentiment | Observation +7 calendar days | 21 calendar days |

The business-day offset excludes weekends, but does not use a historical federal/exchange holiday calendar. These lags do not prove historical availability or eliminate revised-data leakage. Adjusted prices are today's provider-adjusted history, not a captured live feed.

## Metrics

- **Balanced accuracy:** mean of sensitivity for up and not-up classes, reducing the effect of class imbalance.
- **Accuracy:** proportion of correct direction classifications; always compare with the up-class rate.
- **AUC:** probability ranking quality across all decision thresholds.
- **Brier and log loss:** quality of raw, uncalibrated probabilities.
- **CAGR:** geometric return with 252 sessions/year.
- **Sharpe:** daily arithmetic mean net return divided by sample standard deviation, annualized by sqrt(252); zero cash/reference rate.
- **Drawdown:** equity relative to its previous peak, including initial capital as a peak.
- **Profit factor:** sum of positive active-day returns divided by the magnitude of negative active-day returns; null if no losses.
- **Exposure:** fraction of sessions active; not a continuous-time exposure estimate.
- **Bootstrap:** circular adjacent blocks, fixed seed 42; not adjusted for all model-selection or data-vintage effects.

The buy-and-hold benchmark uses adjusted close-to-close returns with one entry and one exit cost. It can earn overnight returns and distributions embedded in adjusted prices. Always-long intraday pays a round trip each session, as does an active model signal. This distinction materially affects the comparison.

## Risks, failure modes and deployment gates

Data revisions and assumed publication lags are the largest timing limitations. Other limitations include one survivor ETF, limited fresh observations, structural change, broad sentiment that may be weak at a one-day horizon, uncalibrated probabilities, unrealistic constant costs, no market impact, no capacity estimate, omitted cash yield and taxes, and model selection across many configurations.

Before investment deployment: obtain licensed point-in-time data and release calendars; freeze a forward protocol; collect timestamped forward signals; estimate realistic fills and financing; test alternative universes and regimes; validate probability calibration on separate development data; document controls and perform independent model review. These are future steps, not implemented production capabilities.
