# Research results

Generated from the committed source snapshots; no fabricated performance numbers.

- Selected on validation: **Logistic / Full**.
- Holdout: 2022-01-03 to 2025-12-31, 1,003 sessions.
- Balanced accuracy: **48.60%**; ROC AUC: **0.488**.
- Long/cash net Sharpe: **0.40**; net annual return: **2.92%**; maximum drawdown: **-6.98%**.

## Model comparison

| Model | Features | Validation balanced accuracy | Holdout balanced accuracy | AUC | Net Sharpe |
|---|---|---:|---:|---:|---:|
| Logistic | Full | 51.48% | 48.60% | 0.488 | 0.40 |
| CatBoost | Market | 50.65% | 50.86% | 0.539 | 0.69 |
| CatBoost | Market + sentiment | 50.26% | 49.89% | 0.503 | 0.26 |
| Majority baseline | None | 50.00% | 50.00% | 0.500 | 0.00 |
| CatBoost | Market + macro | 49.91% | 51.40% | 0.536 | 0.78 |
| Random Forest | Market | 49.73% | 51.53% | 0.511 | -0.07 |
| Random Forest | Market + sentiment | 49.45% | 49.35% | 0.510 | -0.13 |
| CatBoost | Full | 48.62% | 51.08% | 0.514 | 0.63 |
| Random Forest | Full | 48.57% | 49.91% | 0.509 | 0.34 |
| Random Forest | Market + macro | 48.48% | 50.97% | 0.529 | 0.77 |

## Interpretation

A weak or negative result is a research finding, not a reason to retune on the holdout. AUC near 0.5 indicates limited directional discrimination. Validation rankings and holdout performance may diverge across regimes. Classification accuracy is not evidence of trading profitability.

The selected strategy's annualized arithmetic excess return versus the always-long intraday baseline is 6.85%; the 95% moving-block bootstrap interval is [-4.49%, 18.92%] (20-session circular blocks, 1,000 draws). This interval is descriptive and does not correct for model selection or revised source data.

## Critical qualifications

- Historical source series are latest-vintage snapshots. Assumed lags reduce obvious date leakage but cannot prove historical tradability.
- SPY is one survivor ETF. Results do not establish cross-asset generalization or stock-selection skill.
- The signal is generated after close and applied only to next-session open-to-close returns. Buy-and-hold also earns overnight returns and has different exposure.
- Transaction costs are assumed (2 bps per side), not measured fills. No capacity, market impact, borrow, cash yield, or taxes are modeled.
- Model probabilities are uncalibrated. The dashboard's threshold/cost scenarios are post-hoc exploration and cannot be treated as new holdout tests.
- No annual model retraining occurs inside the holdout. Economic revisions, calendar assumptions, missingness, and a limited regime sample remain material risks.

This is a retrospective research portfolio project, not a live forecasting or execution service.
