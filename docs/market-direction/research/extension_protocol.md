# Research extension protocol

The initial 2022–2025 evaluation has been observed and is preserved. It cannot
remain an untouched test set for further tuning. The extension uses 2019–2025
as development validation and reserves **2026-01-01 through 2026-10-30** for a
single final evaluation. This protocol is committed before downloading that
new sample or running the extension search.

The target remains next-session SPY open-to-close direction. New predictors
add short-term reversal, RSI, close location, volatility ratios, calendar
effects and cross-asset context (QQQ, IWM, TLT, HYG, GLD, UUP). All market
features use information through signal-date close. The original assumed
macro and sentiment publication lags are retained. All data are latest
vintage, so this is still not a point-in-time trading certification.

Candidates are fixed before the new holdout is inspected:

- CatBoost: 180 iterations, depth 2/3/4, learning rate 0.04, L2 15;
  enhanced-market or full features; trailing 5 years or expanding history.
- Random Forest: 200 trees, depth 3/6, leaf size 60;
  enhanced-market or full features; trailing 5 or 10 years.
- Logistic regression: standardized inputs, C 0.01/0.1/1;
  enhanced-market or full features; trailing 5 years or expanding history.
- Seed 42. No fresh-holdout early stopping, tuning, or ensemble reweighting.

For each candidate, fit on earlier data and score each annual validation
period 2019–2025, with a five-session purge. Select one model and one
classification threshold from 0.45 to 0.60 inclusive in 0.01 increments by
highest mean annual balanced accuracy (ROC AUC tie-breaker). These many
comparisons can inflate validation performance; the new holdout is the check.

Refit the selected configuration using development labels through 2025,
purging five sessions before the holdout. Freeze the model throughout 2026.
Evaluate only that selected configuration on the new holdout. Compare to
always-up and the original fixed Logistic/Full configuration (no retuning).

The trading overlay uses the chosen classification threshold plus a fixed
0.03 abstention margin. Long next-session open-to-close or cash; 2 bps each
side; zero cash yield. Neither the threshold nor the cost assumption will be
changed after examining the fresh holdout. Classification improvement and
net profitability are separate questions. Bootstrap intervals are descriptive,
not proof of alpha. If the extension fails, report the failure.
