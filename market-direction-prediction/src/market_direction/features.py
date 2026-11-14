"""Causal transforms and conservative assumed-availability joins.

No target or future row is used in any predictor. Latest-vintage source data can
still contain retrospective revisions: lags alone do not remove vintage bias.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import ROOT, verify

MARKET = ["return_1d", "return_5d", "return_20d", "volatility_20d", "distance_sma50", "distance_sma200", "intraday_return", "overnight_return", "high_low_range", "volume_z20"]
MACRO = ["fed_funds", "yield_curve", "vix", "unemployment", "inflation_yoy", "fed_funds_change20", "yield_curve_change20", "vix_change5", "unemployment_change60"]
SENTIMENT = ["news_sentiment", "sentiment_mean5", "sentiment_mean20", "sentiment_change5", "sentiment_z60"]
FEATURE_SETS = {"Market": MARKET, "Market + macro": MARKET + MACRO, "Market + sentiment": MARKET + SENTIMENT, "Full": MARKET + MACRO + SENTIMENT}


def available_join(left: pd.DataFrame, right: pd.DataFrame, value: str, max_age: str) -> pd.DataFrame:
    """Backward as-of join; never borrow a future observation or backfill."""
    right = right[["available_date", value]].dropna().sort_values("available_date")
    if right.available_date.duplicated().any():
        # Calendar dates mapped to business days may collide; retain latest observation.
        right = right.drop_duplicates("available_date", keep="last")
    key = f"{value}_available_date"
    right = right.rename(columns={"available_date": key})
    return pd.merge_asof(left.sort_values("signal_date"), right, left_on="signal_date", right_on=key, direction="backward", tolerance=pd.Timedelta(max_age))


def build(root: Path = ROOT) -> pd.DataFrame:
    verify(root)
    raw = root / "data/raw"
    frame = pd.read_csv(raw / "prices.csv", parse_dates=["date"]).rename(columns={"date": "signal_date"})
    adj = frame.adj_close
    returns = adj.pct_change()
    frame["return_1d"] = returns
    for n in [5, 20]: frame[f"return_{n}d"] = adj.pct_change(n)
    frame["volatility_20d"] = returns.rolling(20).std() * np.sqrt(252)
    for n in [50, 200]: frame[f"distance_sma{n}"] = adj / adj.rolling(n).mean() - 1
    frame["intraday_return"] = frame.close / frame.open - 1
    frame["overnight_return"] = frame.open / frame.close.shift() - 1
    frame["high_low_range"] = frame.high / frame.low - 1
    volume = np.log1p(frame.volume)
    frame["volume_z20"] = (volume - volume.rolling(20).mean()) / volume.rolling(20).std()
    frame["trade_date"] = frame.signal_date.shift(-1)
    frame["target_return"] = frame.intraday_return.shift(-1)
    frame["buy_hold_return"] = returns.shift(-1)
    frame["target_up"] = (frame.target_return > 0).astype(int)

    macro = pd.read_csv(raw / "macro.csv", parse_dates=["observation_date"])
    names = {"DFF": "fed_funds", "T10Y2Y": "yield_curve", "VIXCLS": "vix", "UNRATE": "unemployment", "CPIAUCSL": "inflation_yoy"}
    for series, name in names.items():
        obs = macro.loc[macro.series == series].sort_values("observation_date").copy()
        obs[name] = obs.value
        if series in ("UNRATE", "CPIAUCSL"):
            if series == "CPIAUCSL": obs[name] = obs.value.pct_change(12) * 100
            obs["available_date"] = obs.observation_date + pd.offsets.MonthEnd(0) + pd.Timedelta(days=45)
            age = "100D"
        else:
            obs["available_date"] = obs.observation_date + pd.offsets.BDay(2)
            age = "14D"
        frame = available_join(frame, obs, name, age)
    frame["fed_funds_change20"] = frame.fed_funds.diff(20)
    frame["yield_curve_change20"] = frame.yield_curve.diff(20)
    frame["vix_change5"] = frame.vix.diff(5)
    frame["unemployment_change60"] = frame.unemployment.diff(60)

    sentiment = pd.read_csv(raw / "sentiment.csv", parse_dates=["observation_date"])
    sentiment["available_date"] = sentiment.observation_date + pd.Timedelta(days=7)
    frame = available_join(frame, sentiment, "news_sentiment", "21D")
    for n in [5, 20]: frame[f"sentiment_mean{n}"] = frame.news_sentiment.rolling(n).mean()
    frame["sentiment_change5"] = frame.news_sentiment.diff(5)
    frame["sentiment_z60"] = (frame.news_sentiment - frame.news_sentiment.rolling(60).mean()) / frame.news_sentiment.rolling(60).std()
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=FEATURE_SETS["Full"] + ["target_return", "buy_hold_return", "trade_date"])
    frame = frame.loc[frame.signal_date >= "2006-01-01"].reset_index(drop=True)
    return frame
