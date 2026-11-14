"""Classification diagnostics and fully specified next-session intraday execution."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, log_loss, roc_auc_score


def classification(y, probability, threshold=.5) -> dict:
    p = np.clip(np.asarray(probability), 1e-6, 1 - 1e-6)
    return {"accuracy": float(accuracy_score(y, p >= threshold)), "balanced_accuracy": float(balanced_accuracy_score(y, p >= threshold)), "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None, "brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p, labels=[0, 1]))}


def returns_for(probability, market_return, threshold=.55, cost_bps=2.0) -> tuple[np.ndarray, np.ndarray]:
    """Long SPY next open to close if p >= threshold, otherwise cash.

    Every active day is a round trip. Entry and exit EACH cost cost_bps,
    even when there are two consecutive active days. No overnight position.
    Cash yield, commissions and taxes excluded; bps represent assumed total
    one-way spread/slippage/fees. No leverage, shorting or volatility targeting.
    """
    position = (np.asarray(probability) >= threshold).astype(float)
    net = position * np.asarray(market_return) - position * (2 * cost_bps / 10_000)
    return position, net


def finance(net, position=None) -> dict:
    r = np.asarray(net, dtype=float)
    equity = np.r_[1., np.cumprod(1 + r)]
    dd = equity / np.maximum.accumulate(equity) - 1
    sd = np.std(r, ddof=1) if len(r) > 1 else 0
    active = np.asarray(position, dtype=bool) if position is not None else np.ones(len(r), dtype=bool)
    trades = r[active]
    losses = -trades[trades < 0].sum()
    return {"total_return": float(equity[-1] - 1), "annual_return": float(equity[-1] ** (252 / len(r)) - 1) if len(r) else 0., "annual_volatility": float(sd * np.sqrt(252)), "sharpe": float(np.mean(r) / sd * np.sqrt(252)) if sd > 0 else 0., "max_drawdown": float(dd.min()), "exposure": float(active.mean()) if len(r) else 0., "round_trips": int(active.sum()), "hit_rate": float((trades > 0).mean()) if len(trades) else None, "profit_factor": float(trades[trades > 0].sum() / losses) if losses else None}


def block_bootstrap_difference(strategy, benchmark, seed=42, block=20, repetitions=1000) -> dict:
    """Percentile interval for annualized arithmetic mean difference; descriptive,
    not multiplicity-adjusted evidence of alpha. Resamples adjacent sessions.
    """
    difference = np.asarray(strategy) - np.asarray(benchmark)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repetitions):
        starts = rng.integers(0, len(difference), size=int(np.ceil(len(difference) / block)))
        indices = ((starts[:, None] + np.arange(block)) % len(difference)).ravel()[:len(difference)]
        estimates.append(float(difference[indices].mean() * 252))
    return {"annual_mean_excess": float(difference.mean() * 252), "lower_95": float(np.quantile(estimates, .025)), "upper_95": float(np.quantile(estimates, .975)), "block_sessions": block, "repetitions": repetitions}


def calibration(y, p, bins=10) -> list[dict]:
    frame = pd.DataFrame({"y": y, "p": p})
    frame["bin"] = pd.cut(frame.p, np.linspace(0, 1, bins + 1), include_lowest=True)
    return [{"predicted": float(x.p.mean()), "observed": float(x.y.mean()), "count": len(x)} for _, x in frame.groupby("bin", observed=True) if len(x)]
