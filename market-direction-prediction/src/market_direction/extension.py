"""Preregistered research extension. Original results remain immutable."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import ROOT, download, fetch
from .evaluation import block_bootstrap_difference, calibration, classification, finance, returns_for
from .features import FEATURE_SETS, MACRO, MARKET, SENTIMENT, build

V2 = ROOT / "experiments/v2"
ASSETS = ["QQQ", "IWM", "TLT", "HYG", "GLD", "UUP"]
EXTRA = ["rsi2", "rsi14", "close_location", "vol_ratio", "downside_vol20", "return_lag2", "return_lag3", "return_lag4", "weekday", "month_end_distance", "sentiment_x_trend"]
CROSS = [f"{asset.lower()}_return5" for asset in ASSETS] + ["smallcap_relative5", "credit_relative5"]
ENHANCED = MARKET + EXTRA[:-1] + CROSS
FULL = ENHANCED + MACRO + SENTIMENT + ["sentiment_x_trend"]


def fetch_extension():
    manifest = fetch(V2, end_date="2026-10-30")
    frames = []
    for asset in ASSETS:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{asset}?period1=1072915200&period2=1793404800&interval=1d"
        raw = download(url)
        data = json.loads(raw)["chart"]["result"][0]
        dates = pd.to_datetime(data["timestamp"], unit="s", utc=True).tz_convert("America/New_York").tz_localize(None).normalize()
        frames.append(pd.DataFrame({"date": dates, "symbol": asset, "adj_close": data["indicators"]["adjclose"][0]["adjclose"]}).dropna())
        manifest["sources"].append({"name": asset, "url": url, "retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "response_sha256": hashlib.sha256(raw).hexdigest()})
    path = V2 / "data/raw/cross_assets.csv"
    pd.concat(frames).to_csv(path, index=False, float_format="%.8f")
    manifest["files"][path.name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": sum(map(len, frames))}
    (V2 / "data/raw/provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def enhanced():
    f = build(V2)
    for n in [2, 14]:
        delta = f.adj_close.diff()
        up = delta.clip(lower=0).ewm(alpha=1/n, adjust=False, min_periods=n).mean()
        down = (-delta.clip(upper=0)).ewm(alpha=1/n, adjust=False, min_periods=n).mean()
        f[f"rsi{n}"] = 100 - 100 / (1 + up / down.replace(0, 1e-12))
    f["close_location"] = (f.close - f.low) / (f.high - f.low).replace(0, np.nan)
    f["vol_ratio"] = f.return_1d.rolling(5).std() / f.return_1d.rolling(20).std()
    f["downside_vol20"] = f.return_1d.clip(upper=0).pow(2).rolling(20).mean().pow(.5)
    for lag in [2, 3, 4]: f[f"return_lag{lag}"] = f.return_1d.shift(lag - 1)
    f["weekday"] = f.signal_date.dt.dayofweek
    f["month_end_distance"] = (f.signal_date + pd.offsets.MonthEnd(0) - f.signal_date).dt.days
    f["sentiment_x_trend"] = f.news_sentiment * f.distance_sma200
    cross = pd.read_csv(V2 / "data/raw/cross_assets.csv", parse_dates=["date"])
    for asset in ASSETS:
        series = cross.loc[cross.symbol == asset].set_index("date").adj_close.sort_index()
        f[f"{asset.lower()}_return5"] = f.signal_date.map(series.pct_change(5))
    f["smallcap_relative5"] = f.iwm_return5 - f.return_5d
    f["credit_relative5"] = f.hyg_return5 - f.tlt_return5
    return f.replace([np.inf, -np.inf], np.nan).dropna(subset=FULL).reset_index(drop=True)


def candidates():
    items = []
    for group, columns in [("Enhanced market", ENHANCED), ("Full", FULL)]:
        for depth in [2, 3, 4]:
            for window in [5, 0]: items.append({"family": "CatBoost", "features": group, "columns": columns, "depth": depth, "window": window})
        for depth in [3, 6]:
            for window in [5, 10]: items.append({"family": "Random Forest", "features": group, "columns": columns, "depth": depth, "window": window})
        for c in [.01, .1, 1.]:
            for window in [5, 0]: items.append({"family": "Logistic", "features": group, "columns": columns, "C": c, "window": window})
    for i, spec in enumerate(items): spec["id"] = f"extension_{i:02d}"
    return items


def model_for(spec):
    if spec["family"] == "CatBoost": return CatBoostClassifier(iterations=180, depth=spec["depth"], learning_rate=.04, l2_leaf_reg=15, random_seed=42, thread_count=2, verbose=False, allow_writing_files=False)
    if spec["family"] == "Random Forest": return RandomForestClassifier(n_estimators=200, max_depth=spec["depth"], min_samples_leaf=60, max_features="sqrt", random_state=42, n_jobs=2)
    return make_pipeline(StandardScaler(), LogisticRegression(C=spec["C"], max_iter=1500, random_state=42))


def train_split(f, year, window):
    validation = f.loc[f.trade_date.dt.year == year]
    train = f.loc[f.trade_date < validation.trade_date.min()]
    if window: train = train.loc[train.trade_date >= validation.trade_date.min() - pd.DateOffset(years=window)]
    return train.iloc[:-5], validation


def direction_interval(y, probability, threshold):
    rng = np.random.default_rng(42)
    y = np.asarray(y); predicted = np.asarray(probability) >= threshold
    values = []
    for _ in range(1000):
        starts = rng.integers(0, len(y), size=int(np.ceil(len(y) / 10)))
        indices = ((starts[:, None] + np.arange(10)) % len(y)).ravel()[:len(y)]
        if len(np.unique(y[indices])) == 2: values.append(balanced_accuracy_score(y[indices], predicted[indices]))
    return {"lower_95": float(np.quantile(values, .025)), "upper_95": float(np.quantile(values, .975)), "block_sessions": 10, "repetitions": 1000}


def run_extension():
    f = enhanced()
    thresholds = np.round(np.arange(.45, .601, .01), 2)
    search = []
    for i, spec in enumerate(candidates()):
        annual, probs = [], []
        for year in range(2019, 2026):
            train, validation = train_split(f, year, spec["window"])
            model = model_for(spec)
            model.fit(train[spec["columns"]], train.target_up)
            p = model.predict_proba(validation[spec["columns"]])[:, 1]
            scores = [balanced_accuracy_score(validation.target_up, p >= threshold) for threshold in thresholds]
            annual.append(scores)
            probs.append(roc_auc_score(validation.target_up, p))
        average = np.mean(annual, axis=0)
        best = int(np.argmax(average))
        search.append({**spec, "threshold": float(thresholds[best]), "cv_balanced_accuracy": float(average[best]), "cv_auc": float(np.mean(probs)), "annual_balanced_accuracy": {str(year): float(row[best]) for year, row in zip(range(2019, 2026), annual)}})
        print(f"Extension {i + 1}/32 · {spec['family']} · validation BA {average[best]:.4f}", flush=True)
    selected = max(search, key=lambda s: (s["cv_balanced_accuracy"], s["cv_auc"]))
    # Persist the winner BEFORE reading or scoring fresh test labels.
    out = V2 / "reports";out.mkdir(parents=True, exist_ok=True)
    (out / "development_selection.json").write_text(json.dumps({"selected": selected, "candidates": search}, indent=2) + "\n")
    train, test = train_split(f, 2026, selected["window"])
    model = model_for(selected);model.fit(train[selected["columns"]], train.target_up)
    p = model.predict_proba(test[selected["columns"]])[:, 1]
    threshold = selected["threshold"]
    position, net = returns_for(p, test.target_return, threshold + .03, 2.)
    y = test.target_up.to_numpy()
    baseline_prob = np.repeat(train.target_up.mean(), len(test))
    # Frozen original specification; only refit on pre-2026 development labels.
    original = build(V2)
    original_train, original_test = train_split(original, 2026, 0)
    baseline_model = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=1000, random_state=42))
    baseline_model.fit(original_train[FEATURE_SETS["Full"]], original_train.target_up)
    baseline_predictions = pd.Series(baseline_model.predict_proba(original_test[FEATURE_SETS["Full"]])[:, 1], index=original_test.trade_date)
    reference = baseline_predictions.reindex(test.trade_date).to_numpy()
    reference_position, reference_net = returns_for(reference, test.target_return, .55, 2.)
    values = model.feature_importances_ if selected["family"] != "Logistic" else abs(model[-1].coef_[0])
    values = np.asarray(values) / np.sum(values)
    fresh = {"protocol": "docs/research/extension_protocol.md", "protocol_commit": "5f15846", "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "generated_at_utc": datetime.now(timezone.utc).isoformat(), "selected": selected, "candidate_count": len(search), "threshold_candidates": thresholds.tolist(), "development": {"validation_years": list(range(2019, 2026)), "training_rows": len(train), "train_start": str(train.trade_date.min().date()), "train_end": str(train.trade_date.max().date())}, "holdout": {"start": str(test.trade_date.min().date()), "end": str(test.trade_date.max().date()), "rows": len(test), "up_rate": float(y.mean())}, "classification": classification(y, p, threshold), "direction_interval": direction_interval(y, p, threshold), "strategy": finance(net, position), "trade_threshold": round(threshold + .03, 2), "cost_per_side_bps": 2., "original_specification": {"classification": classification(y, reference), "strategy": finance(reference_net, reference_position)}, "majority": classification(y, baseline_prob), "calibration": calibration(y, p), "bootstrap": block_bootstrap_difference(net, test.target_return.to_numpy() - .0004), "search": search, "feature_importance": [{"feature": feature, "importance": float(value)} for feature, value in zip(selected["columns"], values)], "data": json.loads((V2 / "data/raw/provenance.json").read_text()), "caveat": "2022–2025 was reused as development after the original result. Only the selected extension was scored on 2026. Short fresh holdout, many development comparisons and latest-vintage data limit inference."}
    predictions = pd.DataFrame({"date": test.trade_date.dt.strftime("%Y-%m-%d"), "signal_date": test.signal_date.dt.strftime("%Y-%m-%d"), "probability_up": p, "actual_up": y, "market_return": test.target_return, "buy_hold_return": test.buy_hold_return, "position": position, "net_return": net, "original_probability_up": reference})
    (out / "metrics.json").write_text(json.dumps(fresh, indent=2, allow_nan=False) + "\n")
    predictions.to_csv(out / "predictions.csv", index=False, float_format="%.9f")
    (ROOT / "docs/data/extension.json").write_text(json.dumps({"report": fresh, "observations": predictions.round(9).to_dict(orient="records")}, separators=(",", ":"), allow_nan=False) + "\n")
    predictions.to_csv(ROOT / "docs/data/extension_predictions.csv", index=False, float_format="%.9f")
    print(json.dumps({"selected": {k:v for k,v in selected.items() if k != "columns"}, "fresh_holdout": fresh["classification"], "majority": fresh["majority"], "strategy": fresh["strategy"], "interval": fresh["direction_interval"]}, indent=2), flush=True)
    return fresh
