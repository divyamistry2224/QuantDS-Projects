"""Predeclared expanding-window comparison and one untouched holdout."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import catboost
import numpy as np
import pandas as pd
import sklearn
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import ROOT
from .evaluation import block_bootstrap_difference, calibration, classification, finance, returns_for
from .features import FEATURE_SETS, build

THRESHOLD = .55
COST_BPS = 2.0
GAP = 5
VALIDATION_YEARS = [2019, 2020, 2021]
TEST_START = "2022-01-01"
TEST_END = "2025-12-31"


def split_for_year(frame: pd.DataFrame, year: int, gap=GAP):
    validation = frame.loc[frame.trade_date.dt.year == year]
    # Cut on label realization date, not just feature date; purge five complete
    # sessions before the first validation label. Holdout uses the same rule.
    training = frame.loc[frame.trade_date < validation.trade_date.min()].iloc[:-gap] if gap else frame.loc[frame.trade_date < validation.trade_date.min()]
    if training.empty or validation.empty: raise ValueError(f"Empty split for {year}")
    return training, validation


def estimator(family: str):
    if family == "CatBoost":
        return CatBoostClassifier(iterations=250, depth=4, learning_rate=.04, l2_leaf_reg=8, loss_function="Logloss", random_seed=42, thread_count=2, verbose=False, allow_writing_files=False)
    if family == "Random Forest":
        return RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=40, max_features="sqrt", random_state=42, n_jobs=2)
    if family == "Logistic": return make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=1000, random_state=42))
    raise ValueError(family)


def specs():
    return [{"id": f"{family.lower().replace(' ', '_')}__{name.lower().replace(' + ', '_').replace(' ', '_')}", "family": family, "features": name, "columns": columns} for family in ["CatBoost", "Random Forest"] for name, columns in FEATURE_SETS.items()] + [{"id": "logistic__full", "family": "Logistic", "features": "Full", "columns": FEATURE_SETS["Full"]}]


def run(root: Path = ROOT) -> dict:
    frame = build(root)
    out = root / "reports"
    out.mkdir(exist_ok=True)
    web = root / "docs/data"
    web.mkdir(parents=True, exist_ok=True)
    fold_rows, summaries, all_predictions, importance = [], [], [], []
    train, _ = split_for_year(frame, 2022)
    test = frame.loc[frame.trade_date.between(TEST_START, TEST_END)].copy()
    test = test.reset_index(drop=True)
    if len(test) < 800: raise ValueError("Holdout unexpectedly short")
    for spec in specs():
        print(f"Evaluating {spec['family']} / {spec['features']}", flush=True)
        scores = []
        for year in VALIDATION_YEARS:
            tr, va = split_for_year(frame, year)
            model = estimator(spec["family"])
            model.fit(tr[spec["columns"]], tr.target_up)
            probability = model.predict_proba(va[spec["columns"]])[:, 1]
            score = classification(va.target_up, probability)
            position, net = returns_for(probability, va.target_return, THRESHOLD, COST_BPS)
            fold_rows.append({"id": spec["id"], "family": spec["family"], "features": spec["features"], "year": year, "train_start": str(tr.trade_date.min().date()), "train_end": str(tr.trade_date.max().date()), "validation_start": str(va.trade_date.min().date()), "validation_end": str(va.trade_date.max().date()), "training_rows": len(tr), "validation_rows": len(va), **score, **{f"finance_{k}": v for k, v in finance(net, position).items()}})
            scores.append(score)
        model = estimator(spec["family"])
        model.fit(train[spec["columns"]], train.target_up)
        probability = model.predict_proba(test[spec["columns"]])[:, 1]
        position, net = returns_for(probability, test.target_return, THRESHOLD, COST_BPS)
        values = model.feature_importances_ if spec["family"] != "Logistic" else abs(model[-1].coef_[0])
        values = np.asarray(values) / np.sum(values)
        importance.extend({"model": spec["id"], "feature": feature, "importance": float(value)} for feature, value in zip(spec["columns"], values))
        summaries.append({"id": spec["id"], "family": spec["family"], "features": spec["features"], "feature_count": len(spec["columns"]), "cv_balanced_accuracy": float(np.mean([s["balanced_accuracy"] for s in scores])), "cv_balanced_accuracy_std": float(np.std([s["balanced_accuracy"] for s in scores])), "cv_roc_auc": float(np.mean([s["roc_auc"] for s in scores])), "holdout": classification(test.target_up, probability), "strategy": finance(net, position), "calibration": calibration(test.target_up.to_numpy(), probability)})
        all_predictions.append(pd.DataFrame({"date": test.trade_date.dt.strftime("%Y-%m-%d"), "signal_date": test.signal_date.dt.strftime("%Y-%m-%d"), "model": spec["id"], "probability_up": probability, "actual_up": test.target_up, "market_return": test.target_return, "buy_hold_return": test.buy_hold_return, "position": position, "net_return": net}))

    # Baseline probabilities use only the development training sample.
    for name, p in [("majority_baseline", np.repeat(train.target_up.mean(), len(test)))]:
        summaries.append({"id": name, "family": "Majority baseline", "features": "None", "feature_count": 0, "cv_balanced_accuracy": .5, "cv_balanced_accuracy_std": 0., "cv_roc_auc": .5, "holdout": classification(test.target_up, p), "strategy": finance(*reversed(returns_for(p, test.target_return, THRESHOLD, COST_BPS))), "calibration": calibration(test.target_up.to_numpy(), p)})
        all_predictions.append(pd.DataFrame({"date": test.trade_date.dt.strftime("%Y-%m-%d"), "signal_date": test.signal_date.dt.strftime("%Y-%m-%d"), "model": name, "probability_up": p, "actual_up": test.target_up, "market_return": test.target_return, "buy_hold_return": test.buy_hold_return, "position": (p >= THRESHOLD).astype(int), "net_return": returns_for(p, test.target_return, THRESHOLD, COST_BPS)[1]}))

    # Equal weight per validation year. No test metric participates in selection.
    selected = max(summaries, key=lambda x: (x["cv_balanced_accuracy"], x["cv_roc_auc"]))
    predictions = pd.concat(all_predictions, ignore_index=True)
    chosen = predictions.loc[predictions.model == selected["id"]]
    always_intraday = test.target_return.to_numpy() - 2 * COST_BPS / 10_000
    buyhold = test.buy_hold_return.to_numpy().copy()
    buyhold[0] -= COST_BPS / 10_000
    buyhold[-1] -= COST_BPS / 10_000
    provenance = json.loads((root / "data/raw/provenance.json").read_text())
    try: revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True).strip()
    except subprocess.CalledProcessError: revision = "initial-local-run"
    report = {"title": "Market Direction Research", "generated_at_utc": datetime.now(timezone.utc).isoformat(), "code_revision": revision, "universe": "SPY — S&P 500 ETF", "target": "Next trading session close / open − 1 > 0", "decision_time": "After signal-date close; execute at next session open and exit that session close", "training": {"start": str(train.trade_date.min().date()), "end": str(train.trade_date.max().date()), "rows": len(train)}, "validation_years": VALIDATION_YEARS, "holdout": {"start": str(test.trade_date.min().date()), "end": str(test.trade_date.max().date()), "rows": len(test), "up_rate": float(test.target_up.mean())}, "protocol": {"selection": "Highest mean validation balanced accuracy; ROC AUC tie-breaker. Equal weight per year. Fixed hyperparameters. No holdout tuning.", "gap_sessions": GAP, "threshold": THRESHOLD, "cost_per_side_bps": COST_BPS, "seed": 42, "risk_free_rate": 0., "annual_sessions": 252, "cash_yield": 0., "vintage_warning": provenance["vintage"]}, "selected_model": selected["id"], "models": summaries, "folds": fold_rows, "feature_importance": importance, "benchmarks": {"always_long_intraday": finance(always_intraday), "buy_and_hold_close_to_close": {**finance(buyhold), "round_trips": 1}}, "bootstrap": block_bootstrap_difference(chosen.net_return, always_intraday), "data": provenance, "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__, "catboost": catboost.__version__}, "coverage": {"feature_rows": len(frame), "features": len(FEATURE_SETS["Full"]), "first_signal": str(frame.signal_date.min().date()), "last_signal": str(frame.signal_date.max().date())}}
    (out / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    predictions.to_csv(out / "predictions.csv", index=False, float_format="%.9f")
    pd.DataFrame(fold_rows).to_csv(out / "validation_folds.csv", index=False)
    # Compact browser format: one shared observation table plus probability arrays.
    observations = [{"date": str(r.trade_date.date()), "signal_date": str(r.signal_date.date()), "market_return": round(r.target_return, 9), "buy_hold_return": round(r.buy_hold_return, 9), "actual_up": int(r.target_up), "vix": round(r.vix, 3), "yield_curve": round(r.yield_curve, 3), "fed_funds": round(r.fed_funds, 3), "inflation_yoy": round(r.inflation_yoy, 3), "unemployment": round(r.unemployment, 3), "news_sentiment": round(r.news_sentiment, 5)} for r in test.itertuples()]
    bundle = {"report": report, "observations": observations, "probabilities": {m["id"]: predictions.loc[predictions.model == m["id"], "probability_up"].round(7).tolist() for m in summaries}}
    (web / "research.json").write_text(json.dumps(bundle, separators=(",", ":"), allow_nan=False) + "\n")
    predictions.to_csv(web / "predictions.csv", index=False, float_format="%.9f")
    frame[["signal_date", "trade_date", *FEATURE_SETS["Full"], "target_up", "target_return"]].to_csv(out / "features.csv", index=False, float_format="%.9f")
    write_report(report, out)
    make_figures(report, predictions, out)
    print(f"Selected on validation: {selected['id']}; holdout balanced accuracy: {selected['holdout']['balanced_accuracy']:.4f}", flush=True)
    return report


def write_report(r: dict, out: Path):
    chosen = next(m for m in r["models"] if m["id"] == r["selected_model"])
    lines = ["# Research results", "", "Generated from the committed source snapshots; no fabricated performance numbers.", "", f"- Selected on validation: **{chosen['family']} / {chosen['features']}**.", f"- Holdout: {r['holdout']['start']} to {r['holdout']['end']}, {r['holdout']['rows']:,} sessions.", f"- Balanced accuracy: **{chosen['holdout']['balanced_accuracy']:.2%}**; ROC AUC: **{chosen['holdout']['roc_auc']:.3f}**.", f"- Long/cash net Sharpe: **{chosen['strategy']['sharpe']:.2f}**; net annual return: **{chosen['strategy']['annual_return']:.2%}**; maximum drawdown: **{chosen['strategy']['max_drawdown']:.2%}**.", "", "## Model comparison", "", "| Model | Features | Validation balanced accuracy | Holdout balanced accuracy | AUC | Net Sharpe |", "|---|---|---:|---:|---:|---:|"]
    for m in sorted(r["models"], key=lambda x: x["cv_balanced_accuracy"], reverse=True):
        lines.append(f"| {m['family']} | {m['features']} | {m['cv_balanced_accuracy']:.2%} | {m['holdout']['balanced_accuracy']:.2%} | {m['holdout']['roc_auc']:.3f} | {m['strategy']['sharpe']:.2f} |")
    b = r["bootstrap"]
    lines += ["", "## Interpretation", "", "A weak or negative result is a research finding, not a reason to retune on the holdout. AUC near 0.5 indicates limited directional discrimination. Validation rankings and holdout performance may diverge across regimes. Classification accuracy is not evidence of trading profitability.", "", f"The selected strategy's annualized arithmetic excess return versus the always-long intraday baseline is {b['annual_mean_excess']:.2%}; the 95% moving-block bootstrap interval is [{b['lower_95']:.2%}, {b['upper_95']:.2%}] (20-session circular blocks, 1,000 draws). This interval is descriptive and does not correct for model selection or revised source data.", "", "## Critical qualifications", "", "- Historical source series are latest-vintage snapshots. Assumed lags reduce obvious date leakage but cannot prove historical tradability.", "- SPY is one survivor ETF. Results do not establish cross-asset generalization or stock-selection skill.", "- The signal is generated after close and applied only to next-session open-to-close returns. Buy-and-hold also earns overnight returns and has different exposure.", "- Transaction costs are assumed (2 bps per side), not measured fills. No capacity, market impact, borrow, cash yield, or taxes are modeled.", "- Model probabilities are uncalibrated. The dashboard's threshold/cost scenarios are post-hoc exploration and cannot be treated as new holdout tests.", "- No annual model retraining occurs inside the holdout. Economic revisions, calendar assumptions, missingness, and a limited regime sample remain material risks.", "", "This is a retrospective research portfolio project, not a live forecasting or execution service."]
    text = "\n".join(lines) + "\n"
    (out / "research_report.md").write_text(text)
    (out.parent / "docs/research/research_report.md").write_text(text)


def make_figures(report: dict, predictions: pd.DataFrame, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = out / "figures"
    directory.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    chosen = predictions.loc[predictions.model == report["selected_model"]]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    dates = pd.to_datetime(chosen.date)
    ax.plot(dates, 100 * (1 + chosen.net_return).cumprod(), color="#17695e", label="Selected long/cash · net")
    ax.plot(dates, 100 * (1 + chosen.market_return - 2 * COST_BPS / 10_000).cumprod(), color="#b98133", label="Always-long intraday · net")
    buyhold = chosen.buy_hold_return.to_numpy().copy()
    buyhold[0] -= COST_BPS / 10_000; buyhold[-1] -= COST_BPS / 10_000
    ax.plot(dates, 100 * np.cumprod(1 + buyhold), color="#657b92", label="Buy & hold · different overnight exposure")
    ax.set(ylabel="Growth of $100", title="Untouched holdout · next-session SPY research")
    ax.legend(frameon=False, fontsize=8); ax.grid(alpha=.15)
    fig.tight_layout();fig.savefig(directory / "holdout_equity.png", dpi=180);plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    models = [m for m in report["models"] if m["family"] != "Majority baseline"]
    x = np.arange(len(models)); width = .36
    ax.bar(x - width / 2, [m["cv_balanced_accuracy"] for m in models], width, color="#17695e", label="Validation mean")
    ax.bar(x + width / 2, [m["holdout"]["balanced_accuracy"] for m in models], width, color="#88a1b8", label="Holdout")
    ax.axhline(.5, color="#a35a45", linestyle="--", linewidth=1)
    ax.set_xticks(x, [m["family"] + "\n" + m["features"] for m in models], fontsize=7)
    ax.set(ylim=(.44, .59), ylabel="Balanced accuracy", title="Ablations · identical chronological splits")
    ax.legend(frameon=False);fig.tight_layout();fig.savefig(directory / "model_comparison.png", dpi=180);plt.close(fig)
