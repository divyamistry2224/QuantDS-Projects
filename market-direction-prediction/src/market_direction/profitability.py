"""Exploratory profit-focused comparison after the original test was observed.

This module never calls a reused sample an untouched or confirmatory test.
"""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier

from .data import ROOT
from .evaluation import classification, finance, returns_for
from .experiment import split_for_year
from .features import FEATURE_SETS, build


def run_profitability():
    f = build()
    original = json.loads((ROOT / "reports/metrics.json").read_text())
    models, rows = [], []
    train, _ = split_for_year(f, 2022)
    test = f.loc[f.trade_date.between("2022-01-01", "2025-12-31")].reset_index(drop=True)
    for family in ["Extra Trees", "Histogram GB"]:
        for name in ["Market", "Full"]:
            columns = FEATURE_SETS[name]
            def make():
                if family == "Extra Trees": return ExtraTreesClassifier(n_estimators=300, max_depth=6, min_samples_leaf=40, random_state=42, n_jobs=2)
                return HistGradientBoostingClassifier(max_iter=180, max_leaf_nodes=7, min_samples_leaf=60, learning_rate=.04, l2_regularization=15, early_stopping=False, random_state=42)
            identifier = f"{family.lower().replace(' ', '_')}__{name.lower()}"
            annual = []
            for year in [2019, 2020, 2021]:
                tr, va = split_for_year(f, year)
                model = make();model.fit(tr[columns], tr.target_up)
                probability = model.predict_proba(va[columns])[:, 1]
                position, net = returns_for(probability, va.target_return, .55, 2.)
                annual.append({"year": year, **finance(net, position), **classification(va.target_up, probability)})
            model = make();model.fit(train[columns], train.target_up)
            probability = model.predict_proba(test[columns])[:, 1]
            position, net = returns_for(probability, test.target_return, .55, 2.)
            models.append({"id": identifier, "family": family, "features": name, "cv_net_sharpe": float(np.mean([a["sharpe"] for a in annual])), "cv_balanced_accuracy": float(np.mean([a["balanced_accuracy"] for a in annual])), "holdout": classification(test.target_up, probability), "strategy": finance(net, position), "annual_validation": annual, "status": "Exploratory reused 2022–2025 sample; not fresh confirmation"})
            rows.append(pd.DataFrame({"date": test.trade_date.dt.strftime("%Y-%m-%d"), "model": identifier, "probability_up": probability, "net_return": net, "position": position}))
    for m in original["models"]:
        fold = [r for r in original["folds"] if r["id"] == m["id"]]
        models.append({"id": m["id"], "family": m["family"], "features": m["features"], "cv_net_sharpe": float(np.mean([r["finance_sharpe"] for r in fold])) if fold else 0., "cv_balanced_accuracy": m["cv_balanced_accuracy"], "holdout": m["holdout"], "strategy": m["strategy"], "status": "Original recorded result"})
    selected = max(models, key=lambda m: m["cv_net_sharpe"])
    report = {"objective": "Mean annual validation net Sharpe, with fixed 0.55 trade threshold and 2 bps each side", "selected_on_validation": selected["id"], "models": models, "warning": "Profit-focused analysis was added after observing the original holdout. New model results on 2022–2025 are exploratory. Positive historical profit does not prove future profitability or benchmark outperformance.", "benchmarks": original["benchmarks"], "period": original["holdout"]}
    (ROOT / "reports/profitability.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    predictions = pd.concat(rows, ignore_index=True)
    predictions.to_csv(ROOT / "reports/exploratory_predictions.csv", index=False, float_format="%.9f")
    (ROOT / "docs/data/profitability.json").write_text(json.dumps({"report": report, "probabilities": {m["id"]: predictions.loc[predictions.model == m["id"], "probability_up"].round(7).tolist() for m in models if m["id"] in predictions.model.unique()}}, separators=(",", ":"), allow_nan=False) + "\n")
    print(json.dumps({"selected": selected, "new_models": models[:4]}, indent=2), flush=True)
    return report
