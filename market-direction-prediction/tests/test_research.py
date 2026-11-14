"""Research integrity tests: timing, accounting and snapshot reproducibility."""
import hashlib
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from market_direction.data import ROOT, validate_prices, verify
from market_direction.evaluation import block_bootstrap_difference, finance, returns_for
from market_direction.experiment import split_for_year
from market_direction.features import FEATURE_SETS, available_join, build


@pytest.fixture(scope="module")
def features(): return build()


def test_snapshot_hashes_and_ohlc():
    manifest = verify()
    assert manifest["files"]["prices.csv"]["rows"] > 5000


def test_checksum_rejects_modified_input(tmp_path):
    shutil.copytree(ROOT / "data", tmp_path / "data")
    with (tmp_path / "data/raw/prices.csv").open("a") as file: file.write("\n")
    with pytest.raises(ValueError, match="checksum"): verify(tmp_path)


def test_reject_invalid_price_bounds():
    frame = pd.DataFrame({"date": ["2020-01-01"], "open": [100], "close": [102], "high": [99], "low": [90], "adj_close": [102], "volume": [100]})
    with pytest.raises(ValueError, match="OHLC"): validate_prices(frame)


def test_asof_never_backfills_or_uses_stale_data():
    left = pd.DataFrame({"signal_date": pd.to_datetime(["2020-01-01", "2020-01-04", "2020-02-01"])})
    right = pd.DataFrame({"available_date": pd.to_datetime(["2020-01-03"]), "value": [7]})
    result = available_join(left, right, "value", "7D")
    assert pd.isna(result.value.iloc[0])
    assert result.value.iloc[1] == 7
    assert pd.isna(result.value.iloc[2])


def test_features_have_no_future_availability(features):
    for col in features.filter(like="_available_date"):
        assert (features[col] <= features.signal_date).all()
    assert (features.trade_date > features.signal_date).all()
    assert not features[FEATURE_SETS["Full"]].isna().any().any()
    assert not set(FEATURE_SETS["Full"]) & {"target_up", "target_return", "buy_hold_return"}


def test_future_input_mutation_does_not_change_past_predictors(tmp_path, features):
    shutil.copytree(ROOT / "data", tmp_path / "data")
    raw = tmp_path / "data/raw"
    prices = pd.read_csv(raw / "prices.csv")
    cols = ["open", "high", "low", "close", "adj_close"]
    prices.loc[prices.date > "2023-01-01", cols] *= 1.5
    prices.to_csv(raw / "prices.csv", index=False)
    sentiment = pd.read_csv(raw / "sentiment.csv")
    sentiment.loc[sentiment.observation_date > "2023-01-01", "news_sentiment"] += 100
    sentiment.to_csv(raw / "sentiment.csv", index=False)
    manifest = json.loads((raw / "provenance.json").read_text())
    for file in manifest["files"]:
        manifest["files"][file]["sha256"] = hashlib.sha256((raw / file).read_bytes()).hexdigest()
    (raw / "provenance.json").write_text(json.dumps(manifest))
    other = build(tmp_path)
    cols = ["signal_date", *FEATURE_SETS["Full"]]
    pd.testing.assert_frame_equal(features.loc[features.signal_date < "2023-01-01", cols].reset_index(drop=True), other.loc[other.signal_date < "2023-01-01", cols].reset_index(drop=True), check_exact=False, atol=1e-8)


def test_purged_chronological_splits(features):
    for year in [2019, 2020, 2021, 2022]:
        train, validation = split_for_year(features, year)
        assert train.trade_date.max() < validation.signal_date.min()
        between = features.loc[(features.trade_date > train.trade_date.max()) & (features.trade_date < validation.trade_date.min())]
        assert len(between) == 5
        assert not set(train.index) & set(validation.index)


def test_target_is_next_session_open_to_close(features):
    prices = pd.read_csv(ROOT / "data/raw/prices.csv", parse_dates=["date"]).set_index("date")
    aligned = prices.loc[features.trade_date]
    np.testing.assert_allclose(features.target_return, (aligned.close / aligned.open - 1).to_numpy())


def test_every_active_day_pays_both_sides_even_consecutive_days():
    position, net = returns_for([.6, .6, .4], [.01, -.01, .02], threshold=.55, cost_bps=2)
    np.testing.assert_allclose(net, [.0096, -.0104, 0])
    np.testing.assert_array_equal(position, [1, 1, 0])


def test_costs_monotonically_reduce_return():
    _, free = returns_for([.6] * 5, [.01] * 5, cost_bps=0)
    _, costly = returns_for([.6] * 5, [.01] * 5, cost_bps=10)
    assert finance(costly)["total_return"] < finance(free)["total_return"]


def test_first_day_loss_counts_toward_drawdown():
    assert finance([-.1, 0])["max_drawdown"] == pytest.approx(-.1)
    assert finance([0, 0], [0, 0])["hit_rate"] is None


def test_bootstrap_reproducible_and_zero_difference():
    a = np.linspace(-.02, .03, 80)
    assert block_bootstrap_difference(a, a, repetitions=40)["lower_95"] == 0
    assert block_bootstrap_difference(a, a / 2, repetitions=40) == block_bootstrap_difference(a, a / 2, repetitions=40)
