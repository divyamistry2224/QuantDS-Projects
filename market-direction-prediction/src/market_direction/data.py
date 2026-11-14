"""Public-source ingestion with explicit provenance and normalized snapshots."""
from __future__ import annotations

import hashlib
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
START = "2004-01-01"
END = "2025-12-31"
FRED_SERIES = ["DFF", "T10Y2Y", "VIXCLS", "UNRATE", "CPIAUCSL"]


def download(url: str) -> bytes:
    for attempt in range(3):
        response = requests.get(url, headers={"User-Agent": "MarketDirectionResearch/1.0 (academic research)"}, timeout=45)
        if response.status_code in (429, 500, 502, 503) and attempt < 2:
            time.sleep(2 ** attempt)
            continue
        response.raise_for_status()
        return response.content
    raise RuntimeError("Download retries exhausted")


def validate_prices(df: pd.DataFrame) -> None:
    if df.empty or df.date.duplicated().any() or not df.date.is_monotonic_increasing:
        raise ValueError("Prices must contain unique, ascending dates")
    prices = df[["open", "high", "low", "close", "adj_close"]]
    if prices.isna().any().any() or (prices <= 0).any().any() or df.volume.isna().any() or (df.volume < 0).any():
        raise ValueError("Missing/nonpositive price or invalid volume")
    if ((df.high < df[["open", "close"]].max(axis=1) - 1e-5) | (df.low > df[["open", "close"]].min(axis=1) + 1e-5)).any():
        raise ValueError("OHLC bounds violated")


def fetch(root: Path = ROOT, start_date: str = START, end_date: str = END) -> dict:
    target = root / "data/raw"
    target.mkdir(parents=True, exist_ok=True)
    records = []

    def get(url: str, name: str) -> bytes:
        raw = download(url)
        records.append({"name": name, "url": url, "retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "response_sha256": hashlib.sha256(raw).hexdigest()})
        return raw

    start = int(pd.Timestamp(start_date, tz="UTC").timestamp())
    end = int((pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)).timestamp())
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/SPY?period1={start}&period2={end}&interval=1d"
    result = json.loads(get(url, "SPY OHLCV and adjusted close"))["chart"]["result"][0]
    quote = result["indicators"]["quote"][0]
    prices = pd.DataFrame({"date": pd.to_datetime(result["timestamp"], unit="s", utc=True).tz_convert("America/New_York").tz_localize(None).normalize(), **quote, "adj_close": result["indicators"]["adjclose"][0]["adjclose"]})
    prices = prices.sort_values("date").reset_index(drop=True)
    validate_prices(prices)
    prices.to_csv(target / "prices.csv", index=False, float_format="%.8f")

    frames = []
    for series in FRED_SERIES:
        raw = get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd={start_date}&coed={end_date}", series)
        frame = pd.read_csv(io.BytesIO(raw), na_values=["."])
        frame.columns = ["observation_date", "value"]
        frame["observation_date"] = pd.to_datetime(frame.observation_date)
        frame = frame.loc[frame.observation_date.between(start_date, end_date)].dropna().assign(series=series)
        frames.append(frame)
    pd.concat(frames).to_csv(target / "macro.csv", index=False, float_format="%.8f")
    raw = get("https://www.frbsf.org/wp-content/uploads/news_sentiment_data.xlsx", "FRBSF Daily News Sentiment Index")
    sentiment = pd.read_excel(io.BytesIO(raw), sheet_name="Data")
    sentiment.columns = ["observation_date", "news_sentiment"]
    sentiment["observation_date"] = pd.to_datetime(sentiment.observation_date)
    sentiment = sentiment.loc[sentiment.observation_date.between(start_date, end_date)].dropna().sort_values("observation_date")
    if sentiment.observation_date.duplicated().any():
        raise ValueError("Duplicate sentiment observations")
    sentiment.to_csv(target / "sentiment.csv", index=False, float_format="%.10f")
    manifest = {"snapshot_start": start_date, "snapshot_end": end_date, "sources": records, "files": {}, "vintage": "Latest downloaded revisions, not point-in-time vintages. Availability lags are assumptions, not verified historical publication timestamps."}
    for file in ["prices.csv", "macro.csv", "sentiment.csv"]:
        manifest["files"][file] = {"sha256": hashlib.sha256((target / file).read_bytes()).hexdigest(), "rows": len(pd.read_csv(target / file))}
    (target / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify(root: Path = ROOT) -> dict:
    manifest = json.loads((root / "data/raw/provenance.json").read_text())
    for file, info in manifest["files"].items():
        if hashlib.sha256((root / "data/raw" / file).read_bytes()).hexdigest() != info["sha256"]:
            raise ValueError(f"Snapshot checksum mismatch: {file}")
    validate_prices(pd.read_csv(root / "data/raw/prices.csv", parse_dates=["date"]))
    return manifest
