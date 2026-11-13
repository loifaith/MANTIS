#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# btc_free_dataset_builder.py
#
# Build a Bitcoin prediction dataset using ONLY free sources:
# - Binance spot 1m OHLCV (BTCUSDT)
# - Binance futures funding rate (perpetual, historical)
# - Fear & Greed Index (alternative.me, daily)
# - Google Trends (pytrends) for queries: ["bitcoin", "btc price", "bitcoin crash"]
# - Yahoo Finance (yfinance) cross-assets: SP500 (^GSPC), Gold (GC=F), DXY (DX-Y.NYB) daily
# - Blockchain.com on-chain stats: n-unique-addresses, hash-rate (daily)
#
# Output:
# - A primary 1m dataset with engineered features (CSV Parquet)
# - A daily join table for slow-moving features, forward-filled to minutes
#
# Usage (examples):
#     python btc_free_dataset_builder.py --start 2019-01-01 --end 2025-01-01 --save-dir ./data
#     python btc_free_dataset_builder.py --minutes-only --start 2024-01-01 --end 2024-03-01
#
import os
import sys
import time
import json
import math
import argparse
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any

import pandas as pd
import numpy as np
import requests

# Optional deps: installed via requirements.txt
try:
    from pytrends.request import TrendReq
except Exception:
    TrendReq = None

try:
    import yfinance as yf
except Exception:
    yf = None

# ---------------------------------------------
# Helpers
# ---------------------------------------------
def to_ms(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)

def from_ms(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)

def clamp_end(end: Optional[datetime]) -> datetime:
    return end or datetime.now(tz=timezone.utc)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def safe_request(url: str, params: Optional[Dict[str, Any]] = None, retries: int = 5, backoff: float = 0.5) -> requests.Response:
    for i in range(retries):
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 200:
            return r
        time.sleep(backoff * (2 ** i))
    r.raise_for_status()
    return r

# ---------------------------------------------
# Binance: Spot 1m Klines (BTCUSDT)
# ---------------------------------------------
def fetch_binance_klines_1m(symbol: str, start: datetime, end: Optional[datetime] = None, limit: int = 1000, rate_sleep: float = 0.2) -> pd.DataFrame:
    url = "https://api.binance.com/api/v3/klines"
    end = clamp_end(end)
    start_ms = to_ms(start)
    rows: List[List[Any]] = []
    while True:
        params = {
            "symbol": symbol,
            "interval": "1m",
            "limit": limit,
            "startTime": start_ms
        }
        r = safe_request(url, params)
        data = r.json()
        if not data:
            break
        rows.extend(data)
        # Next page: use last close time + 1 ms to avoid duplicates
        next_ms = int(data[-1][6]) + 1
        if next_ms >= to_ms(end):
            break
        start_ms = next_ms
        time.sleep(rate_sleep)
        if len(data) < limit:
            break
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=[
        "open_time","open","high","low","close","volume","close_time",
        "quote_asset_volume","number_of_trades","taker_buy_base","taker_buy_quote","ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    num_cols = ["open","high","low","close","volume","quote_asset_volume","taker_buy_base","taker_buy_quote"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["number_of_trades"] = pd.to_numeric(df["number_of_trades"], errors="coerce").astype("Int64")
    df = df.drop(columns=["ignore"])
    df = df.sort_values("open_time").reset_index(drop=True)
    return df

# ---------------------------------------------
# Binance: Funding Rate History (BTCUSDT perpetual)
# ---------------------------------------------
def fetch_binance_funding(symbol: str, start: datetime, end: Optional[datetime] = None, limit: int = 1000) -> pd.DataFrame:
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    end = clamp_end(end)
    start_ms = to_ms(start)
    rows: List[Dict[str, Any]] = []
    while True:
        params = {
            "symbol": symbol,
            "limit": limit,
            "startTime": start_ms
        }
        r = safe_request(url, params)
        data = r.json()
        if not data:
            break
        rows.extend(data)
        next_ms = int(data[-1]["fundingTime"]) + 1
        if next_ms >= to_ms(end):
            break
        start_ms = next_ms
        time.sleep(0.2)
        if len(data) < limit:
            break
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    return df.rename(columns={"fundingTime":"timestamp"}).sort_values("timestamp")

# ---------------------------------------------
# Fear & Greed Index (alternative.me), daily
# ---------------------------------------------
def fetch_fear_greed() -> pd.DataFrame:
    url = "https://api.alternative.me/fng/"
    params = {"limit": 0, "format": "json"}
    r = safe_request(url, params)
    data = r.json()["data"]
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
    # value and value_classification
    df["fng_value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[["timestamp","fng_value","value_classification"]].rename(columns={"value_classification":"fng_label"})
    return df.sort_values("timestamp")

# ---------------------------------------------
# Google Trends (pytrends) daily
# ---------------------------------------------
def fetch_google_trends(queries: List[str], start: datetime, end: Optional[datetime] = None, geo: str = "") -> pd.DataFrame:
    if TrendReq is None:
        logging.warning("pytrends not installed; skipping Google Trends.")
        return pd.DataFrame()
    end = clamp_end(end).date()
    start = start.date()
    timeframe = f"{start} {end}"
    tr = TrendReq(hl="en-US", tz=0)
    tr.build_payload(kw_list=queries, timeframe=timeframe, geo=geo, gprop="")
    df = tr.interest_over_time()
    if df.empty:
        return pd.DataFrame()
    df = df.reset_index().rename(columns={"date":"timestamp"})
    # Drop isPartial if present
    if "isPartial" in df.columns:
        df = df.drop(columns=["isPartial"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    # Rename columns
    rename_map = {q: f"gt_{q.replace(' ', '_')}" for q in queries}
    df = df.rename(columns=rename_map)
    return df.sort_values("timestamp")

# ---------------------------------------------
# Yahoo Finance daily series
# ---------------------------------------------
def fetch_yf_daily(tickers: List[str], start: datetime, end: Optional[datetime] = None) -> pd.DataFrame:
    if yf is None:
        logging.warning("yfinance not installed; skipping Yahoo Finance series.")
        return pd.DataFrame()
    end = clamp_end(end)
    df = yf.download(tickers=tickers, start=start.date(), end=(end + timedelta(days=1)).date(), interval="1d", auto_adjust=True, progress=False)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        # Flatten multiindex: (field, ticker) -> field_ticker
        df = df.swaplevel(0,1, axis=1).sort_index(axis=1)
        df.columns = [f"{field.lower()}_{ticker.replace('^','').replace('=','').replace('-','')}" for (ticker, field) in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    df = df.reset_index().rename(columns={"Date":"timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp")

# ---------------------------------------------
# Blockchain.com charts (daily)
# Valid chart names include: 'n-unique-addresses', 'hash-rate', 'n-transactions', 'difficulty'
# docs: https://api.blockchain.info/charts
# ---------------------------------------------
def fetch_blockchain_chart(chart: str, start: datetime, timespan: Optional[str] = None) -> pd.DataFrame:
    # timespan examples: "5years", "all", "1year" etc. If None, we compute via start -> now days.
    if timespan is None:
        days = (datetime.now(tz=timezone.utc) - start).days + 1
        timespan = f"{days}days"
    url = f"https://api.blockchain.info/charts/{chart}"
    params = {"timespan": timespan, "format": "json", "cors": "true"}
    r = safe_request(url, params)
    j = r.json()
    values = j.get("values", [])
    if not values:
        return pd.DataFrame()
    df = pd.DataFrame(values)
    # 'x' seconds, 'y' value
    df["timestamp"] = pd.to_datetime(df["x"], unit="s", utc=True)
    df = df.rename(columns={"y": f"bc_{chart.replace('-', '_')}"})[["timestamp", f"bc_{chart.replace('-', '_')}"]]
    return df.sort_values("timestamp")

# ---------------------------------------------
# Feature engineering (minute-level)
# ---------------------------------------------
def add_intraday_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    # Returns and log-returns
    out["ret_1"] = out["close"].pct_change()
    out["logret_1"] = np.log(out["close"]).diff()
    # Realized volatility (rolling)
    out["rv_5m"] = out["logret_1"].rolling(5).std() * np.sqrt(60*24)
    out["rv_60m"] = out["logret_1"].rolling(60).std() * np.sqrt(60*24)
    # Volume features
    out["vol_z_60m"] = (out["volume"] - out["volume"].rolling(60).mean()) / (out["volume"].rolling(60).std() + 1e-9)
    # Price range
    out["hl_spread_bp"] = (out["high"] - out["low"]) / out["close"] * 10000
    # Taker buy ratio
    out["taker_buy_ratio"] = out["taker_buy_quote"] / (out["quote_asset_volume"] + 1e-9)
    return out

# ---------------------------------------------
# Merge daily slow features to minutes (forward-fill)
# ---------------------------------------------
def as_daily(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    d = df.copy()
    d["date"] = d["timestamp"].dt.floor("D")
    d = d.drop_duplicates("date").set_index("date").sort_index()
    return d

def merge_daily_into_minutes(min_df: pd.DataFrame, daily_tables: List[pd.DataFrame]) -> pd.DataFrame:
    if min_df.empty:
        return min_df
    out = min_df.copy()
    out = out.set_index("open_time")
    # Unify each daily table by day and then reindex onto minute index with ffill
    for d in daily_tables:
        if d.empty: 
            continue
        d2 = d.copy()
        if "timestamp" in d2.columns:
            d2 = d2.rename(columns={"timestamp":"date"})
        d2["date"] = pd.to_datetime(d2["date"], utc=True).dt.floor("D")
        d2 = d2.drop_duplicates("date").set_index("date").sort_index()
        out = out.join(d2, how="left")
    out = out.ffill().reset_index().rename(columns={"open_time":"timestamp"})
    return out

# ---------------------------------------------
# Main
# ---------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Build a free BTC dataset (minute-level with daily sentiment/macros).")
    parser.add_argument("--start", type=str, default="2022-01-01", help="UTC start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="UTC end date (YYYY-MM-DD)")
    parser.add_argument("--save-dir", type=str, default="./data", help="Directory to save outputs")
    parser.add_argument("--minutes-only", action="store_true", help="Skip daily external series and only export minute OHLCV+features")
    parser.add_argument("--no-trends", action="store_true", help="Skip Google Trends (pytrends)")
    parser.add_argument("--no-yf", action="store_true", help="Skip Yahoo Finance cross-assets")
    parser.add_argument("--no-blockchain", action="store_true", help="Skip Blockchain.com on-chain charts")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Binance symbol for spot klines")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc) if args.end else None
    ensure_dir(args.save_dir)

    # 1) Binance 1m klines
    logging.info("Fetching Binance 1m klines...")
    k1 = fetch_binance_klines_1m(args.symbol, start=start, end=end)
    if k1.empty:
        logging.error("No 1m klines fetched. Exiting.")
        sys.exit(1)
    k1 = add_intraday_features(k1)
    k1.to_csv(os.path.join(args.save_dir, f"{args.symbol}_1m_raw.csv"), index=False)
    logging.info(f"Saved {len(k1)} minute rows.")

    if args.minutes_only:
        k1.to_parquet(os.path.join(args.save_dir, f"{args.symbol}_1m_features.parquet"), index=False)
        logging.info("Minutes-only export complete.")
        return

    # 2) Binance funding (approx 8h cadence historically)
    logging.info("Fetching Binance funding rates...")
    fund = fetch_binance_funding(symbol="BTCUSDT", start=start, end=end)

    # 3) Fear & Greed (daily)
    logging.info("Fetching Fear & Greed Index...")
    fng = fetch_fear_greed()

    # 4) Google Trends (daily)
    trends = pd.DataFrame()
    if not args.no_trends:
        logging.info("Fetching Google Trends...")
        queries = ["bitcoin","btc price","bitcoin crash"]
        trends = fetch_google_trends(queries, start=start, end=end)

    # 5) Yahoo Finance daily benchmarks
    yf_daily = pd.DataFrame()
    if not args.no_yf:
        logging.info("Fetching Yahoo Finance macro series...")
        tickers = ["^GSPC","GC=F","DX-Y.NYB"]  # SP500, Gold, DXY
        yf_daily = fetch_yf_daily(tickers, start=start, end=end)

    # 6) Blockchain.com charts (daily)
    bc_daily_list: List[pd.DataFrame] = []
    if not args.no_blockchain:
        logging.info("Fetching Blockchain.com on-chain charts...")
        for chart in ["n-unique-addresses","hash-rate","n-transactions"]:
            try:
                bc = fetch_blockchain_chart(chart, start=start)
                if not bc.empty:
                    bc_daily_list.append(bc)
            except Exception as e:
                logging.warning(f"Blockchain.com chart {chart} failed: {e}")

    # Merge daily
    daily_tables: List[pd.DataFrame] = []
    for d in [fund, fng, trends, yf_daily, *bc_daily_list]:
        if d is not None and not d.empty:
            daily_tables.append(d)

    logging.info("Merging daily features into minute timeline (ffill)...")
    merged = merge_daily_into_minutes(k1, daily_tables)

    # Save outputs
    min_csv = os.path.join(args.save_dir, f"{args.symbol}_1m_merged.csv")
    min_parquet = os.path.join(args.save_dir, f"{args.symbol}_1m_merged.parquet")
    merged.to_csv(min_csv, index=False)
    try:
        merged.to_parquet(min_parquet, index=False)
    except Exception as e:
        logging.warning(f"Parquet save failed: {e}")

    # Also export a daily summary table (day-close resample)
    logging.info("Building daily summary (from minute close)...")
    daily = merged.copy()
    daily = daily.set_index(pd.to_datetime(daily["timestamp"], utc=True)).resample("1D").last().reset_index()
    daily = daily.rename(columns={"index":"timestamp"})
    daily_csv = os.path.join(args.save_dir, f"{args.symbol}_daily_summary.csv")
    daily.to_csv(daily_csv, index=False)

    logging.info(f"Done. Files saved to: {args.save_dir}")
    logging.info(f"- Minutes merged CSV: {min_csv}")
    logging.info(f"- Minutes merged Parquet: {min_parquet}")
    logging.info(f"- Daily summary CSV: {daily_csv}")

if __name__ == "__main__":
    main()