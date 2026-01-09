#!/usr/bin/env python3
"""
Kalshi Sports Markets -> Redis Ingestion (Single File)
Fetches sports markets from Kalshi and stores in Redis Cloud for low-latency access.

Usage:
    python kalshi_to_redis.py                    # Fetch all sports markets
    python kalshi_to_redis.py --sport NFL        # Fetch NFL only
    python kalshi_to_redis.py --continuous       # Run continuously (every 30s)
"""

import argparse
import json
import time
from datetime import datetime, timezone
from typing import Any

import httpx
import redis

# =============================================================================
# CONFIGURATION
# =============================================================================

REDIS_URL = "redis://default:nzLTFSPLBax95HT2kcZcLB67fs4GO2t9@redis-12940.c321.us-east-1-2.ec2.cloud.redislabs.com:12940"
KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"

# Redis key structure
KEY_MARKET = "kalshi:m:{ticker}"           # Hash: market details
KEY_SERIES = "kalshi:s:{series}"           # Set: market tickers in series
KEY_SPORT_IDX = "kalshi:sport:{sport}"     # Set: all tickers for a sport
KEY_PRICES = "kalshi:prices"               # ZSet: ticker -> yes_price * 100
KEY_UPDATED = "kalshi:last_updated"        # String: timestamp

# =============================================================================
# REDIS CLIENT
# =============================================================================

def get_redis() -> redis.Redis:
    """Connect to Redis Cloud."""
    return redis.from_url(REDIS_URL, decode_responses=True)


def store_market(r: redis.Redis, market: dict, series_ticker: str, sport: str) -> None:
    """Store a single market in Redis with indexing."""
    ticker = market.get("ticker", "")
    if not ticker:
        return

    # Extract pricing (handle different API response formats)
    yes_price = market.get("yes_bid") or market.get("yes_price") or 0
    no_price = market.get("no_bid") or market.get("no_price") or 0
    last_price = market.get("last_price", 0)

    # Build market hash
    market_data = {
        "ticker": ticker,
        "series": series_ticker,
        "sport": sport,
        "title": market.get("title", ""),
        "subtitle": market.get("subtitle", ""),
        "status": market.get("status", ""),
        "yes_bid": market.get("yes_bid", 0),
        "yes_ask": market.get("yes_ask", 0),
        "no_bid": market.get("no_bid", 0),
        "no_ask": market.get("no_ask", 0),
        "yes_price": yes_price,
        "no_price": no_price,
        "last_price": last_price,
        "volume": market.get("volume", 0),
        "volume_24h": market.get("volume_24h", 0),
        "open_time": market.get("open_time", ""),
        "close_time": market.get("close_time", ""),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    pipe = r.pipeline()

    # Store market hash
    pipe.hset(KEY_MARKET.format(ticker=ticker), mapping=market_data)
    pipe.expire(KEY_MARKET.format(ticker=ticker), 86400)  # 24h TTL

    # Add to series index
    pipe.sadd(KEY_SERIES.format(series=series_ticker), ticker)
    pipe.expire(KEY_SERIES.format(series=series_ticker), 86400)

    # Add to sport index
    if sport:
        pipe.sadd(KEY_SPORT_IDX.format(sport=sport.lower()), ticker)
        pipe.expire(KEY_SPORT_IDX.format(sport=sport.lower()), 86400)

    # Add to price sorted set (for quick scanning)
    if yes_price:
        pipe.zadd(KEY_PRICES, {ticker: float(yes_price) * 100})

    pipe.execute()


# =============================================================================
# KALSHI API CLIENT
# =============================================================================

class KalshiClient:
    """Simple Kalshi API client with retry logic."""

    def __init__(self, timeout: float = 20.0, retries: int = 5):
        self.client = httpx.Client(timeout=timeout)
        self.retries = retries

    def close(self):
        self.client.close()

    def get(self, endpoint: str, params: dict = None) -> dict | None:
        """GET with exponential backoff on rate limits."""
        url = f"{KALSHI_BASE}{endpoint}"

        for attempt in range(self.retries):
            try:
                resp = self.client.get(url, params=params)
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code in (429, 500, 502, 503):
                    wait = (2 ** attempt) * 0.5
                    print(f"  Retry {attempt+1}/{self.retries} (HTTP {resp.status_code}), wait {wait:.1f}s")
                    time.sleep(wait)
                    continue
                print(f"  Error: HTTP {resp.status_code} for {endpoint}")
                return None
            except httpx.RequestError as e:
                print(f"  Network error: {e}")
                time.sleep(2 ** attempt * 0.5)
        return None

    def get_sports(self) -> list[str]:
        """Discover available sports."""
        resp = self.get("/search/filters_by_sport")
        if resp:
            return resp.get("sport_ordering", [])
        return []

    def get_series(self, category: str = "Sports") -> list[dict]:
        """Fetch all series in a category."""
        resp = self.get("/series", {"category": category})
        if resp:
            return resp.get("series", [])
        return []

    def get_markets(self, series_ticker: str, status: str = "open", limit: int = 200) -> list[dict]:
        """Fetch markets for a series with pagination."""
        all_markets = []
        cursor = None

        while True:
            params = {"series_ticker": series_ticker, "status": status, "limit": limit}
            if cursor:
                params["cursor"] = cursor

            resp = self.get("/markets", params)
            if not resp:
                break

            markets = resp.get("markets", [])
            if not markets:
                break

            all_markets.extend(markets)
            cursor = resp.get("cursor", "")
            if not cursor:
                break

            time.sleep(0.15)  # Rate limit

        return all_markets


# =============================================================================
# MAIN INGESTION
# =============================================================================

def ingest_to_redis(sport_filter: str = None, max_series: int = None) -> dict:
    """Fetch Kalshi sports markets and store in Redis."""

    r = get_redis()
    client = KalshiClient()
    stats = {"series": 0, "markets": 0, "errors": 0}

    try:
        # Test Redis connection
        r.ping()
        print(f"Connected to Redis Cloud")

        # Discover sports
        sports = client.get_sports()
        print(f"Found {len(sports)} sports: {sports[:5]}...")

        # Get series
        series_list = client.get_series("Sports")
        print(f"Found {len(series_list)} series")

        # Apply filters
        if sport_filter:
            sport_lower = sport_filter.lower()
            series_list = [
                s for s in series_list
                if sport_lower in (s.get("title") or "").lower()
                or sport_lower in str(s.get("tags", [])).lower()
            ]
            print(f"Filtered to {len(series_list)} series for '{sport_filter}'")

        if max_series:
            series_list = series_list[:max_series]

        # Process each series
        for series in series_list:
            series_ticker = series.get("ticker", "")
            if not series_ticker:
                continue

            stats["series"] += 1
            print(f"\n[{stats['series']}/{len(series_list)}] {series_ticker}")

            try:
                # Infer sport from series title/ticker
                title = (series.get("title") or "").lower()
                sport = "unknown"
                for s in ["nfl", "nba", "mlb", "nhl", "soccer", "tennis", "golf", "mma", "ncaa"]:
                    if s in title or s in series_ticker.lower():
                        sport = s
                        break

                markets = client.get_markets(series_ticker)
                print(f"  -> {len(markets)} markets")

                for market in markets:
                    store_market(r, market, series_ticker, sport)
                    stats["markets"] += 1

                time.sleep(0.15)

            except Exception as e:
                print(f"  ERROR: {e}")
                stats["errors"] += 1

        # Update timestamp
        r.set(KEY_UPDATED, datetime.now(timezone.utc).isoformat())

    finally:
        client.close()

    return stats


def run_continuous(interval: int = 30, sport_filter: str = None):
    """Run ingestion continuously."""
    print(f"Running continuously every {interval}s (Ctrl+C to stop)")

    while True:
        try:
            print(f"\n{'='*50}")
            print(f"Starting ingestion at {datetime.now()}")
            stats = ingest_to_redis(sport_filter=sport_filter)
            print(f"\nCompleted: {stats['markets']} markets from {stats['series']} series")
            print(f"Sleeping {interval}s...")
            time.sleep(interval)
        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Kalshi -> Redis ingestion")
    parser.add_argument("--sport", help="Filter by sport (e.g., NFL, NBA)")
    parser.add_argument("--max-series", type=int, help="Max series to process")
    parser.add_argument("--continuous", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=30, help="Refresh interval (seconds)")
    args = parser.parse_args()

    print("Kalshi Sports -> Redis Ingestion")
    print(f"Redis: {REDIS_URL.split('@')[1]}")
    print()

    if args.continuous:
        run_continuous(interval=args.interval, sport_filter=args.sport)
    else:
        stats = ingest_to_redis(sport_filter=args.sport, max_series=args.max_series)
        print(f"\n{'='*50}")
        print(f"DONE: {stats['markets']} markets, {stats['series']} series, {stats['errors']} errors")


if __name__ == "__main__":
    main()
