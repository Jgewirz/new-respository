#!/usr/bin/env python3
"""
Kalshi Sports Markets Consumer Script.

Production-grade script to fetch sports market data from Kalshi's REST API.
Implements discovery, pagination, optional authentication, and robust error handling.

Usage:
    python scripts/consume_kalshi_sports_markets.py --sport NFL --with-orderbook --stdout

Docs References:
    - Quick Start Market Data: https://docs.kalshi.com/getting_started/quick_start_market_data
    - Authenticated Requests: https://docs.kalshi.com/getting_started/quick_start_authenticated_requests
    - Pagination: https://docs.kalshi.com/getting_started/pagination
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default base URLs per docs:
# - Public market data: https://api.elections.kalshi.com (quick_start_market_data)
# - Demo authenticated: https://demo-api.kalshi.co (quick_start_authenticated_requests)
# - Production authenticated: https://api.kalshi.com
DEFAULT_BASE_URL = "https://api.elections.kalshi.com"
DEFAULT_TRADE_API_PREFIX = "/trade-api/v2"
DEFAULT_DEMO_BASE_URL = "https://demo-api.kalshi.co"

# Endpoint limits per API reference
MAX_MARKETS_LIMIT = 1000  # /markets max limit per docs


@dataclass
class Config:
    """Runtime configuration from env vars and CLI args."""

    base_url: str = DEFAULT_BASE_URL
    trade_api_prefix: str = DEFAULT_TRADE_API_PREFIX
    demo_base_url: str = DEFAULT_DEMO_BASE_URL
    api_key_id: str | None = None
    private_key_path: str | None = None
    use_auth: bool = False

    # CLI options
    sport: str | None = None
    competition: str | None = None
    status: str = "open"
    limit: int = 100
    max_series: int | None = None
    max_markets_per_series: int | None = None
    with_orderbook: bool = False
    out: str = "kalshi_sports_markets.jsonl"
    stdout: bool = False
    timeout: float = 20.0
    sleep: float = 0.1
    retries: int = 5

    @property
    def full_base_url(self) -> str:
        """Return base URL with trade API prefix."""
        return f"{self.base_url}{self.trade_api_prefix}"


@dataclass
class Stats:
    """Accumulator for run statistics."""

    sports_discovered: int = 0
    series_scanned: int = 0
    markets_written: int = 0
    orderbooks_fetched: int = 0
    errors: list[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)


# =============================================================================
# AUTHENTICATION (RSA-PSS Signing)
# Docs: https://docs.kalshi.com/getting_started/quick_start_authenticated_requests
#
# Headers required:
#   KALSHI-ACCESS-KEY: API key ID
#   KALSHI-ACCESS-TIMESTAMP: Unix timestamp in milliseconds
#   KALSHI-ACCESS-SIGNATURE: Base64-encoded RSA-PSS signature
#
# Signature message format: "{timestamp}{METHOD}{path}"
#   - timestamp: milliseconds since epoch (string)
#   - METHOD: HTTP method uppercase (GET, POST, etc.)
#   - path: Request path WITHOUT query parameters (e.g., /trade-api/v2/markets)
#
# Signing algorithm: RSA-PSS with SHA-256, salt length = digest length
# =============================================================================


def load_private_key(key_path: str) -> Any:
    """
    Load RSA private key from PEM file.
    Per docs, the key should be in PEM format (.key or .pem).
    """
    from cryptography.hazmat.primitives import serialization

    path = Path(key_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Private key file not found: {path}")

    key_data = path.read_bytes()
    private_key = serialization.load_pem_private_key(key_data, password=None)
    return private_key


def sign_request(
    private_key: Any,
    timestamp_ms: int,
    method: str,
    path: str,
) -> str:
    """
    Sign a request per Kalshi's authentication docs.

    Signature message: "{timestamp}{METHOD}{path}"
    - path is WITHOUT query params
    - Uses RSA-PSS with SHA-256

    Returns: Base64-encoded signature string
    """
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding

    # Build message exactly as documented: timestamp + method + path (no query)
    # Docs: "message = timestamp_str + method + path"
    message = f"{timestamp_ms}{method.upper()}{path}"
    message_bytes = message.encode("utf-8")

    # RSA-PSS signing with SHA-256 per docs
    signature = private_key.sign(
        message_bytes,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH,
        ),
        hashes.SHA256(),
    )

    return base64.b64encode(signature).decode("utf-8")


def get_auth_headers(
    api_key_id: str,
    private_key: Any,
    method: str,
    path: str,
) -> dict[str, str]:
    """
    Generate authentication headers per docs.

    Headers:
    - KALSHI-ACCESS-KEY: The API key ID
    - KALSHI-ACCESS-TIMESTAMP: Current time in milliseconds
    - KALSHI-ACCESS-SIGNATURE: RSA-PSS signature (base64)
    """
    timestamp_ms = int(time.time() * 1000)
    signature = sign_request(private_key, timestamp_ms, method, path)

    return {
        "KALSHI-ACCESS-KEY": api_key_id,
        "KALSHI-ACCESS-TIMESTAMP": str(timestamp_ms),
        "KALSHI-ACCESS-SIGNATURE": signature,
    }


# =============================================================================
# HTTP CLIENT WITH RETRIES
# =============================================================================


class KalshiClient:
    """
    HTTP client for Kalshi API with retry logic and optional authentication.

    Implements exponential backoff on 429/5xx errors as recommended.
    """

    def __init__(self, config: Config, stats: Stats) -> None:
        self.config = config
        self.stats = stats
        self.client = httpx.Client(timeout=config.timeout)
        self.private_key: Any | None = None

        # Load auth credentials if enabled
        if config.use_auth:
            if not config.api_key_id or not config.private_key_path:
                print(
                    "ERROR: --use-auth requires KALSHI_API_KEY_ID and "
                    "KALSHI_PRIVATE_KEY_PATH environment variables.",
                    file=sys.stderr,
                )
                sys.exit(2)
            try:
                self.private_key = load_private_key(config.private_key_path)
                print(f"Loaded private key from: {config.private_key_path}")
            except Exception as e:
                print(f"ERROR: Failed to load private key: {e}", file=sys.stderr)
                sys.exit(2)

    def close(self) -> None:
        self.client.close()

    def _build_headers(self, method: str, path: str) -> dict[str, str]:
        """Build request headers, optionally including auth."""
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        if self.config.use_auth and self.private_key and self.config.api_key_id:
            auth_headers = get_auth_headers(
                self.config.api_key_id,
                self.private_key,
                method,
                path,
            )
            headers.update(auth_headers)

        return headers

    def get(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """
        Make GET request with retries and exponential backoff.

        Args:
            endpoint: API endpoint path (e.g., /search/filters_by_sport)
            params: Query parameters

        Returns:
            JSON response dict or None on failure
        """
        url = f"{self.config.full_base_url}{endpoint}"

        # Path for signing excludes query params per docs
        sign_path = f"{self.config.trade_api_prefix}{endpoint}"

        for attempt in range(self.config.retries):
            try:
                headers = self._build_headers("GET", sign_path)
                response = self.client.get(url, params=params, headers=headers)

                if response.status_code == 200:
                    return response.json()

                # Retry on 429 (rate limit) or 5xx (server errors)
                if response.status_code == 429 or response.status_code >= 500:
                    wait_time = (2**attempt) * 0.5  # Exponential backoff
                    print(
                        f"  Retry {attempt + 1}/{self.config.retries} "
                        f"(status {response.status_code}), waiting {wait_time:.1f}s..."
                    )
                    time.sleep(wait_time)
                    continue

                # Non-retryable error
                error_msg = f"HTTP {response.status_code} for {endpoint}: {response.text[:200]}"
                self.stats.add_error(error_msg)
                return None

            except httpx.RequestError as e:
                wait_time = (2**attempt) * 0.5
                print(
                    f"  Network error on attempt {attempt + 1}/{self.config.retries}: {e}"
                )
                if attempt < self.config.retries - 1:
                    time.sleep(wait_time)
                else:
                    self.stats.add_error(f"Network error for {endpoint}: {e}")
                    return None

        return None

    def get_paginated(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        results_key: str = "markets",
        max_items: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Fetch all pages from a paginated endpoint.

        Pagination per docs (https://docs.kalshi.com/getting_started/pagination):
        - Request includes 'cursor' and 'limit' params
        - Response includes 'cursor' field; empty string means no more pages

        Args:
            endpoint: API endpoint
            params: Base query parameters
            results_key: Key in response containing the list of items
            max_items: Optional cap on total items to fetch

        Returns:
            List of all items across pages
        """
        all_items: list[dict[str, Any]] = []
        params = dict(params) if params else {}
        cursor: str | None = None

        while True:
            if cursor:
                params["cursor"] = cursor

            response = self.get(endpoint, params)
            if not response:
                break

            items = response.get(results_key, [])
            if not items:
                break

            all_items.extend(items)

            # Check if we've hit our max
            if max_items and len(all_items) >= max_items:
                all_items = all_items[:max_items]
                break

            # Get next cursor; empty string or missing means done
            cursor = response.get("cursor", "")
            if not cursor:
                break

            # Respect rate limiting
            time.sleep(self.config.sleep)

        return all_items


# =============================================================================
# DISCOVERY AND FETCHING LOGIC
# =============================================================================


def discover_sports(client: KalshiClient) -> tuple[list[str], dict[str, Any]]:
    """
    Fetch available sports filters.

    Endpoint: GET /trade-api/v2/search/filters_by_sport
    Docs: https://docs.kalshi.com/api-reference/search/get-filters-for-sports

    Returns:
        Tuple of (sport_ordering list, full filters_by_sports dict)
    """
    print("Discovering sports filters...")
    response = client.get("/search/filters_by_sport")

    if not response:
        return [], {}

    # Response contains sport_ordering (list) and filters_by_sports (dict)
    sport_ordering = response.get("sport_ordering", [])
    filters_by_sports = response.get("filters_by_sports", {})

    client.stats.sports_discovered = len(sport_ordering)
    print(f"  Found {len(sport_ordering)} sports: {sport_ordering}")

    return sport_ordering, filters_by_sports


def discover_categories(client: KalshiClient) -> dict[str, list[str]]:
    """
    Fetch category tags for series filtering.

    Endpoint: GET /trade-api/v2/search/tags_by_categories
    Docs: https://docs.kalshi.com/api-reference/search/get-tags-for-series-categories

    Returns:
        Dict mapping category names to their tags
    """
    print("Discovering category tags...")
    response = client.get("/search/tags_by_categories")

    if not response:
        return {}

    # Response structure: { "category_name": ["tag1", "tag2", ...], ... }
    categories = response.get("tags_by_categories", response)

    # Handle case where response is the dict directly
    if isinstance(categories, dict):
        print(f"  Found {len(categories)} categories: {list(categories.keys())}")
        return categories

    return {}


def find_sports_category(categories: dict[str, list[str]]) -> tuple[str, list[str]]:
    """
    Find the sports category (case-insensitive search).

    Returns:
        Tuple of (category_key, tags_list)
    """
    for key, tags in categories.items():
        if key.lower() == "sports":
            return key, tags

    # Fallback: look for any category containing "sport"
    for key, tags in categories.items():
        if "sport" in key.lower():
            return key, tags

    # Default fallback
    print("  Warning: 'sports' category not found, using 'Sports' as fallback")
    return "Sports", []


def fetch_series(
    client: KalshiClient,
    category: str,
    tags: list[str] | None = None,
    sport_filter: str | None = None,
) -> list[dict[str, Any]]:
    """
    Fetch series list, optionally filtered by category and tags.

    Endpoint: GET /trade-api/v2/series
    Docs: https://docs.kalshi.com/api-reference/market/get-series-list

    Query params:
        - category: Filter by category
        - tags: Comma-separated list of tags
    """
    print(f"Fetching series for category '{category}'...")

    params: dict[str, Any] = {}
    if category:
        params["category"] = category

    # If we have tags and a sport filter, try to match
    if tags and sport_filter:
        matching_tags = [t for t in tags if sport_filter.lower() in t.lower()]
        if matching_tags:
            params["tags"] = ",".join(matching_tags)
            print(f"  Filtering by tags: {matching_tags}")

    response = client.get("/series", params)

    if not response:
        return []

    series_list = response.get("series", [])
    print(f"  Found {len(series_list)} series")

    return series_list


def fetch_markets_for_series(
    client: KalshiClient,
    series_ticker: str,
    status: str,
    limit: int,
    max_markets: int | None = None,
) -> list[dict[str, Any]]:
    """
    Fetch markets for a given series with pagination.

    Endpoint: GET /trade-api/v2/markets
    Docs: https://docs.kalshi.com/api-reference/market/get-markets

    Query params:
        - series_ticker: Filter by series
        - status: open, closed, settled
        - limit: Max per page (up to 1000)
        - cursor: Pagination cursor
    """
    params = {
        "series_ticker": series_ticker,
        "status": status,
        "limit": min(limit, MAX_MARKETS_LIMIT),
    }

    markets = client.get_paginated(
        "/markets",
        params=params,
        results_key="markets",
        max_items=max_markets,
    )

    return markets


def fetch_orderbook(client: KalshiClient, ticker: str) -> dict[str, Any] | None:
    """
    Fetch orderbook for a specific market.

    Endpoint: GET /trade-api/v2/markets/{ticker}/orderbook
    Docs: https://docs.kalshi.com/api-reference/market/get-market-orderbook

    Note from docs: Orderbook returns bids only, not asks.
    """
    response = client.get(f"/markets/{ticker}/orderbook")

    if response:
        client.stats.orderbooks_fetched += 1

    return response


def normalize_market(
    market: dict[str, Any],
    series_ticker: str,
    orderbook: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Normalize market data into a consistent schema.

    Handles field variations between API versions:
    - API reference shows: yes_bid, yes_ask, last_price
    - Some examples show: yes_price
    - We prefer API reference fields and fall back safely.
    """
    # Core identifiers
    normalized = {
        "series_ticker": series_ticker,
        "event_ticker": market.get("event_ticker", ""),
        "market_ticker": market.get("ticker", ""),
        "title": market.get("title", ""),
        "subtitle": market.get("subtitle", ""),
        "status": market.get("status", ""),
    }

    # Timing fields (may not all be present)
    for time_field in ["open_time", "close_time", "expiration_time"]:
        if time_field in market:
            normalized[time_field] = market[time_field]

    # Pricing fields - prefer API reference fields, fall back to alternatives
    # API ref: yes_bid, yes_ask, no_bid, no_ask, last_price
    # Legacy/alt: yes_price, no_price
    pricing = {}

    # Yes side
    if "yes_bid" in market:
        pricing["yes_bid"] = market["yes_bid"]
    if "yes_ask" in market:
        pricing["yes_ask"] = market["yes_ask"]
    if "yes_price" in market and "yes_bid" not in pricing:
        # Fallback: use yes_price as both bid and ask indicator
        pricing["yes_price"] = market["yes_price"]

    # No side
    if "no_bid" in market:
        pricing["no_bid"] = market["no_bid"]
    if "no_ask" in market:
        pricing["no_ask"] = market["no_ask"]
    if "no_price" in market and "no_bid" not in pricing:
        pricing["no_price"] = market["no_price"]

    # Last trade price
    if "last_price" in market:
        pricing["last_price"] = market["last_price"]

    normalized["pricing"] = pricing

    # Volume and liquidity
    if "volume" in market:
        normalized["volume"] = market["volume"]
    if "volume_24h" in market:
        normalized["volume_24h"] = market["volume_24h"]
    if "liquidity" in market:
        normalized["liquidity"] = market["liquidity"]

    # Orderbook data (top 5 levels)
    if orderbook:
        ob_data = {}

        # Yes bids (per docs, orderbook returns bids only)
        yes_bids = orderbook.get("yes", [])
        if yes_bids:
            ob_data["yes_bids"] = yes_bids[:5]

        # No bids
        no_bids = orderbook.get("no", [])
        if no_bids:
            ob_data["no_bids"] = no_bids[:5]

        if ob_data:
            normalized["orderbook"] = ob_data

    # Metadata
    normalized["_meta"] = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source_endpoint": "/markets",
    }

    return normalized


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Consume Kalshi Sports market data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--sport",
        type=str,
        help="Sport to filter (e.g., NFL, NBA). Case-insensitive.",
    )
    parser.add_argument(
        "--competition",
        type=str,
        help="Competition to filter within the sport",
    )
    parser.add_argument(
        "--status",
        type=str,
        default="open",
        choices=["open", "closed", "settled"],
        help="Market status filter",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help=f"Results per page (max {MAX_MARKETS_LIMIT})",
    )
    parser.add_argument(
        "--max-series",
        type=int,
        help="Maximum number of series to scan",
    )
    parser.add_argument(
        "--max-markets-per-series",
        type=int,
        help="Maximum markets to fetch per series",
    )
    parser.add_argument(
        "--with-orderbook",
        action="store_true",
        help="Fetch orderbook for each market",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="kalshi_sports_markets.jsonl",
        help="Output file path (JSONL format)",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print summary to stdout",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="HTTP request timeout in seconds",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.1,
        help="Sleep between requests in seconds",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="Number of retries on failure",
    )
    parser.add_argument(
        "--use-auth",
        action="store_true",
        help="Enable authenticated requests (requires env vars)",
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    """Build configuration from env vars and CLI args."""
    config = Config(
        # Env vars with defaults
        base_url=os.environ.get("KALSHI_BASE_URL", DEFAULT_BASE_URL),
        trade_api_prefix=os.environ.get("KALSHI_TRADE_API_PREFIX", DEFAULT_TRADE_API_PREFIX),
        demo_base_url=os.environ.get("KALSHI_DEMO_BASE_URL", DEFAULT_DEMO_BASE_URL),
        api_key_id=os.environ.get("KALSHI_API_KEY_ID"),
        private_key_path=os.environ.get("KALSHI_PRIVATE_KEY_PATH"),
        use_auth=os.environ.get("KALSHI_USE_AUTH", "false").lower() == "true" or args.use_auth,
        # CLI args
        sport=args.sport,
        competition=args.competition,
        status=args.status,
        limit=min(args.limit, MAX_MARKETS_LIMIT),
        max_series=args.max_series,
        max_markets_per_series=args.max_markets_per_series,
        with_orderbook=args.with_orderbook,
        out=args.out,
        stdout=args.stdout,
        timeout=args.timeout,
        sleep=args.sleep,
        retries=args.retries,
    )

    return config


def filter_series_by_sport(
    series_list: list[dict[str, Any]],
    sport: str,
    filters_by_sports: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Filter series list to match the requested sport.

    Uses filters_by_sports data to identify relevant series.
    Falls back to text matching on series title/tags.
    """
    sport_lower = sport.lower()

    # Get series tickers from sport filters if available
    sport_data = filters_by_sports.get(sport) or filters_by_sports.get(sport.upper())
    if sport_data:
        # Extract series tickers from the sport filter data
        series_tickers = set()
        if isinstance(sport_data, dict):
            for key, value in sport_data.items():
                if isinstance(value, list):
                    series_tickers.update(value)
                elif isinstance(value, dict):
                    for sub_value in value.values():
                        if isinstance(sub_value, list):
                            series_tickers.update(sub_value)

        if series_tickers:
            filtered = [s for s in series_list if s.get("ticker") in series_tickers]
            if filtered:
                return filtered

    # Fallback: match on title or tags
    filtered = []
    for series in series_list:
        title = (series.get("title") or "").lower()
        tags = [t.lower() for t in (series.get("tags") or [])]

        if sport_lower in title or sport_lower in tags:
            filtered.append(series)

    return filtered if filtered else series_list


def print_summary(stats: Stats, config: Config) -> None:
    """Print human-friendly end-of-run summary."""
    print("\n" + "=" * 60)
    print("KALSHI SPORTS MARKETS - RUN SUMMARY")
    print("=" * 60)
    print(f"Sports discovered:    {stats.sports_discovered}")
    print(f"Series scanned:       {stats.series_scanned}")
    print(f"Markets written:      {stats.markets_written}")
    if config.with_orderbook:
        print(f"Orderbooks fetched:   {stats.orderbooks_fetched}")
    print(f"Errors encountered:   {len(stats.errors)}")

    if stats.errors:
        print("\nFirst 5 errors:")
        for i, error in enumerate(stats.errors[:5], 1):
            print(f"  {i}. {error[:100]}...")

    print(f"\nOutput file: {config.out}")
    print("=" * 60)


def main() -> int:
    """Main entry point."""
    args = parse_args()
    config = build_config(args)
    stats = Stats()

    print(f"Kalshi Sports Markets Consumer")
    print(f"Base URL: {config.base_url}")
    print(f"Auth enabled: {config.use_auth}")
    if config.sport:
        print(f"Sport filter: {config.sport}")
    print()

    client = KalshiClient(config, stats)

    try:
        # Step 1: Discover available sports
        # Endpoint: GET /search/filters_by_sport
        sport_ordering, filters_by_sports = discover_sports(client)

        if config.sport:
            # Validate sport exists (case-insensitive)
            sport_upper = config.sport.upper()
            valid_sports = [s.upper() for s in sport_ordering]

            if sport_upper not in valid_sports:
                print(f"\nERROR: Sport '{config.sport}' not found.")
                print(f"Available sports: {sport_ordering}")
                return 2

        # Step 2: Discover category tags
        # Endpoint: GET /search/tags_by_categories
        categories = discover_categories(client)
        sports_category, sports_tags = find_sports_category(categories)

        # Step 3: Fetch series
        # Endpoint: GET /series
        series_list = fetch_series(
            client,
            category=sports_category,
            tags=sports_tags,
            sport_filter=config.sport,
        )

        # Filter series by sport if specified
        if config.sport and series_list:
            series_list = filter_series_by_sport(
                series_list,
                config.sport,
                filters_by_sports,
            )
            print(f"  Filtered to {len(series_list)} series matching '{config.sport}'")

        # Apply max_series limit
        if config.max_series and len(series_list) > config.max_series:
            series_list = series_list[: config.max_series]
            print(f"  Limited to {config.max_series} series")

        # Step 4: Fetch markets for each series
        # Endpoint: GET /markets with pagination
        output_path = Path(config.out)
        with output_path.open("w", encoding="utf-8") as outfile:
            for series in series_list:
                series_ticker = series.get("ticker", "")
                if not series_ticker:
                    continue

                stats.series_scanned += 1
                print(f"\nFetching markets for series: {series_ticker}")

                try:
                    markets = fetch_markets_for_series(
                        client,
                        series_ticker=series_ticker,
                        status=config.status,
                        limit=config.limit,
                        max_markets=config.max_markets_per_series,
                    )

                    print(f"  Found {len(markets)} markets")

                    for market in markets:
                        ticker = market.get("ticker", "")

                        # Optionally fetch orderbook
                        # Endpoint: GET /markets/{ticker}/orderbook
                        orderbook = None
                        if config.with_orderbook and ticker:
                            orderbook = fetch_orderbook(client, ticker)
                            time.sleep(config.sleep)

                        # Normalize and write
                        normalized = normalize_market(market, series_ticker, orderbook)
                        outfile.write(json.dumps(normalized) + "\n")
                        stats.markets_written += 1

                    time.sleep(config.sleep)

                except Exception as e:
                    error_msg = f"Error processing series {series_ticker}: {e}"
                    stats.add_error(error_msg)
                    print(f"  ERROR: {e}")
                    continue

        # Print summary
        if config.stdout:
            print_summary(stats, config)
        else:
            print(f"\nDone. Wrote {stats.markets_written} markets to {config.out}")
            if stats.errors:
                print(f"Encountered {len(stats.errors)} errors.")

        return 0 if not stats.errors else 1

    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
