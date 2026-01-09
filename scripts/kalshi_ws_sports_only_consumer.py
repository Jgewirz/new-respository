#!/usr/bin/env python3
"""
kalshi_ws_sports_only_consumer.py

Production-grade Kalshi WebSocket consumer that subscribes ONLY to sports markets.
Uses REST discovery to build the set of sports market_tickers, then connects to
the Kalshi WebSocket and streams ticker/trade/orderbook updates as JSONL.

Usage:
    python scripts/kalshi_ws_sports_only_consumer.py

Environment Variables:
    KALSHI_ENV              - "demo" or "prod" (default: demo)
    KALSHI_KEY_ID           - API key ID
    KALSHI_PRIVATE_KEY_PATH - Path to RSA private key PEM file
    SPORTS_TAGS             - Optional comma-separated tags filter for /series
    SPORTS_SERIES_TICKERS   - Optional comma-separated series tickers (skips discovery)
"""

import asyncio
import base64
import hashlib
import json
import os
import sys
import time
from typing import Optional
from urllib.parse import urlencode

import aiohttp
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


# =============================================================================
# Configuration
# =============================================================================

class KalshiConfig:
    """Configuration loaded from environment variables."""

    def __init__(self):
        self.env = os.getenv("KALSHI_ENV", "demo").lower()
        self.key_id = os.getenv("KALSHI_KEY_ID", "")
        self.private_key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH", "")
        self.sports_tags = os.getenv("SPORTS_TAGS", "")
        self.sports_series_tickers = os.getenv("SPORTS_SERIES_TICKERS", "")

        # Validate required fields
        if not self.key_id:
            raise ValueError("KALSHI_KEY_ID environment variable is required")
        if not self.private_key_path:
            raise ValueError("KALSHI_PRIVATE_KEY_PATH environment variable is required")
        if not os.path.isfile(self.private_key_path):
            raise ValueError(f"Private key file not found: {self.private_key_path}")

        # Load private key
        with open(self.private_key_path, "rb") as f:
            self.private_key = serialization.load_pem_private_key(f.read(), password=None)

        # Set URLs based on environment
        if self.env == "prod":
            self.rest_base_url = "https://api.elections.kalshi.com/trade-api/v2"
            self.ws_url = "wss://api.elections.kalshi.com/trade-api/ws/v2"
        else:
            self.rest_base_url = "https://demo-api.kalshi.co/trade-api/v2"
            self.ws_url = "wss://demo-api.kalshi.co/trade-api/ws/v2"

    def get_series_tickers_list(self) -> list[str]:
        """Parse SPORTS_SERIES_TICKERS into a list."""
        if not self.sports_series_tickers:
            return []
        return [t.strip() for t in self.sports_series_tickers.split(",") if t.strip()]

    def get_tags_list(self) -> list[str]:
        """Parse SPORTS_TAGS into a list."""
        if not self.sports_tags:
            return []
        return [t.strip() for t in self.sports_tags.split(",") if t.strip()]


# =============================================================================
# Authentication / Signing
# =============================================================================

def sign_request(
    private_key,
    timestamp_ms: int,
    method: str,
    path: str
) -> str:
    """
    Sign a request using RSA-PSS with SHA256.

    The signature message format is: timestamp + method + path
    Returns base64-encoded signature.
    """
    message = f"{timestamp_ms}{method}{path}"
    message_bytes = message.encode("utf-8")

    signature = private_key.sign(
        message_bytes,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )

    return base64.b64encode(signature).decode("utf-8")


def get_auth_headers(config: KalshiConfig, method: str, path: str) -> dict[str, str]:
    """Generate authentication headers for a REST request."""
    timestamp_ms = int(time.time() * 1000)
    signature = sign_request(config.private_key, timestamp_ms, method, path)

    return {
        "KALSHI-ACCESS-KEY": config.key_id,
        "KALSHI-ACCESS-SIGNATURE": signature,
        "KALSHI-ACCESS-TIMESTAMP": str(timestamp_ms),
        "Content-Type": "application/json",
        "Accept": "application/json"
    }


def get_ws_auth_headers(config: KalshiConfig) -> dict[str, str]:
    """Generate authentication headers for WebSocket connection."""
    timestamp_ms = int(time.time() * 1000)
    # WebSocket signature uses GET method and the WS path
    ws_path = "/trade-api/ws/v2"
    signature = sign_request(config.private_key, timestamp_ms, "GET", ws_path)

    return {
        "KALSHI-ACCESS-KEY": config.key_id,
        "KALSHI-ACCESS-SIGNATURE": signature,
        "KALSHI-ACCESS-TIMESTAMP": str(timestamp_ms)
    }


# =============================================================================
# REST Client for Market Discovery
# =============================================================================

class KalshiRestClient:
    """REST client for Kalshi market discovery."""

    def __init__(self, config: KalshiConfig):
        self.config = config
        self.base_url = config.rest_base_url

    async def _request(
        self,
        session: aiohttp.ClientSession,
        method: str,
        endpoint: str,
        params: Optional[dict] = None
    ) -> dict:
        """Make an authenticated REST request."""
        # Build path with query string for signature
        path = f"/trade-api/v2{endpoint}"
        if params:
            query_string = urlencode(params)
            path_with_query = f"{path}?{query_string}"
        else:
            path_with_query = path

        headers = get_auth_headers(self.config, method, path_with_query)
        url = f"{self.base_url}{endpoint}"

        async with session.request(method, url, headers=headers, params=params) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f"REST request failed: {resp.status} - {text}")
            return await resp.json()

    async def get_sports_series(self, session: aiohttp.ClientSession) -> list[dict]:
        """
        Fetch series with category=sports.

        GET /series?category=sports[&tags=...]
        """
        params = {"category": "sports"}

        tags = self.config.get_tags_list()
        if tags:
            params["tags"] = ",".join(tags)

        data = await self._request(session, "GET", "/series", params)
        return data.get("series", [])

    async def get_markets_for_series(
        self,
        session: aiohttp.ClientSession,
        series_ticker: str,
        status: str = "open"
    ) -> list[dict]:
        """
        Fetch all open markets for a given series_ticker.
        Paginates through all results using cursor.

        GET /markets?series_ticker=...&status=open&limit=1000
        """
        all_markets = []
        cursor = None
        limit = 1000

        while True:
            params = {
                "series_ticker": series_ticker,
                "status": status,
                "limit": limit
            }
            if cursor:
                params["cursor"] = cursor

            data = await self._request(session, "GET", "/markets", params)
            markets = data.get("markets", [])
            all_markets.extend(markets)

            cursor = data.get("cursor")
            if not cursor or len(markets) < limit:
                break

        return all_markets

    async def discover_sports_market_tickers(self, session: aiohttp.ClientSession) -> list[str]:
        """
        Discover all sports market tickers via REST.

        1. If SPORTS_SERIES_TICKERS is set, use those directly.
        2. Otherwise, GET /series?category=sports to find series.
        3. For each series, paginate GET /markets to collect market tickers.
        """
        # Check if explicit series tickers provided
        explicit_series = self.config.get_series_tickers_list()

        if explicit_series:
            series_tickers = explicit_series
            print(f"[discovery] Using explicit series tickers: {series_tickers}", file=sys.stderr)
        else:
            # Discover series via REST
            series_list = await self.get_sports_series(session)
            series_tickers = [s.get("ticker") for s in series_list if s.get("ticker")]
            print(f"[discovery] Found {len(series_tickers)} sports series", file=sys.stderr)

        # Collect market tickers from all series
        market_tickers = []
        for series_ticker in series_tickers:
            try:
                markets = await self.get_markets_for_series(session, series_ticker)
                tickers = [m.get("ticker") for m in markets if m.get("ticker")]
                market_tickers.extend(tickers)
                print(f"[discovery] Series {series_ticker}: {len(tickers)} open markets", file=sys.stderr)
            except Exception as e:
                print(f"[discovery] Error fetching markets for {series_ticker}: {e}", file=sys.stderr)

        # Deduplicate
        market_tickers = list(set(market_tickers))
        print(f"[discovery] Total unique market tickers: {len(market_tickers)}", file=sys.stderr)

        return market_tickers


# =============================================================================
# WebSocket Consumer
# =============================================================================

class KalshiWebSocketConsumer:
    """
    WebSocket consumer for Kalshi market data.
    Subscribes to ticker, trade, and orderbook_delta channels for specified markets.
    """

    def __init__(self, config: KalshiConfig, market_tickers: list[str]):
        self.config = config
        self.market_tickers = market_tickers
        self.ws = None
        self.subscription_ids = {}
        self._running = False

    async def connect(self, session: aiohttp.ClientSession):
        """Establish authenticated WebSocket connection."""
        headers = get_ws_auth_headers(self.config)

        print(f"[ws] Connecting to {self.config.ws_url}", file=sys.stderr)
        self.ws = await session.ws_connect(
            self.config.ws_url,
            headers=headers,
            heartbeat=30.0
        )
        print("[ws] Connected successfully", file=sys.stderr)

    async def subscribe_to_channels(self):
        """
        Subscribe to ticker, trade, and orderbook_delta channels.
        Uses market_tickers to scope subscriptions to sports markets only.
        """
        if not self.ws:
            raise Exception("WebSocket not connected")

        if not self.market_tickers:
            print("[ws] No market tickers to subscribe to", file=sys.stderr)
            return

        # Channels to subscribe
        channels = ["ticker", "trade", "orderbook_delta"]

        # Subscribe command with market_tickers
        subscribe_cmd = {
            "id": 1,
            "cmd": "subscribe",
            "params": {
                "channels": channels,
                "market_tickers": self.market_tickers
            }
        }

        print(f"[ws] Subscribing to channels {channels} for {len(self.market_tickers)} markets", file=sys.stderr)
        await self.ws.send_json(subscribe_cmd)

    async def list_subscriptions(self):
        """Send list_subscriptions command for debugging."""
        if not self.ws:
            return

        list_cmd = {
            "id": 99,
            "cmd": "list_subscriptions",
            "params": {}
        }
        await self.ws.send_json(list_cmd)

    async def update_subscription(
        self,
        sid: int,
        market_tickers: list[str],
        action: str = "add_markets"
    ):
        """
        Update an existing subscription.

        action: "add_markets" or "delete_markets"
        """
        if not self.ws:
            return

        update_cmd = {
            "id": 2,
            "cmd": "update_subscription",
            "params": {
                "sids": [sid],
                "market_tickers": market_tickers,
                "action": action
            }
        }
        await self.ws.send_json(update_cmd)

    def _emit_jsonl(self, data: dict):
        """Emit a JSONL line to stdout."""
        # Add timestamp
        data["_received_at"] = int(time.time() * 1000)
        print(json.dumps(data, separators=(",", ":")))
        sys.stdout.flush()

    async def consume(self):
        """
        Main consumption loop.
        Reads messages from WebSocket and emits JSONL to stdout.
        """
        if not self.ws:
            raise Exception("WebSocket not connected")

        self._running = True
        print("[ws] Starting message consumption loop", file=sys.stderr)

        try:
            async for msg in self.ws:
                if not self._running:
                    break

                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        await self._handle_message(data)
                    except json.JSONDecodeError as e:
                        print(f"[ws] JSON decode error: {e}", file=sys.stderr)

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    print(f"[ws] WebSocket error: {self.ws.exception()}", file=sys.stderr)
                    break

                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    print("[ws] WebSocket closed", file=sys.stderr)
                    break

        except asyncio.CancelledError:
            print("[ws] Consumption cancelled", file=sys.stderr)
            raise
        except Exception as e:
            print(f"[ws] Consumption error: {e}", file=sys.stderr)
            raise

    async def _handle_message(self, data: dict):
        """
        Handle incoming WebSocket message.

        Message types:
        - subscribed: Subscription confirmation
        - ticker: Market ticker update
        - trade: Trade occurred
        - orderbook_snapshot: Initial orderbook state
        - orderbook_delta: Incremental orderbook update
        - error: Error message
        """
        msg_type = data.get("type")

        if msg_type == "subscribed":
            # Store subscription IDs
            sid = data.get("sid")
            channel = data.get("msg", {}).get("channel")
            if sid and channel:
                self.subscription_ids[channel] = sid
            print(f"[ws] Subscribed: sid={sid}, channel={channel}", file=sys.stderr)

        elif msg_type == "error":
            error_msg = data.get("msg", {})
            print(f"[ws] Error: {error_msg}", file=sys.stderr)

        elif msg_type in ("ticker", "trade", "orderbook_snapshot", "orderbook_delta"):
            # Emit as JSONL
            self._emit_jsonl(data)

        elif msg_type == "subscriptions":
            # Response to list_subscriptions
            print(f"[ws] Current subscriptions: {json.dumps(data)}", file=sys.stderr)

        else:
            # Unknown or other message types - still emit
            if msg_type:
                self._emit_jsonl(data)

    async def close(self):
        """Close WebSocket connection."""
        self._running = False
        if self.ws and not self.ws.closed:
            await self.ws.close()
            print("[ws] Connection closed", file=sys.stderr)


# =============================================================================
# Main Entry Point
# =============================================================================

async def main():
    """
    Main entry point.

    1. Load configuration from environment.
    2. Discover sports market tickers via REST.
    3. Connect to WebSocket and subscribe to channels.
    4. Stream JSONL output until interrupted.
    """
    print("[main] Starting Kalshi Sports-Only WebSocket Consumer", file=sys.stderr)

    # Load configuration
    try:
        config = KalshiConfig()
    except ValueError as e:
        print(f"[main] Configuration error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[main] Environment: {config.env}", file=sys.stderr)
    print(f"[main] REST URL: {config.rest_base_url}", file=sys.stderr)
    print(f"[main] WS URL: {config.ws_url}", file=sys.stderr)

    # Create aiohttp session
    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Discover sports market tickers
        rest_client = KalshiRestClient(config)
        try:
            market_tickers = await rest_client.discover_sports_market_tickers(session)
        except Exception as e:
            print(f"[main] Market discovery failed: {e}", file=sys.stderr)
            sys.exit(1)

        if not market_tickers:
            print("[main] No sports markets found. Exiting.", file=sys.stderr)
            sys.exit(0)

        # Create WebSocket consumer
        consumer = KalshiWebSocketConsumer(config, market_tickers)

        try:
            # Connect to WebSocket
            await consumer.connect(session)

            # Subscribe to channels
            await consumer.subscribe_to_channels()

            # Optionally list subscriptions for debugging
            await consumer.list_subscriptions()

            # Start consuming messages
            await consumer.consume()

        except KeyboardInterrupt:
            print("\n[main] Interrupted by user", file=sys.stderr)

        except Exception as e:
            print(f"[main] Error: {e}", file=sys.stderr)
            raise

        finally:
            await consumer.close()

    print("[main] Shutdown complete", file=sys.stderr)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[main] Exiting...", file=sys.stderr)
        sys.exit(0)
