"""
Kalshi WebSocket Consumer - First Integration Point

Minimal WebSocket client that:
1. Authenticates with Kalshi using RSA-PSS
2. Discovers sports markets via REST
3. Connects to WebSocket and subscribes
4. Logs incoming messages with sports filtering
5. Optionally routes messages to QuestDB/Redis via WebSocketProcessor

This is the foundation for real-time data ingestion.

Usage:
    # From arb-kalshi-sportsbook directory
    python -m app.connectors.kalshi.ws_consumer

    # With storage integration (requires Docker services)
    python -m app.connectors.kalshi.ws_consumer --with-storage

    # Dry run (no credentials needed)
    python -m app.connectors.kalshi.ws_consumer --dry-run

Environment Variables:
    KALSHI_ENV: "demo" or "prod" (default: demo)
    KALSHI_KEY_ID: API Key ID
    KALSHI_PRIVATE_KEY_PATH: Path to PEM file

Based on Kalshi WebSocket Docs:
    https://docs.kalshi.com/getting_started/quick_start_websockets
"""

import asyncio
import json
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiohttp

from app.connectors.kalshi.auth import KalshiAuth


# =============================================================================
# CONFIGURATION
# =============================================================================

# Sports series we care about (matches resolver.py)
SPORTS_SERIES = ["KXNFL", "KXNCAAF", "KXNBA", "KXNCAAB", "KXMLB", "KXNHL"]

# API URLs by environment
URLS = {
    "prod": {
        "rest": "https://api.elections.kalshi.com/trade-api/v2",
        "ws": "wss://api.elections.kalshi.com/trade-api/ws/v2",
    },
    "demo": {
        "rest": "https://demo-api.kalshi.co/trade-api/v2",
        "ws": "wss://demo-api.kalshi.co/trade-api/ws/v2",
    },
}


# =============================================================================
# STATISTICS TRACKING
# =============================================================================

@dataclass
class ConsumerStats:
    """Track WebSocket consumer statistics."""
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    markets_discovered: int = 0
    messages_received: int = 0
    ticker_messages: int = 0
    trade_messages: int = 0
    orderbook_messages: int = 0
    other_messages: int = 0
    errors: int = 0
    rejected_non_sports: int = 0

    def summary(self) -> str:
        uptime = (datetime.now(timezone.utc) - self.started_at).total_seconds()
        rate = self.messages_received / uptime if uptime > 0 else 0
        return (
            f"Uptime: {uptime:.1f}s | "
            f"Markets: {self.markets_discovered} | "
            f"Messages: {self.messages_received} ({rate:.1f}/s) | "
            f"Tickers: {self.ticker_messages} | "
            f"Trades: {self.trade_messages} | "
            f"Rejected: {self.rejected_non_sports}"
        )


# =============================================================================
# WEBSOCKET CONSUMER
# =============================================================================

class KalshiWebSocketConsumer:
    """
    Minimal Kalshi WebSocket consumer for sports markets.

    Flow:
    1. Initialize with auth and environment
    2. discover_markets() - GET /markets for each sports series
    3. connect() - Establish WebSocket connection
    4. subscribe() - Subscribe to discovered market tickers
    5. consume() - Process incoming messages

    Optional: Pass a WebSocketProcessor to route messages to QuestDB/Redis.
    """

    def __init__(
        self,
        auth: KalshiAuth,
        env: str = "demo",
        series_filter: list[str] = None,
        processor=None,  # Optional WebSocketProcessor for storage integration
    ):
        self.auth = auth
        self.env = env
        self.series_filter = series_filter or SPORTS_SERIES
        self.processor = processor  # WebSocketProcessor instance

        # URLs
        self.rest_url = URLS[env]["rest"]
        self.ws_url = URLS[env]["ws"]

        # State
        self.markets: dict[str, dict] = {}  # ticker -> market data
        self.stats = ConsumerStats()
        self._running = False
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._session: Optional[aiohttp.ClientSession] = None

    @classmethod
    def from_env(cls) -> "KalshiWebSocketConsumer":
        """Create consumer from environment variables."""
        auth = KalshiAuth.from_env()
        env = os.environ.get("KALSHI_ENV", "demo").lower()

        # Optional series filter
        series_env = os.environ.get("SPORTS_SERIES_TICKERS", "")
        series_filter = None
        if series_env:
            series_filter = [s.strip() for s in series_env.split(",") if s.strip()]

        return cls(auth=auth, env=env, series_filter=series_filter)

    # =========================================================================
    # SPORTS FILTERING
    # =========================================================================

    def _is_sports_ticker(self, ticker: str) -> bool:
        """
        Check if ticker belongs to a sports series.

        Ticker format: KXNFL-25JAN12-BUF
        Returns True only if prefix is in SPORTS_SERIES.
        """
        if not ticker:
            return False
        prefix = ticker.split("-")[0] if "-" in ticker else ticker
        return prefix in SPORTS_SERIES

    # =========================================================================
    # REST API - MARKET DISCOVERY
    # =========================================================================

    async def discover_markets(self) -> int:
        """
        Discover open sports markets via REST API.

        Returns:
            Number of markets discovered
        """
        print(f"\n[Discovery] Fetching sports markets from {self.rest_url}")

        for series in self.series_filter:
            try:
                markets = await self._get_markets_for_series(series)
                for m in markets:
                    ticker = m.get("ticker", "")
                    if ticker:
                        self.markets[ticker] = m
                print(f"  {series}: {len(markets)} open markets")
            except Exception as e:
                print(f"  {series}: ERROR - {e}")
                self.stats.errors += 1

        self.stats.markets_discovered = len(self.markets)
        print(f"\n[Discovery] Total: {self.stats.markets_discovered} markets")
        return self.stats.markets_discovered

    async def _get_markets_for_series(self, series_ticker: str) -> list[dict]:
        """Fetch markets for a specific series."""
        path = "/markets"
        params = {
            "series_ticker": series_ticker,
            "status": "open",
            "limit": 1000,
        }

        # Build URL with query params
        query = "&".join(f"{k}={v}" for k, v in params.items())
        full_url = f"{self.rest_url}{path}?{query}"

        # Get auth headers (path without query params)
        headers = self.auth.get_headers("GET", f"/trade-api/v2{path}")

        async with self._session.get(full_url, headers=headers) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f"HTTP {resp.status}: {text[:200]}")
            data = await resp.json()
            return data.get("markets", [])

    # =========================================================================
    # WEBSOCKET CONNECTION
    # =========================================================================

    async def connect(self) -> bool:
        """
        Establish WebSocket connection with authentication.

        Returns:
            True if connected successfully
        """
        print(f"\n[WebSocket] Connecting to {self.ws_url}")

        # Get auth headers for WebSocket handshake
        headers = self.auth.get_headers("GET", "/trade-api/ws/v2")

        try:
            self._ws = await self._session.ws_connect(
                self.ws_url,
                headers=headers,
                heartbeat=30.0,
                receive_timeout=60.0,
            )
            print("[WebSocket] Connected successfully")
            return True

        except Exception as e:
            print(f"[WebSocket] Connection failed: {e}")
            self.stats.errors += 1
            return False

    async def subscribe(self, channels: list[str] = None) -> bool:
        """
        Subscribe to channels for discovered markets.

        Args:
            channels: List of channels (default: ticker, trade)

        Returns:
            True if subscription sent
        """
        if not self._ws:
            print("[WebSocket] Not connected")
            return False

        if not self.markets:
            print("[WebSocket] No markets discovered")
            return False

        channels = channels or ["ticker", "trade"]
        tickers = list(self.markets.keys())

        # Kalshi subscription format
        subscribe_cmd = {
            "id": 1,
            "cmd": "subscribe",
            "params": {
                "channels": channels,
                "market_tickers": tickers,
            }
        }

        print(f"[WebSocket] Subscribing to {len(tickers)} markets, channels: {channels}")
        await self._ws.send_json(subscribe_cmd)
        return True

    # =========================================================================
    # MESSAGE PROCESSING
    # =========================================================================

    async def consume(self):
        """
        Main consumption loop - process incoming WebSocket messages.

        Runs until stopped or disconnected.
        """
        if not self._ws:
            print("[WebSocket] Not connected")
            return

        self._running = True
        print("\n[Consumer] Starting message consumption...")
        print("[Consumer] Press Ctrl+C to stop\n")

        try:
            async for msg in self._ws:
                if not self._running:
                    break

                if msg.type == aiohttp.WSMsgType.TEXT:
                    await self._handle_message(msg.data)

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    print(f"[WebSocket] Error: {self._ws.exception()}")
                    self.stats.errors += 1
                    break

                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    print("[WebSocket] Connection closed")
                    break

        except asyncio.CancelledError:
            print("\n[Consumer] Cancelled")
        except Exception as e:
            print(f"\n[Consumer] Error: {e}")
            self.stats.errors += 1

    async def _handle_message(self, raw: str):
        """Parse and process a WebSocket message."""
        self.stats.messages_received += 1

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            print(f"[Consumer] Invalid JSON: {raw[:100]}")
            self.stats.errors += 1
            return

        msg_type = data.get("type", "")
        msg = data.get("msg", {})

        # Extract ticker for sports filtering
        ticker = msg.get("market_ticker", "")

        # For data messages, validate sports ticker
        if msg_type in ("ticker", "trade", "orderbook_delta", "orderbook_snapshot"):
            if not self._is_sports_ticker(ticker):
                self.stats.rejected_non_sports += 1
                return

        # Route by message type
        if msg_type == "ticker":
            self._handle_ticker(msg)
        elif msg_type == "trade":
            self._handle_trade(msg)
        elif msg_type in ("orderbook_delta", "orderbook_snapshot"):
            self._handle_orderbook(msg, msg_type)
        elif msg_type == "subscribed":
            self._handle_subscribed(data)
        elif msg_type == "error":
            self._handle_error(data)
        else:
            self.stats.other_messages += 1

        # Print stats periodically
        if self.stats.messages_received % 100 == 0:
            print(f"[Stats] {self.stats.summary()}")

    def _handle_ticker(self, msg: dict):
        """Handle ticker (price update) message."""
        self.stats.ticker_messages += 1

        ticker = msg.get("market_ticker", "")
        yes_bid = msg.get("yes_bid", 0)
        yes_ask = msg.get("yes_ask", 0)
        volume = msg.get("volume", 0)
        ts = msg.get("ts", 0)

        # Format timestamp
        if ts:
            dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
            ts_str = dt.strftime("%H:%M:%S.%f")[:-3]
        else:
            ts_str = "N/A"

        print(f"[Ticker] {ticker}: bid={yes_bid}c ask={yes_ask}c vol={volume} @{ts_str}")

        # Update cached market data
        if ticker in self.markets:
            self.markets[ticker]["yes_bid"] = yes_bid
            self.markets[ticker]["yes_ask"] = yes_ask
            self.markets[ticker]["volume"] = volume

        # Route to processor for storage (QuestDB/Redis)
        if self.processor:
            self.processor.process_ticker(msg)

    def _handle_trade(self, msg: dict):
        """Handle trade execution message."""
        self.stats.trade_messages += 1

        ticker = msg.get("market_ticker", "")
        price = msg.get("yes_price") or msg.get("no_price") or msg.get("price", 0)
        count = msg.get("count", 0)
        side = msg.get("side", "")
        taker_side = msg.get("taker_side", "")
        ts = msg.get("ts", 0)

        if ts:
            dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
            ts_str = dt.strftime("%H:%M:%S.%f")[:-3]
        else:
            ts_str = "N/A"

        print(f"[Trade] {ticker}: {count}x{side}@{price}c taker={taker_side} @{ts_str}")

        # Route to processor for storage (QuestDB/Redis)
        if self.processor:
            self.processor.process_trade(msg)

    def _handle_orderbook(self, msg: dict, msg_type: str):
        """Handle orderbook update message."""
        self.stats.orderbook_messages += 1

        ticker = msg.get("market_ticker", "")
        # Just count for now - full orderbook tracking would go to QuestDB
        print(f"[Orderbook] {ticker}: {msg_type}")

    def _handle_subscribed(self, data: dict):
        """Handle subscription confirmation."""
        print(f"[Subscribed] {data}")

    def _handle_error(self, data: dict):
        """Handle error message."""
        self.stats.errors += 1
        print(f"[Error] {data}")

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    async def run(self):
        """
        Main entry point - full lifecycle.

        1. Create HTTP session
        2. Discover markets
        3. Connect WebSocket
        4. Subscribe
        5. Consume until stopped
        """
        print("=" * 60)
        print("KALSHI WEBSOCKET CONSUMER")
        print("=" * 60)
        print(f"Environment: {self.env}")
        print(f"REST URL: {self.rest_url}")
        print(f"WebSocket URL: {self.ws_url}")
        print(f"Series Filter: {self.series_filter}")

        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            self._session = session

            # Step 1: Discover markets
            count = await self.discover_markets()
            if count == 0:
                print("\n[Consumer] No markets found. Exiting.")
                return

            # Step 2: Connect WebSocket
            if not await self.connect():
                print("\n[Consumer] Connection failed. Exiting.")
                return

            # Step 3: Subscribe
            if not await self.subscribe(channels=["ticker", "trade"]):
                print("\n[Consumer] Subscription failed. Exiting.")
                return

            # Step 4: Consume
            await self.consume()

        print(f"\n[Consumer] Final Stats: {self.stats.summary()}")

    def stop(self):
        """Signal consumer to stop."""
        self._running = False
        if self._ws:
            asyncio.create_task(self._ws.close())


# =============================================================================
# DRY RUN MODE (Simulated Data)
# =============================================================================

async def dry_run():
    """
    Dry run with simulated Kalshi WebSocket messages.

    Tests message parsing and filtering logic without actual API connection.
    """
    print("=" * 60)
    print("DRY RUN - Simulated WebSocket Messages")
    print("=" * 60)
    print()

    # Simulated messages based on Kalshi WebSocket docs
    test_messages = [
        # Sports ticker (should process)
        {
            "type": "ticker",
            "msg": {
                "market_ticker": "KXNFL-26JAN12-BUF",
                "yes_bid": 45,
                "yes_ask": 47,
                "no_bid": 53,
                "no_ask": 55,
                "volume": 12500,
                "ts": int(time.time() * 1000)
            }
        },
        # Sports trade (should process)
        {
            "type": "trade",
            "msg": {
                "market_ticker": "KXNBA-26JAN12-LAL",
                "yes_price": 52,
                "count": 10,
                "side": "yes",
                "taker_side": "yes",
                "ts": int(time.time() * 1000)
            }
        },
        # Non-sports ticker (should reject - politics)
        {
            "type": "ticker",
            "msg": {
                "market_ticker": "PRES-2028-DEM",
                "yes_bid": 30,
                "yes_ask": 32,
                "volume": 50000,
                "ts": int(time.time() * 1000)
            }
        },
        # Subscription confirmation
        {
            "type": "subscribed",
            "channel": "ticker",
            "market_tickers": ["KXNFL-26JAN12-BUF"]
        },
        # Error message
        {
            "type": "error",
            "code": "invalid_ticker",
            "message": "Market not found"
        },
        # NBA ticker
        {
            "type": "ticker",
            "msg": {
                "market_ticker": "KXNBA-26JAN15-GSW",
                "yes_bid": 60,
                "yes_ask": 62,
                "no_bid": 38,
                "no_ask": 40,
                "volume": 8500,
                "ts": int(time.time() * 1000)
            }
        },
        # MLB trade
        {
            "type": "trade",
            "msg": {
                "market_ticker": "KXMLB-26APR10-NYY",
                "yes_price": 55,
                "count": 25,
                "side": "yes",
                "taker_side": "no",
                "ts": int(time.time() * 1000)
            }
        },
    ]

    # Create consumer with mock state
    class MockConsumer(KalshiWebSocketConsumer):
        def __init__(self, processor=None):
            self.stats = ConsumerStats()
            self.processor = processor
            self.markets = {
                "KXNFL-26JAN12-BUF": {"ticker": "KXNFL-26JAN12-BUF"},
                "KXNBA-26JAN12-LAL": {"ticker": "KXNBA-26JAN12-LAL"},
            }

    # Check if storage integration requested
    processor = None
    if "--with-storage" in sys.argv:
        print("Storage integration enabled (dry run mode - no actual writes)")
        from app.connectors.kalshi.ws_processor import WebSocketProcessor
        processor = WebSocketProcessor(write_questdb=False, write_redis=False)

    # Check if detector integration requested
    detector = None
    if "--with-detector" in sys.argv:
        print("Detector integration enabled")
        from app.services.realtime_detector import RealtimeDetector
        detector = RealtimeDetector.from_env()

        # Populate detector with mock market metadata
        mock_market_metadata = [
            {"ticker": "KXNFL-26JAN12-BUF", "title": "Buffalo Bills to beat Jacksonville Jaguars", "series_ticker": "KXNFL"},
            {"ticker": "KXNBA-26JAN12-LAL", "title": "Los Angeles Lakers to beat Golden State Warriors", "series_ticker": "KXNBA"},
            {"ticker": "KXNBA-26JAN15-GSW", "title": "Golden State Warriors to beat Phoenix Suns", "series_ticker": "KXNBA"},
            {"ticker": "KXMLB-26APR10-NYY", "title": "New York Yankees to beat Boston Red Sox", "series_ticker": "KXMLB"},
        ]
        detector.populate_markets(mock_market_metadata)
        detector.start()

        # Register detector with processor if both enabled
        if processor:
            processor.on_ticker(detector.on_market_update)
            print("  -> Detector registered with processor")

    # Check if take-profit integration requested
    take_profit_monitor = None
    if "--with-take-profit" in sys.argv:
        print("Take-profit monitor enabled")
        from app.execution.position_store import PositionStore, PositionSide
        from app.execution.take_profit import TakeProfitMonitor

        position_store = PositionStore(redis_url=None)  # No Redis in dry run
        take_profit_monitor = TakeProfitMonitor(position_store)

        # Register some mock positions to test take-profit
        take_profit_monitor.register_position(
            ticker="KXNFL-26JAN12-BUF",
            side=PositionSide.YES,
            contracts=25,
            entry_price=44,  # Entry at 44c, tick shows 45c bid
            entry_edge=5,
        )
        take_profit_monitor.register_position(
            ticker="KXNBA-26JAN15-GSW",
            side=PositionSide.YES,
            contracts=10,
            entry_price=58,  # Entry at 58c, tick shows 60c bid
            entry_edge=5,
        )

        # Track exit signals
        exit_signals = []
        def on_exit(signal):
            exit_signals.append(signal)
            print(f"  [EXIT SIGNAL] {signal.summary()}")

        take_profit_monitor.on_exit_signal(on_exit)

        # Register with processor
        if processor:
            processor.set_take_profit_monitor(take_profit_monitor)
            print("  -> Take-profit monitor registered with processor")

    consumer = MockConsumer(processor=processor)

    print("Processing simulated messages:\n")
    for i, msg in enumerate(test_messages, 1):
        print(f"--- Message {i} ---")
        raw = json.dumps(msg)
        await consumer._handle_message(raw)
        print()

    print("=" * 60)
    print(f"DRY RUN COMPLETE")
    print(f"Consumer Stats: {consumer.stats.summary()}")

    if processor:
        print(f"Processor Stats: {processor.stats.summary()}")

    if detector:
        print(f"Detector Stats: {detector.stats.summary()}")
        detector.stop()

    if take_profit_monitor:
        tp_stats = take_profit_monitor.get_stats()
        print(f"Take-Profit Stats:")
        print(f"  Price updates: {tp_stats['price_updates']}")
        print(f"  Positions checked: {tp_stats['positions_checked']}")
        print(f"  Take-profits: {tp_stats['take_profits_triggered']}")
        print(f"  Stop-losses: {tp_stats['stop_losses_triggered']}")

    print()
    print("Expected Results:")
    print("  - 4 sports messages processed (2 tickers, 2 trades)")
    print("  - 1 non-sports message rejected (PRES-2028-DEM)")
    print("  - 2 other messages (subscribed, error)")
    if take_profit_monitor:
        print("  - Take-profit positions monitored on ticker updates")


# =============================================================================
# MAIN
# =============================================================================

async def main():
    """Main entry point."""
    # Check for dry run mode
    if "--dry-run" in sys.argv or "-d" in sys.argv:
        await dry_run()
        return

    # Load environment from .env file
    env_file = Path(__file__).parent.parent.parent.parent / ".env"
    if env_file.exists():
        print(f"Loading environment from: {env_file}")
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())

    # Initialize processor if storage integration requested
    processor = None
    if "--with-storage" in sys.argv:
        print("[Main] Storage integration enabled")
        from app.connectors.kalshi.ws_processor import WebSocketProcessor
        processor = WebSocketProcessor.from_env()
        if not processor.connect():
            print("[Main] WARNING: Some storage backends failed to connect")

    # Initialize detector if detection integration requested
    realtime_detector = None
    if "--with-detector" in sys.argv:
        print("[Main] Realtime detector enabled")
        from app.services.realtime_detector import RealtimeDetector
        realtime_detector = RealtimeDetector.from_env()
        realtime_detector.start()

        # Register detector with processor if both enabled
        if processor:
            processor.on_ticker(realtime_detector.on_market_update)
            print("[Main] Detector registered with processor callbacks")

    # Initialize take-profit monitor if requested
    take_profit_monitor = None
    if "--with-take-profit" in sys.argv:
        print("[Main] Take-profit monitor enabled")
        from app.execution.position_store import PositionStore
        from app.execution.take_profit import TakeProfitMonitor

        position_store = PositionStore.from_env()
        loaded = position_store.load_from_redis()
        if loaded > 0:
            print(f"[Main] Loaded {loaded} positions from Redis")

        take_profit_monitor = TakeProfitMonitor.from_env(position_store)

        # Define exit handler
        def on_exit_signal(exit_signal):
            print(f"[EXIT SIGNAL] {exit_signal.summary()}")
            # TODO: Execute actual exit order via Kalshi API

        take_profit_monitor.on_exit_signal(on_exit_signal)

        # Register with processor
        if processor:
            processor.set_take_profit_monitor(take_profit_monitor)
            print("[Main] Take-profit monitor registered with processor")

    # Create and run consumer
    try:
        auth = KalshiAuth.from_env()
        env = os.environ.get("KALSHI_ENV", "demo").lower()
        consumer = KalshiWebSocketConsumer(auth=auth, env=env, processor=processor)
    except Exception as e:
        print(f"[Error] Failed to initialize consumer: {e}")
        print("\nMake sure these environment variables are set:")
        print("  KALSHI_KEY_ID - Your API key ID")
        print("  KALSHI_PRIVATE_KEY_PATH - Path to your private key PEM file")
        print("  KALSHI_ENV - 'demo' or 'prod' (default: demo)")
        if processor:
            processor.close()
        return

    # Setup shutdown handler
    def shutdown(signum, frame):
        print("\n[Consumer] Shutdown requested...")
        consumer.stop()
        if realtime_detector:
            realtime_detector.stop()
        if processor:
            processor.close()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        await consumer.run()
    finally:
        if realtime_detector:
            realtime_detector.stop()
        if processor:
            processor.close()


if __name__ == "__main__":
    asyncio.run(main())
