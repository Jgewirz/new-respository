"""
WebSocket Message Processor - Routes Messages to Storage Backends

Receives parsed WebSocket messages from ws_consumer and routes them to:
1. QuestDB via ILP (kalshi_ticks, kalshi_trades tables)
2. Redis hot cache (for real-time lookups)
3. Callback hooks (for detector integration)
4. TakeProfitMonitor (for real-time exit monitoring)

This is the integration layer between WebSocket data and the detection pipeline.

Usage:
    from app.connectors.kalshi.ws_processor import WebSocketProcessor

    processor = WebSocketProcessor.from_env()
    processor.connect()

    # In ws_consumer message handler:
    processor.process_ticker(msg)
    processor.process_trade(msg)

    processor.close()

    # With take-profit monitoring:
    from app.execution.take_profit import TakeProfitMonitor
    from app.execution.position_store import PositionStore

    store = PositionStore.from_env()
    monitor = TakeProfitMonitor(store)
    processor.set_take_profit_monitor(monitor)
"""

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

import redis

from app.data.questdb import QuestDBILPClient

# =============================================================================
# CONFIGURATION
# =============================================================================

# Sports series prefixes (matches ws_consumer.py)
SPORTS_SERIES = {"KXNFL", "KXNCAAF", "KXNBA", "KXNCAAB", "KXMLB", "KXNHL"}

# Map series prefix to sport name
SERIES_TO_SPORT = {
    "KXNFL": "nfl",
    "KXNCAAF": "ncaaf",
    "KXNBA": "nba",
    "KXNCAAB": "ncaab",
    "KXMLB": "mlb",
    "KXNHL": "nhl",
}

# Redis key patterns for Kalshi market data
KEY_KALSHI_MARKET = "kalshi:m:{ticker}"  # Hash: market data
KEY_KALSHI_MARKETS_SET = "kalshi:markets"  # Set: all market tickers
KEY_KALSHI_LAST_UPDATE = "kalshi:last_update"  # String: timestamp
REDIS_TTL = 3600  # 1 hour TTL


# =============================================================================
# STATISTICS
# =============================================================================

@dataclass
class ProcessorStats:
    """Track processor statistics."""
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ticks_processed: int = 0
    trades_processed: int = 0
    orderbook_processed: int = 0
    questdb_writes: int = 0
    questdb_errors: int = 0
    redis_writes: int = 0
    redis_errors: int = 0
    callbacks_fired: int = 0
    take_profit_checks: int = 0
    take_profit_exits: int = 0

    def summary(self) -> str:
        uptime = (datetime.now(timezone.utc) - self.started_at).total_seconds()
        total = self.ticks_processed + self.trades_processed
        rate = total / uptime if uptime > 0 else 0
        return (
            f"Uptime: {uptime:.1f}s | "
            f"Ticks: {self.ticks_processed} | "
            f"Trades: {self.trades_processed} | "
            f"QuestDB: {self.questdb_writes} | "
            f"Redis: {self.redis_writes} | "
            f"TP Exits: {self.take_profit_exits} | "
            f"Rate: {rate:.1f}/s"
        )


# =============================================================================
# WEBSOCKET PROCESSOR
# =============================================================================

class WebSocketProcessor:
    """
    Routes WebSocket messages to storage backends.

    Thread-safe for concurrent message processing.
    Uses the existing QuestDB and Redis patterns from the codebase.
    """

    def __init__(
        self,
        questdb_host: str = "localhost",
        questdb_port: int = 9009,
        redis_url: str = "redis://localhost:6379/0",
        write_questdb: bool = True,
        write_redis: bool = True,
    ):
        """
        Initialize processor.

        Args:
            questdb_host: QuestDB ILP host
            questdb_port: QuestDB ILP port
            redis_url: Redis connection URL
            write_questdb: Enable QuestDB writes
            write_redis: Enable Redis writes
        """
        self.questdb_host = questdb_host
        self.questdb_port = questdb_port
        self.redis_url = redis_url
        self.write_questdb = write_questdb
        self.write_redis = write_redis

        # Connections (lazy init)
        self._ilp: Optional[QuestDBILPClient] = None
        self._redis: Optional[redis.Redis] = None

        # Stats
        self.stats = ProcessorStats()

        # Callbacks
        self._on_ticker: Optional[Callable[[dict], None]] = None
        self._on_trade: Optional[Callable[[dict], None]] = None

        # Take-profit monitor (optional)
        self._take_profit_monitor = None

    @classmethod
    def from_env(cls) -> "WebSocketProcessor":
        """Create processor from environment variables."""
        return cls(
            questdb_host=os.environ.get("QUESTDB_ILP_HOST", "localhost"),
            questdb_port=int(os.environ.get("QUESTDB_ILP_PORT", "9009")),
            redis_url=os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
            write_questdb=os.environ.get("WRITE_QUESTDB", "true").lower() == "true",
            write_redis=os.environ.get("WRITE_REDIS", "true").lower() == "true",
        )

    # =========================================================================
    # CONNECTION MANAGEMENT
    # =========================================================================

    def connect(self) -> bool:
        """
        Establish connections to storage backends.

        Returns:
            True if all enabled backends connected successfully
        """
        success = True

        if self.write_questdb:
            try:
                self._ilp = QuestDBILPClient(self.questdb_host, self.questdb_port)
                self._ilp.connect()
                print(f"[Processor] QuestDB connected: {self.questdb_host}:{self.questdb_port}")
            except Exception as e:
                print(f"[Processor] QuestDB connection failed: {e}")
                self._ilp = None
                success = False

        if self.write_redis:
            try:
                self._redis = redis.from_url(self.redis_url, decode_responses=True)
                self._redis.ping()
                print(f"[Processor] Redis connected: {self.redis_url}")
            except Exception as e:
                print(f"[Processor] Redis connection failed: {e}")
                self._redis = None
                success = False

        return success

    def close(self):
        """Close all connections."""
        if self._ilp:
            try:
                self._ilp.close()
            except Exception:
                pass
            self._ilp = None

        if self._redis:
            try:
                self._redis.close()
            except Exception:
                pass
            self._redis = None

        print(f"[Processor] Closed. Final stats: {self.stats.summary()}")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()

    # =========================================================================
    # CALLBACK REGISTRATION
    # =========================================================================

    def on_ticker(self, callback: Callable[[dict], None]):
        """Register callback for ticker messages (for detector)."""
        self._on_ticker = callback

    def on_trade(self, callback: Callable[[dict], None]):
        """Register callback for trade messages."""
        self._on_trade = callback

    def set_take_profit_monitor(self, monitor):
        """
        Set take-profit monitor for real-time exit tracking.

        Args:
            monitor: TakeProfitMonitor instance
        """
        self._take_profit_monitor = monitor
        print("[Processor] Take-profit monitor registered")

    # =========================================================================
    # MESSAGE PROCESSING
    # =========================================================================

    def process_ticker(self, msg: dict) -> bool:
        """
        Process a ticker (price update) message.

        Args:
            msg: Parsed ticker message from WebSocket

        Returns:
            True if processed successfully
        """
        ticker = msg.get("market_ticker", "")
        if not ticker:
            return False

        # Extract series and sport
        series = self._extract_series(ticker)
        sport = SERIES_TO_SPORT.get(series, "")

        # Extract prices
        yes_bid = msg.get("yes_bid", 0)
        yes_ask = msg.get("yes_ask", 0)
        no_bid = msg.get("no_bid") or (100 - yes_ask if yes_ask else 0)
        no_ask = msg.get("no_ask") or (100 - yes_bid if yes_bid else 0)
        volume = msg.get("volume", 0)
        ts_ms = msg.get("ts", int(time.time() * 1000))
        timestamp_ns = ts_ms * 1_000_000  # Convert ms to ns

        self.stats.ticks_processed += 1

        # Write to QuestDB
        if self._ilp:
            try:
                self._ilp.write_tick(
                    ticker=ticker,
                    yes_bid=yes_bid,
                    yes_ask=yes_ask,
                    no_bid=no_bid,
                    no_ask=no_ask,
                    volume=volume,
                    series=series,
                    sport=sport,
                    timestamp_ns=timestamp_ns,
                )
                self.stats.questdb_writes += 1
            except Exception as e:
                print(f"[Processor] QuestDB write error: {e}")
                self.stats.questdb_errors += 1

        # Write to Redis
        if self._redis:
            try:
                self._write_ticker_to_redis(ticker, msg, series, sport)
                self.stats.redis_writes += 1
            except Exception as e:
                print(f"[Processor] Redis write error: {e}")
                self.stats.redis_errors += 1

        # Fire callback
        if self._on_ticker:
            try:
                self._on_ticker(msg)
                self.stats.callbacks_fired += 1
            except Exception as e:
                print(f"[Processor] Ticker callback error: {e}")

        # Check take-profit/stop-loss thresholds
        if self._take_profit_monitor:
            try:
                self.stats.take_profit_checks += 1
                exit_signal = self._take_profit_monitor.on_price_update(
                    ticker=ticker,
                    yes_bid=yes_bid,
                    no_bid=no_bid,
                )
                if exit_signal:
                    self.stats.take_profit_exits += 1
            except Exception as e:
                print(f"[Processor] Take-profit check error: {e}")

        return True

    def process_trade(self, msg: dict) -> bool:
        """
        Process a trade execution message.

        Args:
            msg: Parsed trade message from WebSocket

        Returns:
            True if processed successfully
        """
        ticker = msg.get("market_ticker", "")
        if not ticker:
            return False

        # Extract trade data
        price = msg.get("yes_price") or msg.get("no_price") or msg.get("price", 0)
        count = msg.get("count", 0)
        side = msg.get("side", "yes")
        taker_side = msg.get("taker_side", "")
        ts_ms = msg.get("ts", int(time.time() * 1000))
        timestamp_ns = ts_ms * 1_000_000

        self.stats.trades_processed += 1

        # Write to QuestDB
        if self._ilp:
            try:
                self._ilp.write_trade(
                    ticker=ticker,
                    price=price,
                    count=count,
                    side=side,
                    taker_side=taker_side,
                    timestamp_ns=timestamp_ns,
                )
                self.stats.questdb_writes += 1
            except Exception as e:
                print(f"[Processor] QuestDB trade write error: {e}")
                self.stats.questdb_errors += 1

        # Fire callback
        if self._on_trade:
            try:
                self._on_trade(msg)
                self.stats.callbacks_fired += 1
            except Exception as e:
                print(f"[Processor] Trade callback error: {e}")

        return True

    def process_orderbook(self, msg: dict, msg_type: str) -> bool:
        """
        Process orderbook delta/snapshot message.

        Args:
            msg: Parsed orderbook message
            msg_type: "orderbook_delta" or "orderbook_snapshot"

        Returns:
            True if processed successfully
        """
        self.stats.orderbook_processed += 1
        # Orderbook processing can be added later if needed
        return True

    # =========================================================================
    # REDIS HELPERS
    # =========================================================================

    def _write_ticker_to_redis(self, ticker: str, msg: dict, series: str, sport: str):
        """Write ticker data to Redis for quick lookups."""
        key = KEY_KALSHI_MARKET.format(ticker=ticker)

        data = {
            "ticker": ticker,
            "series": series,
            "sport": sport,
            "yes_bid": msg.get("yes_bid", 0),
            "yes_ask": msg.get("yes_ask", 0),
            "no_bid": msg.get("no_bid", 0),
            "no_ask": msg.get("no_ask", 0),
            "volume": msg.get("volume", 0),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "ts": msg.get("ts", 0),
        }

        pipe = self._redis.pipeline()
        pipe.hset(key, mapping=data)
        pipe.expire(key, REDIS_TTL)
        pipe.sadd(KEY_KALSHI_MARKETS_SET, ticker)
        pipe.set(KEY_KALSHI_LAST_UPDATE, datetime.now(timezone.utc).isoformat())
        pipe.execute()

    def get_market_from_redis(self, ticker: str) -> Optional[dict]:
        """Get market data from Redis cache."""
        if not self._redis:
            return None
        key = KEY_KALSHI_MARKET.format(ticker=ticker)
        return self._redis.hgetall(key) or None

    def get_all_markets_from_redis(self) -> list[str]:
        """Get all cached market tickers."""
        if not self._redis:
            return []
        return list(self._redis.smembers(KEY_KALSHI_MARKETS_SET))

    # =========================================================================
    # HELPERS
    # =========================================================================

    @staticmethod
    def _extract_series(ticker: str) -> str:
        """Extract series prefix from ticker (e.g., KXNFL from KXNFL-26JAN12-BUF)."""
        if "-" in ticker:
            return ticker.split("-")[0]
        return ticker


# =============================================================================
# DRY RUN / VALIDATION
# =============================================================================

def dry_run():
    """
    Dry run to validate ILP line format and Redis writes.

    Tests message processing without actual connections.
    """
    print("=" * 60)
    print("WEBSOCKET PROCESSOR - DRY RUN VALIDATION")
    print("=" * 60)
    print()

    # Test messages
    test_ticker = {
        "market_ticker": "KXNFL-26JAN12-BUF",
        "yes_bid": 45,
        "yes_ask": 47,
        "no_bid": 53,
        "no_ask": 55,
        "volume": 12500,
        "ts": 1736697600000,  # 2025-01-12 12:00:00 UTC
    }

    test_trade = {
        "market_ticker": "KXNBA-26JAN15-LAL",
        "yes_price": 52,
        "count": 10,
        "side": "yes",
        "taker_side": "yes",
        "ts": 1736697600000,
    }

    # Validate ILP line format for ticker
    print("1. Validating ILP Ticker Format:")
    print("-" * 40)
    ticker = test_ticker["market_ticker"]
    series = WebSocketProcessor._extract_series(ticker)
    sport = SERIES_TO_SPORT.get(series, "")
    ts_ns = test_ticker["ts"] * 1_000_000

    ilp_line = (
        f"kalshi_ticks,ticker={ticker}"
        f",series={series}"
        f",sport={sport}"
        f" yes_bid={test_ticker['yes_bid']}i"
        f",yes_ask={test_ticker['yes_ask']}i"
        f",no_bid={test_ticker['no_bid']}i"
        f",no_ask={test_ticker['no_ask']}i"
        f",volume={test_ticker['volume']}i"
        f" {ts_ns}"
    )
    print(f"ILP Line:\n  {ilp_line}")
    print()

    # Validate ILP line format for trade
    print("2. Validating ILP Trade Format:")
    print("-" * 40)
    ticker = test_trade["market_ticker"]
    ts_ns = test_trade["ts"] * 1_000_000

    ilp_line = (
        f"kalshi_trades,ticker={ticker}"
        f",side={test_trade['side']}"
        f",taker_side={test_trade['taker_side']}"
        f" price={test_trade['yes_price']}i"
        f",count={test_trade['count']}i"
        f" {ts_ns}"
    )
    print(f"ILP Line:\n  {ilp_line}")
    print()

    # Validate Redis key format
    print("3. Validating Redis Key Format:")
    print("-" * 40)
    redis_key = KEY_KALSHI_MARKET.format(ticker="KXNFL-26JAN12-BUF")
    print(f"Market Key: {redis_key}")
    print(f"Markets Set: {KEY_KALSHI_MARKETS_SET}")
    print(f"Last Update: {KEY_KALSHI_LAST_UPDATE}")
    print()

    # Test series extraction
    print("4. Validating Series Extraction:")
    print("-" * 40)
    test_tickers = [
        "KXNFL-26JAN12-BUF",
        "KXNBA-26JAN15-LAL",
        "KXMLB-26APR10-NYY",
        "KXNCAAB-26MAR20-DUKE",
    ]
    for t in test_tickers:
        series = WebSocketProcessor._extract_series(t)
        sport = SERIES_TO_SPORT.get(series, "unknown")
        print(f"  {t} -> series={series}, sport={sport}")
    print()

    # Test full processing with mock processor
    print("5. Testing Full Processing Flow:")
    print("-" * 40)

    callback_fired = {"ticker": 0, "trade": 0}

    def on_ticker(msg):
        callback_fired["ticker"] += 1
        print(f"  [Callback] Ticker: {msg['market_ticker']}")

    def on_trade(msg):
        callback_fired["trade"] += 1
        print(f"  [Callback] Trade: {msg['market_ticker']}")

    # Create processor without connections
    processor = WebSocketProcessor(write_questdb=False, write_redis=False)
    processor.on_ticker(on_ticker)
    processor.on_trade(on_trade)

    # Process test messages
    processor.process_ticker(test_ticker)
    processor.process_trade(test_trade)

    print()
    print(f"Stats: {processor.stats.summary()}")
    print(f"Callbacks fired: ticker={callback_fired['ticker']}, trade={callback_fired['trade']}")
    print()

    print("=" * 60)
    print("DRY RUN VALIDATION COMPLETE")
    print("=" * 60)
    print()
    print("Expected Results:")
    print("  - ILP lines match QuestDB schema (kalshi_ticks, kalshi_trades)")
    print("  - Redis keys follow naming convention (kalshi:m:{ticker})")
    print("  - Series extraction works for all sports prefixes")
    print("  - Callbacks fire correctly")


def test_with_connections():
    """Test with actual QuestDB and Redis connections."""
    print("=" * 60)
    print("WEBSOCKET PROCESSOR - CONNECTION TEST")
    print("=" * 60)
    print()

    # Load environment
    from pathlib import Path
    env_file = Path(__file__).parent.parent.parent.parent / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())

    processor = WebSocketProcessor.from_env()

    print("Connecting to storage backends...")
    if not processor.connect():
        print("WARNING: Some backends failed to connect")
        print("This is OK if you haven't started Docker services yet")
        print()

    # Test messages
    test_ticker = {
        "market_ticker": "KXNFL-26JAN12-TEST",
        "yes_bid": 45,
        "yes_ask": 47,
        "no_bid": 53,
        "no_ask": 55,
        "volume": 100,
        "ts": int(time.time() * 1000),
    }

    print("\nProcessing test ticker...")
    processor.process_ticker(test_ticker)

    print(f"\nStats: {processor.stats.summary()}")

    # Check Redis
    if processor._redis:
        market = processor.get_market_from_redis("KXNFL-26JAN12-TEST")
        if market:
            print(f"\nRedis market data: {market}")
        else:
            print("\nNo market data in Redis (write may have failed)")

    processor.close()


if __name__ == "__main__":
    import sys

    if "--test-connections" in sys.argv:
        test_with_connections()
    else:
        dry_run()
