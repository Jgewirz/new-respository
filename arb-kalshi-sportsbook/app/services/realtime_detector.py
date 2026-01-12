"""
Real-Time Edge Detector - WebSocket Integration

Receives ticker updates from WebSocketProcessor and runs edge detection
on each market update instead of batch polling.

This is the bridge between real-time data and the detection algorithm:
    WebSocket -> Processor -> RealtimeDetector -> Signal -> Executor

Features:
1. Rate-limited detection (max N detections per ticker per second)
2. Market metadata caching (title, event_time for detector)
3. Integration with EventResolver for sportsbook mapping
4. Redis lookup for sportsbook consensus
5. CircuitBreaker pre-check before emitting signals

Usage:
    from app.services.realtime_detector import RealtimeDetector

    detector = RealtimeDetector.from_env()
    detector.start()

    # Register with WebSocket processor
    processor.on_ticker(detector.on_market_update)

    # Or manual call
    detector.on_market_update(ticker_msg)
"""

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

import redis

from app.arb.detector import (
    EdgeDetector,
    KalshiMarket,
    SportsbookConsensus,
    Signal,
    get_profile,
)
from app.mapping.resolver import EventResolver, MappedEvent, KALSHI_SERIES_TO_SPORT


# =============================================================================
# CONFIGURATION
# =============================================================================

# Rate limiting: minimum ms between detections per ticker
DEFAULT_MIN_DETECTION_INTERVAL_MS = 100  # Max 10 detections/second per ticker

# Redis keys for sportsbook consensus (match odds_ingest.py)
KEY_CONSENSUS = "odds:consensus:{event_id}:{team}"

# Default hours to event if unknown
DEFAULT_HOURS_TO_EVENT = 12.0


# =============================================================================
# STATISTICS
# =============================================================================

@dataclass
class DetectorStats:
    """Track realtime detector statistics."""
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updates_received: int = 0
    updates_rate_limited: int = 0
    detections_run: int = 0
    signals_generated: int = 0
    signals_tradeable: int = 0
    signals_blocked: int = 0
    resolver_misses: int = 0
    consensus_misses: int = 0
    errors: int = 0

    def summary(self) -> str:
        uptime = (datetime.now(timezone.utc) - self.started_at).total_seconds()
        rate = self.updates_received / uptime if uptime > 0 else 0
        return (
            f"Uptime: {uptime:.1f}s | "
            f"Updates: {self.updates_received} ({rate:.1f}/s) | "
            f"Detections: {self.detections_run} | "
            f"Signals: {self.signals_tradeable}/{self.signals_generated} tradeable | "
            f"Blocked: {self.signals_blocked}"
        )


# =============================================================================
# MARKET CACHE
# =============================================================================

@dataclass
class CachedMarket:
    """
    Cached market data combining REST metadata + live WebSocket prices.

    REST Discovery provides: title, event_time, series_ticker, status
    WebSocket provides: yes_bid, yes_ask, volume (real-time)
    """
    ticker: str
    title: str = ""
    series_ticker: str = ""
    status: str = "active"
    event_time: Optional[datetime] = None

    # Live prices from WebSocket
    yes_bid: int = 0
    yes_ask: int = 0
    no_bid: int = 0
    no_ask: int = 0
    volume: int = 0
    last_update: float = 0  # Unix timestamp ms

    def to_kalshi_market(self) -> KalshiMarket:
        """Convert to KalshiMarket for detector."""
        return KalshiMarket(
            ticker=self.ticker,
            title=self.title,
            yes_bid=self.yes_bid,
            yes_ask=self.yes_ask,
            no_bid=self.no_bid,
            no_ask=self.no_ask,
            volume=self.volume,
            status=self.status,
            event_time=self.event_time,
        )


# =============================================================================
# REALTIME DETECTOR
# =============================================================================

class RealtimeDetector:
    """
    Real-time edge detection triggered by WebSocket market updates.

    Flow:
    1. Receive ticker update from WebSocketProcessor
    2. Rate-limit check (skip if too frequent)
    3. Build KalshiMarket from cached data + live prices
    4. Resolve to sportsbook event via EventResolver
    5. Get sportsbook consensus from Redis
    6. Build SportsbookConsensus
    7. Run EdgeDetector.detect()
    8. Check CircuitBreaker (if configured)
    9. Fire callback with Signal

    Thread-safe: Uses per-ticker rate limiting with simple timestamp check.
    """

    def __init__(
        self,
        edge_detector: EdgeDetector,
        resolver: Optional[EventResolver] = None,
        redis_client: Optional[redis.Redis] = None,
        circuit_breaker=None,
        min_interval_ms: int = DEFAULT_MIN_DETECTION_INTERVAL_MS,
    ):
        """
        Initialize realtime detector.

        Args:
            edge_detector: Configured EdgeDetector instance
            resolver: EventResolver for Kalshi->sportsbook mapping
            redis_client: Redis for consensus lookups
            circuit_breaker: Optional CircuitBreaker for risk checks
            min_interval_ms: Minimum ms between detections per ticker
        """
        self.detector = edge_detector
        self.resolver = resolver
        self.redis = redis_client
        self.breaker = circuit_breaker
        self.min_interval_ms = min_interval_ms

        # Market metadata cache (populated from REST discovery)
        self._markets: dict[str, CachedMarket] = {}

        # Rate limiting: ticker -> last detection timestamp
        self._last_detection: dict[str, float] = {}

        # Callback for signals
        self._on_signal: Optional[Callable[[Signal], None]] = None

        # Stats
        self.stats = DetectorStats()

        # Running state
        self._running = False

    @classmethod
    def from_env(cls) -> "RealtimeDetector":
        """Create detector from environment variables."""
        # Load trading profile
        profile = os.environ.get("TRADING_PROFILE", "STANDARD")
        bankroll = float(os.environ.get("TRADING_BANKROLL", "10000"))

        edge_detector = EdgeDetector(
            profile=profile,
            bankroll=bankroll,
        )

        # Redis connection
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        try:
            redis_client = redis.from_url(redis_url, decode_responses=True)
            redis_client.ping()
        except Exception:
            redis_client = None

        # Rate limiting
        min_interval = int(os.environ.get("MIN_DETECTION_INTERVAL_MS", str(DEFAULT_MIN_DETECTION_INTERVAL_MS)))

        # EventResolver (without odds client - we use Redis for consensus)
        resolver = EventResolver(odds_client=None, redis_store=None)

        return cls(
            edge_detector=edge_detector,
            resolver=resolver,
            redis_client=redis_client,
            min_interval_ms=min_interval,
        )

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    def start(self):
        """Start the detector."""
        self._running = True
        print(f"[RealtimeDetector] Started with profile: {self.detector.profile_name}")
        print(f"[RealtimeDetector] Rate limit: {self.min_interval_ms}ms per ticker")

    def stop(self):
        """Stop the detector."""
        self._running = False
        print(f"[RealtimeDetector] Stopped. Stats: {self.stats.summary()}")

    # =========================================================================
    # CALLBACK REGISTRATION
    # =========================================================================

    def on_signal(self, callback: Callable[[Signal], None]):
        """Register callback for tradeable signals."""
        self._on_signal = callback

    # =========================================================================
    # MARKET CACHE MANAGEMENT
    # =========================================================================

    def populate_markets(self, markets: list[dict]):
        """
        Populate market cache from REST discovery.

        Args:
            markets: List of market dicts from Kalshi REST API
        """
        for m in markets:
            ticker = m.get("ticker", "")
            if not ticker:
                continue

            # Parse event time if present
            event_time = None
            if m.get("close_time"):
                try:
                    event_time = datetime.fromisoformat(m["close_time"].replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    pass

            self._markets[ticker] = CachedMarket(
                ticker=ticker,
                title=m.get("title", ""),
                series_ticker=m.get("series_ticker", ""),
                status=m.get("status", "active"),
                event_time=event_time,
            )

        print(f"[RealtimeDetector] Populated {len(self._markets)} markets")

    def get_cached_market(self, ticker: str) -> Optional[CachedMarket]:
        """Get cached market by ticker."""
        return self._markets.get(ticker)

    # =========================================================================
    # MAIN DETECTION FLOW
    # =========================================================================

    def on_market_update(self, msg: dict):
        """
        Handle market update from WebSocket.

        This is the main entry point called by WebSocketProcessor.

        Args:
            msg: Ticker message dict with market_ticker, yes_bid, yes_ask, etc.
        """
        if not self._running:
            return

        self.stats.updates_received += 1
        ticker = msg.get("market_ticker", "")

        if not ticker:
            return

        # 1. Rate limit check
        if not self._should_detect(ticker):
            self.stats.updates_rate_limited += 1
            return

        # 2. Update cached market with live prices
        cached = self._update_market_cache(ticker, msg)
        if not cached:
            return

        # 3. Run detection
        try:
            signal = self._run_detection(cached, msg)
            if signal:
                self._handle_signal(signal)
        except Exception as e:
            print(f"[RealtimeDetector] Detection error for {ticker}: {e}")
            self.stats.errors += 1

    def _should_detect(self, ticker: str) -> bool:
        """
        Check if we should run detection for this ticker.

        Rate limits to min_interval_ms between detections per ticker.
        """
        now = time.time() * 1000  # Current time in ms
        last = self._last_detection.get(ticker, 0)

        if now - last < self.min_interval_ms:
            return False

        self._last_detection[ticker] = now
        return True

    def _update_market_cache(self, ticker: str, msg: dict) -> Optional[CachedMarket]:
        """
        Update market cache with live prices from WebSocket.

        If market not in cache, create minimal entry.
        """
        if ticker not in self._markets:
            # Create minimal cache entry (missing title means resolver may fail)
            self._markets[ticker] = CachedMarket(
                ticker=ticker,
                title=msg.get("title", ""),  # Usually not in ticker msg
            )

        cached = self._markets[ticker]

        # Update live prices
        cached.yes_bid = msg.get("yes_bid", cached.yes_bid)
        cached.yes_ask = msg.get("yes_ask", cached.yes_ask)
        cached.no_bid = msg.get("no_bid", cached.no_bid)
        cached.no_ask = msg.get("no_ask", cached.no_ask)
        cached.volume = msg.get("volume", cached.volume)
        cached.last_update = msg.get("ts", time.time() * 1000)

        # Calculate no prices if not provided
        if cached.no_bid == 0 and cached.yes_ask > 0:
            cached.no_bid = 100 - cached.yes_ask
        if cached.no_ask == 0 and cached.yes_bid > 0:
            cached.no_ask = 100 - cached.yes_bid

        return cached

    def _run_detection(self, cached: CachedMarket, msg: dict) -> Optional[Signal]:
        """
        Run edge detection for a market.

        Returns Signal if detection completed, None otherwise.
        """
        self.stats.detections_run += 1

        # Convert to KalshiMarket
        kalshi = cached.to_kalshi_market()

        # If no title, we can't resolve to sportsbook
        if not kalshi.title:
            # Try to use ticker pattern to infer (limited capability)
            return None

        # Resolve to sportsbook event
        mapped = None
        if self.resolver:
            mapped = self.resolver.resolve(kalshi)

        if not mapped:
            self.stats.resolver_misses += 1
            return None

        # Build SportsbookConsensus from mapped event + Redis lookup
        consensus = self._build_consensus(mapped)
        if not consensus:
            self.stats.consensus_misses += 1
            return None

        # Calculate hours to event
        hours_to_event = DEFAULT_HOURS_TO_EVENT
        if cached.event_time:
            delta = cached.event_time - datetime.now(timezone.utc)
            hours_to_event = max(0, delta.total_seconds() / 3600)

        # Run detection
        signal = self.detector.detect(kalshi, consensus, hours_to_event)
        self.stats.signals_generated += 1

        return signal

    def _build_consensus(self, mapped: MappedEvent) -> Optional[SportsbookConsensus]:
        """
        Build SportsbookConsensus from mapped event.

        Queries Redis for consensus probability and book details.
        """
        if not mapped.consensus_prob:
            # Try Redis lookup if we have redis client
            if self.redis:
                try:
                    consensus = self._get_consensus_from_redis(
                        mapped.sportsbook_event_id,
                        mapped.target_team_canonical,
                    )
                    if consensus:
                        mapped.consensus_prob = consensus
                except Exception:
                    pass

        if not mapped.consensus_prob:
            return None

        # Build consensus object
        # Note: In real usage, we'd also get book_probs from Redis
        # For now, use simplified consensus
        return SportsbookConsensus(
            event_id=mapped.sportsbook_event_id,
            team=mapped.target_team,
            sport=mapped.sport,
            consensus_prob=mapped.consensus_prob,
            book_probs={"consensus": mapped.consensus_prob},  # Simplified
            book_count=3,  # Assume 3 books for now
            book_spread=2.0,  # Assume tight spread
            updated_at=datetime.now(timezone.utc),
        )

    def _get_consensus_from_redis(self, event_id: str, team: str) -> Optional[int]:
        """Get consensus probability from Redis."""
        if not self.redis:
            return None

        # Normalize team name for Redis key
        team_key = team.lower().replace("_", " ").replace(" ", "_")

        key = KEY_CONSENSUS.format(event_id=event_id, team=team_key)
        value = self.redis.get(key)

        if value:
            try:
                return int(value)
            except (ValueError, TypeError):
                pass

        return None

    def _handle_signal(self, signal: Signal):
        """Handle a generated signal."""
        if signal.should_trade:
            self.stats.signals_tradeable += 1

            # Check circuit breaker if configured
            if self.breaker:
                risk_cents = int(signal.risk_amount * 100)
                check = self.breaker.check_trade(
                    ticker=signal.kalshi.ticker,
                    contracts=signal.recommended_contracts,
                    risk_cents=risk_cents,
                )

                if not check.allowed:
                    self.stats.signals_blocked += 1
                    print(f"[RealtimeDetector] Signal blocked: {check.reason}")
                    return

            # Log tradeable signal
            print(
                f"[Signal] {signal.action} {signal.kalshi.ticker} | "
                f"Edge: {signal.edge.best_edge}c | "
                f"Conf: {signal.confidence_score} ({signal.confidence_tier}) | "
                f"Contracts: {signal.recommended_contracts}"
            )

            # Fire callback
            if self._on_signal:
                try:
                    self._on_signal(signal)
                except Exception as e:
                    print(f"[RealtimeDetector] Signal callback error: {e}")


# =============================================================================
# DRY RUN / VALIDATION
# =============================================================================

def dry_run():
    """
    Dry run to validate realtime detector integration.

    Tests detection flow with simulated market data.
    """
    print("=" * 60)
    print("REALTIME DETECTOR - DRY RUN VALIDATION")
    print("=" * 60)
    print()

    # Create detector with STANDARD profile
    detector = RealtimeDetector.from_env()
    detector.start()

    # Track signals
    signals_received = []

    def on_signal(signal: Signal):
        signals_received.append(signal)
        print(f"  [Callback] Received signal: {signal.action}")

    detector.on_signal(on_signal)

    # Populate with mock market data (simulating REST discovery)
    mock_markets = [
        {
            "ticker": "KXNFL-26JAN12-BUF",
            "title": "Buffalo Bills to beat Jacksonville Jaguars",
            "series_ticker": "KXNFL",
            "status": "active",
            "close_time": "2026-01-12T18:00:00Z",
        },
        {
            "ticker": "KXNBA-26JAN15-LAL",
            "title": "Los Angeles Lakers to beat Golden State Warriors",
            "series_ticker": "KXNBA",
            "status": "active",
            "close_time": "2026-01-15T20:00:00Z",
        },
    ]

    print("1. Populating market cache:")
    detector.populate_markets(mock_markets)
    print()

    # Simulate ticker updates
    mock_updates = [
        # NFL update with good edge (sportsbook consensus would be ~55%)
        {
            "market_ticker": "KXNFL-26JAN12-BUF",
            "yes_bid": 45,
            "yes_ask": 48,  # If consensus is 55%, edge = 7c
            "no_bid": 52,
            "no_ask": 55,
            "volume": 5000,
            "ts": int(time.time() * 1000),
        },
        # NBA update
        {
            "market_ticker": "KXNBA-26JAN15-LAL",
            "yes_bid": 58,
            "yes_ask": 61,
            "no_bid": 39,
            "no_ask": 42,
            "volume": 3000,
            "ts": int(time.time() * 1000),
        },
        # Rapid update (should be rate-limited)
        {
            "market_ticker": "KXNFL-26JAN12-BUF",
            "yes_bid": 46,
            "yes_ask": 49,
            "volume": 5100,
            "ts": int(time.time() * 1000),
        },
    ]

    print("2. Processing ticker updates:")
    print("-" * 40)
    for i, update in enumerate(mock_updates, 1):
        print(f"  Update {i}: {update['market_ticker']}")
        detector.on_market_update(update)

    print()
    print("3. Stats:")
    print(f"  {detector.stats.summary()}")
    print()

    print("4. Validation Results:")
    print(f"  - Updates received: {detector.stats.updates_received}")
    print(f"  - Rate limited: {detector.stats.updates_rate_limited}")
    print(f"  - Detections run: {detector.stats.detections_run}")
    print(f"  - Resolver misses: {detector.stats.resolver_misses} (expected - no Redis)")
    print(f"  - Signals generated: {detector.stats.signals_generated}")
    print()

    # Rate limiting test
    print("5. Rate Limiting Test:")
    print("-" * 40)
    detector._last_detection.clear()  # Reset rate limiting

    start = time.time()
    for i in range(20):
        detector.on_market_update(mock_updates[0])

    elapsed = time.time() - start
    print(f"  20 rapid updates processed in {elapsed*1000:.1f}ms")
    print(f"  Rate limited: {detector.stats.updates_rate_limited}")
    print()

    detector.stop()

    print("=" * 60)
    print("DRY RUN VALIDATION COMPLETE")
    print("=" * 60)
    print()
    print("Expected Behavior:")
    print("  - Market cache populated from REST discovery")
    print("  - Ticker updates trigger detection (with rate limiting)")
    print("  - Resolver misses expected without Redis consensus data")
    print("  - Rate limiting prevents excessive detections")


if __name__ == "__main__":
    dry_run()
