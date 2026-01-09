# WebSocket Integration Plan

## Current State Analysis

### Existing WebSocket Consumer (`scripts/kalshi_ws_sports_only_consumer.py`)

The current implementation has a **disconnect** from the main data pipeline:

```
Current Flow (BROKEN for real-time):

WebSocket -> JSONL to stdout -> manual pipe -> jsonl_ingest.py -> QuestDB
                                    ^
                                    |
                              HUMAN INTERVENTION REQUIRED
```

**Problems:**
1. No direct integration with QuestDB - data goes to stdout
2. No Redis cache updates - arb pipeline can't see real-time data
3. No connection to execution layer - can't act on opportunities
4. Batch-oriented design instead of streaming

### Target Architecture

```
Target Flow (INTEGRATED):

                                    +---> QuestDB ILP (kalshi_ticks, kalshi_trades)
                                    |
Kalshi WebSocket --> Processor -----+---> Redis Cache (hot market data)
                                    |
                                    +---> ArbPipeline --> Execution
```

---

## Sports Market Filtering Strategy (CRITICAL)

### Ingestion Prerogative: Sports Markets ONLY

**This system MUST only ingest sports markets from Kalshi.** Kalshi has thousands of markets across categories (politics, economics, weather, crypto, etc.) but we only want sports betting markets for arbitrage detection against traditional sportsbooks.

### Supported Sports Series

The system filters using the `KALSHI_SERIES_TO_SPORT` mapping defined in `app/mapping/resolver.py`:

| Kalshi Series | Odds API Sport Key | Sport |
|---------------|-------------------|-------|
| `KXNFL` | `americanfootball_nfl` | NFL Football |
| `KXNCAAF` | `americanfootball_ncaaf` | College Football |
| `KXNBA` | `basketball_nba` | NBA Basketball |
| `KXNCAAB` | `basketball_ncaab` | College Basketball |
| `KXMLB` | `baseball_mlb` | MLB Baseball |
| `KXNHL` | `icehockey_nhl` | NHL Hockey |

**Any market not in these series is REJECTED at ingestion time.**

### Filtering Implementation (Two-Phase Approach)

#### Phase 1: REST Discovery Filter (Before WebSocket Connection)

```python
# CORRECT: Only fetch markets from sports series
SPORTS_SERIES = ["KXNFL", "KXNBA", "KXMLB", "KXNHL", "KXNCAAF", "KXNCAAB"]

for series in SPORTS_SERIES:
    # GET /markets?series_ticker=KXNFL&status=open
    markets = await client.get_markets(series_ticker=series, status="open")
    # Only sports markets returned - no politics, economics, etc.
```

#### Phase 2: WebSocket Subscription Filter

```python
# Subscribe ONLY to discovered sports market tickers
subscribe_cmd = {
    "id": 1,
    "cmd": "subscribe",
    "params": {
        "channels": ["ticker", "trade", "orderbook_delta"],
        "market_tickers": sports_market_tickers  # Pre-filtered list
    }
}
```

#### Phase 3: Runtime Validation (Belt & Suspenders)

Even after filtering at discovery, validate every incoming message:

```python
def _is_sports_ticker(self, ticker: str) -> bool:
    """
    Validate ticker belongs to a sports series.

    Ticker format: KXNFL-25JAN12-BUF
    Sports prefixes: KXNFL, KXNBA, KXMLB, KXNHL, KXNCAAF, KXNCAAB
    """
    if not ticker:
        return False

    # Extract series prefix (everything before first hyphen or full ticker)
    prefix = ticker.split("-")[0] if "-" in ticker else ticker

    # Check against whitelist
    return prefix in SPORTS_SERIES_PREFIXES

def process_message(self, data: dict):
    """Route message with sports validation."""
    msg = data.get("msg", {})
    ticker = msg.get("market_ticker", "")

    # REJECT non-sports tickers
    if not self._is_sports_ticker(ticker):
        self.stats.rejected_non_sports += 1
        return  # Silently drop

    # Process sports market...
```

### API Endpoints for Sports Filtering

| Endpoint | Purpose | Parameters |
|----------|---------|------------|
| `GET /series` | List all series | `category=sports` |
| `GET /markets` | List markets | `series_ticker=KXNFL`, `status=open` |
| `GET /search/filters_by_sport` | Get sport-specific filters | - |

### Configuration Options

```bash
# Environment Variables for Sports Filtering
# ==========================================

# Explicit series list (overrides auto-discovery)
SPORTS_SERIES_TICKERS=KXNFL,KXNBA,KXMLB,KXNHL

# Auto-discover all sports series (alternative)
SPORTS_CATEGORY_FILTER=sports

# Only open markets (recommended)
MARKET_STATUS_FILTER=open

# Optional: Tag-based filtering for specific sports
SPORTS_TAGS=nfl,nba,mlb,nhl
```

### Validation Metrics

Track filtering effectiveness:

```python
@dataclass
class FilteringStats:
    """Track sports filtering metrics."""
    markets_discovered: int = 0      # From REST discovery
    markets_cached: int = 0          # Added to cache
    messages_received: int = 0       # From WebSocket
    messages_processed: int = 0      # Sports markets processed
    rejected_non_sports: int = 0     # Non-sports rejected
    rejected_unknown: int = 0        # Unknown tickers rejected
```

### Why Strict Filtering Matters

1. **Data Quality**: Non-sports markets have no sportsbook equivalent for arbitrage
2. **Storage Efficiency**: QuestDB/Redis only store actionable data
3. **Detection Accuracy**: Edge detector only runs on mappable markets
4. **API Costs**: Reduce unnecessary processing and storage costs
5. **Performance**: Fewer markets = faster detection cycles

---

## Integration Gap Analysis

### Message Format Compatibility

**WebSocket Messages (from Kalshi docs):**
```json
{
    "type": "ticker",
    "msg": {
        "market_ticker": "KXNFL-25JAN12-BUF",
        "yes_bid": 45,
        "yes_ask": 47,
        "no_bid": 53,
        "no_ask": 55,
        "volume": 12500,
        "ts": 1704672000000
    }
}
```

**QuestDB ILP Format (existing in questdb.py):**
```python
def write_tick(ticker, yes_bid, yes_ask, no_bid, no_ask, volume, series, sport, timestamp_ns):
    # Matches WebSocket data perfectly!
```

**Gap:** The WebSocket message has `ts` in milliseconds, ILP needs nanoseconds. Simple conversion: `ts * 1_000_000`.

### KalshiMarket Dataclass Compatibility

**JSONL KalshiMarket (jsonl_reader.py):**
```python
@dataclass
class KalshiMarket:
    market_ticker: str
    series_ticker: str      # NOT in WebSocket ticker message
    event_ticker: str       # NOT in WebSocket ticker message
    title: str              # NOT in WebSocket ticker message
    subtitle: str           # NOT in WebSocket ticker message
    status: str             # NOT in WebSocket ticker message
    yes_bid: int
    yes_ask: int
    no_bid: int
    no_ask: int
    ...
```

**WebSocket Ticker Message:**
- Has: `market_ticker`, `yes_bid`, `yes_ask`, `no_bid`, `no_ask`, `volume`, `ts`
- Missing: `series_ticker`, `event_ticker`, `title`, `subtitle`, `status`

**Gap:** WebSocket ticker messages don't include metadata. Need to:
1. Pre-fetch market metadata via REST during discovery
2. Cache metadata by ticker
3. Enrich WebSocket messages with cached metadata

### Detector KalshiMarket (detector.py) Compatibility

```python
@dataclass
class KalshiMarket:  # For edge detection
    ticker: str
    title: str          # REQUIRED for team name parsing
    yes_bid: int
    yes_ask: int
    no_bid: int
    no_ask: int
    volume: int
    status: str
```

**Critical:** The `title` field is required for `EventResolver.resolve()` to parse the team name.

---

## Implementation Phases

### Phase 1: Market Metadata Cache

**Purpose:** Bridge the gap between WebSocket (no metadata) and detector (needs title).

**New File:** `app/connectors/kalshi/market_cache.py`

```python
"""
Market metadata cache for WebSocket enrichment.

WebSocket ticker messages only contain pricing data, not market metadata.
This cache stores metadata from REST API discovery for message enrichment.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional
import threading

@dataclass
class CachedMarket:
    """Market metadata cached from REST discovery."""
    ticker: str
    series_ticker: str
    event_ticker: str
    title: str
    subtitle: str
    status: str
    # Pricing (updated by WebSocket)
    yes_bid: int = 0
    yes_ask: int = 0
    no_bid: int = 0
    no_ask: int = 0
    volume: int = 0
    # Timestamps
    last_rest_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_ws_update: Optional[datetime] = None

class MarketMetadataCache:
    """
    Thread-safe cache for market metadata.

    Flow:
    1. REST discovery populates cache with full market data
    2. WebSocket updates prices in cache
    3. Detector reads enriched market data from cache
    """

    def __init__(self):
        self._cache: Dict[str, CachedMarket] = {}
        self._lock = threading.RLock()

    def populate_from_rest(self, markets: list[dict]) -> int:
        """
        Populate cache from REST API market discovery.
        Called during WebSocket initialization.

        Returns: Number of markets cached
        """
        with self._lock:
            for m in markets:
                ticker = m.get("ticker", "")
                if not ticker:
                    continue
                self._cache[ticker] = CachedMarket(
                    ticker=ticker,
                    series_ticker=m.get("series_ticker", ""),
                    event_ticker=m.get("event_ticker", ""),
                    title=m.get("title", ""),
                    subtitle=m.get("subtitle", ""),
                    status=m.get("status", "open"),
                    yes_bid=m.get("yes_bid", 0),
                    yes_ask=m.get("yes_ask", 0),
                    no_bid=m.get("no_bid", 0),
                    no_ask=m.get("no_ask", 0),
                    volume=m.get("volume", 0),
                )
            return len(self._cache)

    def update_from_ticker(self, msg: dict) -> Optional[CachedMarket]:
        """
        Update cache from WebSocket ticker message.
        Returns the enriched market or None if ticker unknown.
        """
        ticker = msg.get("market_ticker", "")
        if not ticker:
            return None

        with self._lock:
            cached = self._cache.get(ticker)
            if cached is None:
                return None  # Unknown ticker, skip

            # Update pricing
            cached.yes_bid = msg.get("yes_bid", cached.yes_bid)
            cached.yes_ask = msg.get("yes_ask", cached.yes_ask)
            cached.no_bid = msg.get("no_bid", cached.no_bid)
            cached.no_ask = msg.get("no_ask", cached.no_ask)
            cached.volume = msg.get("volume", cached.volume)
            cached.last_ws_update = datetime.now(timezone.utc)

            return cached

    def get(self, ticker: str) -> Optional[CachedMarket]:
        """Get cached market by ticker."""
        with self._lock:
            return self._cache.get(ticker)

    def get_all_active(self) -> list[CachedMarket]:
        """Get all active markets."""
        with self._lock:
            return [m for m in self._cache.values() if m.status == "open"]

    def to_detector_market(self, cached: CachedMarket):
        """Convert cached market to detector KalshiMarket."""
        from app.arb.detector import KalshiMarket
        return KalshiMarket(
            ticker=cached.ticker,
            title=cached.title,
            yes_bid=cached.yes_bid,
            yes_ask=cached.yes_ask,
            no_bid=cached.no_bid,
            no_ask=cached.no_ask,
            volume=cached.volume,
            status=cached.status,
        )
```

---

### Phase 2: WebSocket Message Processor

**Purpose:** Process WebSocket messages and route to appropriate storage.

**New File:** `app/connectors/kalshi/ws_processor.py`

```python
"""
WebSocket message processor for real-time data routing.

Processes incoming WebSocket messages and routes them to:
1. QuestDB ILP - Time-series storage
2. Redis Cache - Hot data for arb detection
3. Market Cache - In-memory enrichment
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Callable
import logging

from app.data.questdb import QuestDBILPClient
from app.connectors.kalshi.market_cache import MarketMetadataCache, CachedMarket

logger = logging.getLogger(__name__)

# Sports series whitelist (must match resolver.py)
SPORTS_SERIES_PREFIXES = {"KXNFL", "KXNCAAF", "KXNBA", "KXNCAAB", "KXMLB", "KXNHL"}

@dataclass
class ProcessorStats:
    """Statistics for monitoring."""
    ticker_messages: int = 0
    trade_messages: int = 0
    orderbook_messages: int = 0
    unknown_tickers: int = 0
    rejected_non_sports: int = 0  # Non-sports markets filtered out
    errors: int = 0
    ilp_writes: int = 0
    started_at: datetime = None

    def __post_init__(self):
        if self.started_at is None:
            self.started_at = datetime.now(timezone.utc)

class WebSocketProcessor:
    """
    Process WebSocket messages into storage layers.

    Message Types:
    - ticker: Price updates -> QuestDB kalshi_ticks, Redis cache
    - trade: Executions -> QuestDB kalshi_trades
    - orderbook_delta: Book changes -> QuestDB kalshi_orderbook
    """

    def __init__(
        self,
        market_cache: MarketMetadataCache,
        ilp_client: QuestDBILPClient = None,
        redis_url: str = None,
        on_market_update: Callable[[CachedMarket], None] = None,
    ):
        self.market_cache = market_cache
        self.ilp = ilp_client
        self.redis_url = redis_url
        self.on_market_update = on_market_update
        self.stats = ProcessorStats()
        self._redis = None

    def connect_storage(self):
        """Connect to storage backends."""
        if self.ilp is None:
            self.ilp = QuestDBILPClient()
            self.ilp.connect()

        if self.redis_url:
            import redis
            self._redis = redis.from_url(self.redis_url)

    def close(self):
        """Close storage connections."""
        if self.ilp:
            self.ilp.close()
        if self._redis:
            self._redis.close()

    def _is_sports_ticker(self, ticker: str) -> bool:
        """
        Validate ticker belongs to a sports series.

        Ticker format: KXNFL-25JAN12-BUF
        Returns True only if prefix is in SPORTS_SERIES_PREFIXES.
        """
        if not ticker:
            return False
        prefix = ticker.split("-")[0] if "-" in ticker else ticker
        return prefix in SPORTS_SERIES_PREFIXES

    def process_message(self, data: dict):
        """
        Route message to appropriate handler.

        IMPORTANT: Only processes sports market messages.
        Non-sports markets are rejected and counted in stats.

        Args:
            data: Parsed WebSocket message
        """
        msg_type = data.get("type")
        msg = data.get("msg", {})

        # Validate sports ticker for data messages
        if msg_type in ("ticker", "trade", "orderbook_snapshot", "orderbook_delta"):
            ticker = msg.get("market_ticker", "")
            if not self._is_sports_ticker(ticker):
                self.stats.rejected_non_sports += 1
                return  # Silently drop non-sports markets

        try:
            if msg_type == "ticker":
                self._process_ticker(msg)
            elif msg_type == "trade":
                self._process_trade(msg)
            elif msg_type in ("orderbook_snapshot", "orderbook_delta"):
                self._process_orderbook(msg, msg_type)
            elif msg_type == "subscribed":
                logger.info(f"Subscribed: {data}")
            elif msg_type == "error":
                logger.error(f"WebSocket error: {msg}")
                self.stats.errors += 1
        except Exception as e:
            logger.error(f"Error processing {msg_type}: {e}")
            self.stats.errors += 1

    def _process_ticker(self, msg: dict):
        """Process ticker (price update) message."""
        self.stats.ticker_messages += 1

        # Update market cache and get enriched market
        cached = self.market_cache.update_from_ticker(msg)
        if cached is None:
            self.stats.unknown_tickers += 1
            return

        # Convert timestamp (ms -> ns)
        ts_ms = msg.get("ts", 0)
        ts_ns = ts_ms * 1_000_000

        # Write to QuestDB
        if self.ilp:
            self.ilp.write_tick(
                ticker=cached.ticker,
                yes_bid=cached.yes_bid,
                yes_ask=cached.yes_ask,
                no_bid=cached.no_bid,
                no_ask=cached.no_ask,
                volume=cached.volume,
                series=cached.series_ticker,
                sport=self._extract_sport(cached.series_ticker),
                timestamp_ns=ts_ns,
            )
            self.stats.ilp_writes += 1

        # Update Redis hot cache
        if self._redis:
            key = f"kalshi:market:{cached.ticker}"
            self._redis.hset(key, mapping={
                "yes_bid": cached.yes_bid,
                "yes_ask": cached.yes_ask,
                "no_bid": cached.no_bid,
                "no_ask": cached.no_ask,
                "volume": cached.volume,
                "updated_at": ts_ms,
            })
            self._redis.expire(key, 3600)  # 1 hour TTL

        # Callback for real-time detection
        if self.on_market_update:
            self.on_market_update(cached)

    def _process_trade(self, msg: dict):
        """Process trade execution message."""
        self.stats.trade_messages += 1

        ticker = msg.get("market_ticker", "")
        if not ticker:
            return

        ts_ms = msg.get("ts", 0)
        ts_ns = ts_ms * 1_000_000

        if self.ilp:
            self.ilp.write_trade(
                ticker=ticker,
                price=msg.get("price", 0),
                count=msg.get("count", 0),
                side=msg.get("side", "yes"),
                taker_side=msg.get("taker_side", ""),
                timestamp_ns=ts_ns,
            )
            self.stats.ilp_writes += 1

    def _process_orderbook(self, msg: dict, msg_type: str):
        """Process orderbook delta/snapshot message."""
        self.stats.orderbook_messages += 1

        ticker = msg.get("market_ticker", "")
        if not ticker:
            return

        ts_ms = msg.get("ts", 0)
        ts_ns = ts_ms * 1_000_000

        # For deltas, write each price level change
        if msg_type == "orderbook_delta" and self.ilp:
            for side in ["yes", "no"]:
                levels = msg.get(side, [])
                for level in levels:
                    self.ilp.write_orderbook_delta(
                        ticker=ticker,
                        side=side,
                        price=level.get("price", 0),
                        delta=level.get("delta", 0),
                        timestamp_ns=ts_ns,
                    )
                    self.stats.ilp_writes += 1

    @staticmethod
    def _extract_sport(series_ticker: str) -> str:
        """Extract sport from series ticker (e.g., KXNFL -> nfl)."""
        if not series_ticker:
            return ""
        # KXNFL-25JAN12 -> nfl
        prefix = series_ticker.split("-")[0] if "-" in series_ticker else series_ticker
        if prefix.startswith("KX"):
            return prefix[2:].lower()
        return prefix.lower()

    def get_stats(self) -> dict:
        """Get processor statistics."""
        return {
            "ticker_messages": self.stats.ticker_messages,
            "trade_messages": self.stats.trade_messages,
            "orderbook_messages": self.stats.orderbook_messages,
            "unknown_tickers": self.stats.unknown_tickers,
            "rejected_non_sports": self.stats.rejected_non_sports,  # Filtering metric
            "ilp_writes": self.stats.ilp_writes,
            "errors": self.stats.errors,
            "uptime_seconds": (datetime.now(timezone.utc) - self.stats.started_at).total_seconds(),
        }
```

---

### Phase 3: Real-Time Arb Detection

**Purpose:** Trigger arb detection on each market update.

**New File:** `app/services/realtime_detector.py`

```python
"""
Real-time arbitrage detection service.

Triggers edge detection on each WebSocket market update.
Integrates with:
- MarketMetadataCache for enriched market data
- EventResolver for sportsbook mapping
- EdgeDetector for signal generation
- CircuitBreaker for risk controls
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Callable
import logging
import threading
from queue import Queue

from app.arb.detector import EdgeDetector, Signal, KalshiMarket, SportsbookConsensus
from app.mapping.resolver import EventResolver
from app.services.odds_ingest import OddsRedisStore
from app.execution import CircuitBreaker, ExecutionOrder, ExecutionMode
from app.connectors.kalshi.market_cache import CachedMarket

logger = logging.getLogger(__name__)

@dataclass
class DetectionStats:
    """Real-time detection statistics."""
    updates_received: int = 0
    detections_run: int = 0
    signals_generated: int = 0
    tradeable_signals: int = 0
    blocked_by_breaker: int = 0
    errors: int = 0

class RealtimeDetector:
    """
    Real-time arbitrage detector triggered by WebSocket updates.

    Flow:
    1. Receive market update from WebSocket processor
    2. Resolve to sportsbook consensus
    3. Run edge detection
    4. Check circuit breaker
    5. Emit signal for execution
    """

    def __init__(
        self,
        resolver: EventResolver,
        detector: EdgeDetector,
        breaker: CircuitBreaker,
        redis_store: OddsRedisStore,
        on_signal: Callable[[Signal], None] = None,
        min_detection_interval_ms: int = 100,  # Rate limit per ticker
    ):
        self.resolver = resolver
        self.detector = detector
        self.breaker = breaker
        self.redis_store = redis_store
        self.on_signal = on_signal
        self.min_interval_ms = min_detection_interval_ms

        self.stats = DetectionStats()
        self._last_detection: dict[str, float] = {}  # ticker -> timestamp
        self._lock = threading.Lock()

    def on_market_update(self, cached: CachedMarket):
        """
        Handle market update from WebSocket processor.

        Called by WebSocketProcessor.on_market_update callback.
        """
        self.stats.updates_received += 1

        try:
            # Rate limit per ticker
            if not self._should_detect(cached.ticker):
                return

            # Convert to detector format
            kalshi = KalshiMarket(
                ticker=cached.ticker,
                title=cached.title,
                yes_bid=cached.yes_bid,
                yes_ask=cached.yes_ask,
                no_bid=cached.no_bid,
                no_ask=cached.no_ask,
                volume=cached.volume,
                status=cached.status,
            )

            # Resolve to sportsbook
            # Note: resolver.resolve() expects a Market-like object
            class MarketProxy:
                def __init__(self, c):
                    self.ticker = c.ticker
                    self.title = c.title
                    self.yes_ask = c.yes_ask
                    self.no_ask = c.no_ask
                    self.yes_bid = c.yes_bid
                    self.no_bid = c.no_bid
                    self.volume = c.volume
                    self.status = c.status

            mapped = self.resolver.resolve(MarketProxy(cached))
            if mapped is None or mapped.consensus_prob is None:
                return

            # Build sportsbook consensus
            odds_data = self.redis_store.get_event_odds(
                mapped.sport,
                mapped.sportsbook_event_id,
                mapped.target_team_canonical,
            )

            if odds_data and "draftkings_prob" in odds_data:
                book_probs = {
                    "draftkings": int(float(odds_data.get("draftkings_prob", 0)) * 100),
                    "fanduel": int(float(odds_data.get("fanduel_prob", 0)) * 100),
                    "betmgm": int(float(odds_data.get("betmgm_prob", 0)) * 100),
                    "caesars": int(float(odds_data.get("caesars_prob", 0)) * 100),
                }
                book_probs = {k: v for k, v in book_probs.items() if v > 0}
                book_spread = max(book_probs.values()) - min(book_probs.values()) if book_probs else 0
            else:
                c = mapped.consensus_prob
                book_probs = {"draftkings": c, "fanduel": c, "betmgm": c, "caesars": c}
                book_spread = 0.0

            sportsbook = SportsbookConsensus(
                event_id=mapped.sportsbook_event_id,
                team=mapped.target_team,
                sport=mapped.sport,
                consensus_prob=mapped.consensus_prob,
                book_probs=book_probs,
                book_count=len(book_probs),
                book_spread=book_spread,
                updated_at=datetime.now(timezone.utc),
            )

            # Run detection
            self.stats.detections_run += 1
            signal = self.detector.detect(kalshi, sportsbook, hours_to_event=12)

            if signal.should_trade:
                self.stats.signals_generated += 1

                # Check circuit breaker
                check = self.breaker.check_trade(
                    ticker=signal.kalshi.ticker,
                    contracts=signal.recommended_contracts,
                    risk_cents=int(signal.risk_amount * 100),
                    sport=mapped.sport,
                )

                if check.allowed:
                    self.stats.tradeable_signals += 1
                    if self.on_signal:
                        self.on_signal(signal)
                else:
                    self.stats.blocked_by_breaker += 1
                    logger.info(f"Signal blocked by breaker: {check.reason}")

        except Exception as e:
            self.stats.errors += 1
            logger.error(f"Detection error for {cached.ticker}: {e}")

    def _should_detect(self, ticker: str) -> bool:
        """Check if enough time has passed since last detection for this ticker."""
        now = datetime.now(timezone.utc).timestamp() * 1000

        with self._lock:
            last = self._last_detection.get(ticker, 0)
            if now - last < self.min_interval_ms:
                return False
            self._last_detection[ticker] = now
            return True

    def get_stats(self) -> dict:
        """Get detection statistics."""
        return {
            "updates_received": self.stats.updates_received,
            "detections_run": self.stats.detections_run,
            "signals_generated": self.stats.signals_generated,
            "tradeable_signals": self.stats.tradeable_signals,
            "blocked_by_breaker": self.stats.blocked_by_breaker,
            "errors": self.stats.errors,
        }
```

---

### Phase 4: Integrated WebSocket Client

**Purpose:** Production-ready WebSocket client with full integration.

**New File:** `app/connectors/kalshi/ws_client.py`

```python
"""
Integrated Kalshi WebSocket Client.

Production-ready client that:
1. Discovers ONLY SPORTS markets via REST (filtered by series_ticker)
2. Caches market metadata for WebSocket message enrichment
3. Connects to WebSocket with RSA-PSS authentication
4. Routes messages through processor (with sports validation)
5. Triggers real-time arbitrage detection

IMPORTANT: This client ONLY ingests sports markets.
Non-sports markets (politics, economics, weather, etc.) are excluded.
"""

import asyncio
import logging
import os
import signal
import sys
from datetime import datetime, timezone
from typing import Optional

import aiohttp

from app.connectors.kalshi.auth import KalshiAuth
from app.connectors.kalshi.market_cache import MarketMetadataCache
from app.connectors.kalshi.ws_processor import WebSocketProcessor
from app.services.realtime_detector import RealtimeDetector
from app.arb.detector import EdgeDetector, Signal
from app.mapping.resolver import EventResolver, KALSHI_SERIES_TO_SPORT
from app.services.odds_ingest import OddsRedisStore
from app.execution import CircuitBreaker
from app.data.questdb import QuestDBILPClient

logger = logging.getLogger(__name__)

# Default sports series to subscribe to (matches resolver.py mapping)
DEFAULT_SPORTS_SERIES = list(KALSHI_SERIES_TO_SPORT.keys())
# ["KXNFL", "KXNCAAF", "KXNBA", "KXNCAAB", "KXMLB", "KXNHL"]

class IntegratedWebSocketClient:
    """
    Full-stack WebSocket client with integrated detection and execution.

    Usage:
        client = IntegratedWebSocketClient.from_env()
        await client.run()
    """

    def __init__(
        self,
        auth: KalshiAuth,
        rest_base_url: str,
        ws_url: str,
        market_cache: MarketMetadataCache,
        processor: WebSocketProcessor,
        detector: Optional[RealtimeDetector] = None,
        series_tickers: list[str] = None,
    ):
        self.auth = auth
        self.rest_base_url = rest_base_url
        self.ws_url = ws_url
        self.market_cache = market_cache
        self.processor = processor
        self.detector = detector
        # SPORTS FILTERING: Only these series will be discovered and subscribed
        self.series_tickers = series_tickers or DEFAULT_SPORTS_SERIES

        self.ws = None
        self._running = False
        self._session: Optional[aiohttp.ClientSession] = None

    @classmethod
    def from_env(
        cls,
        on_signal: callable = None,
    ) -> "IntegratedWebSocketClient":
        """
        Create client from environment variables.

        Environment:
            KALSHI_ENV: "demo" or "prod"
            KALSHI_KEY_ID: API key ID
            KALSHI_PRIVATE_KEY_PATH: Path to PEM file
            REDIS_URL: Redis connection URL
            QUESTDB_ILP_HOST: QuestDB host
            TRADING_PROFILE: Detection profile

        Sports Filtering (optional):
            SPORTS_SERIES_TICKERS: Comma-separated series (default: all sports)
                Example: "KXNFL,KXNBA" to only track NFL and NBA
        """
        env = os.getenv("KALSHI_ENV", "demo").lower()

        # Parse sports series from environment (or use all sports)
        series_env = os.getenv("SPORTS_SERIES_TICKERS", "")
        if series_env:
            series_tickers = [s.strip() for s in series_env.split(",") if s.strip()]
            logger.info(f"Sports filter: {series_tickers}")
        else:
            series_tickers = DEFAULT_SPORTS_SERIES
            logger.info(f"Sports filter: ALL ({series_tickers})")

        if env == "prod":
            rest_base_url = "https://api.elections.kalshi.com/trade-api/v2"
            ws_url = "wss://api.elections.kalshi.com/trade-api/ws/v2"
        else:
            rest_base_url = "https://demo-api.kalshi.co/trade-api/v2"
            ws_url = "wss://demo-api.kalshi.co/trade-api/ws/v2"

        # Authentication
        auth = KalshiAuth(
            key_id=os.environ["KALSHI_KEY_ID"],
            private_key_path=os.environ["KALSHI_PRIVATE_KEY_PATH"],
        )

        # Components
        market_cache = MarketMetadataCache()

        # QuestDB ILP client
        ilp = QuestDBILPClient(
            host=os.getenv("QUESTDB_ILP_HOST", "localhost"),
            port=int(os.getenv("QUESTDB_ILP_PORT", "9009")),
        )

        # Redis URL
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

        # Processor with callback for detection
        processor = WebSocketProcessor(
            market_cache=market_cache,
            ilp_client=ilp,
            redis_url=redis_url,
        )

        # Real-time detector (optional)
        detector = None
        if on_signal:
            redis_store = OddsRedisStore(redis_url)
            resolver = EventResolver(redis_store=redis_store)
            edge_detector = EdgeDetector(
                profile=os.getenv("TRADING_PROFILE", "STANDARD"),
                bankroll=float(os.getenv("TRADING_BANKROLL", "10000")),
            )
            breaker = CircuitBreaker.from_env()

            detector = RealtimeDetector(
                resolver=resolver,
                detector=edge_detector,
                breaker=breaker,
                redis_store=redis_store,
                on_signal=on_signal,
            )
            processor.on_market_update = detector.on_market_update

        return cls(
            auth=auth,
            rest_base_url=rest_base_url,
            ws_url=ws_url,
            market_cache=market_cache,
            processor=processor,
            detector=detector,
            series_tickers=series_tickers,  # Sports filtering
        )

    async def run(self):
        """
        Main entry point.

        1. Discover markets via REST
        2. Populate metadata cache
        3. Connect WebSocket
        4. Subscribe to channels
        5. Process messages until stopped
        """
        logger.info("Starting Integrated WebSocket Client")

        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            self._session = session

            # Phase 1: Discover markets
            logger.info("Discovering sports markets...")
            markets = await self._discover_markets()

            if not markets:
                logger.error("No markets found. Exiting.")
                return

            # Phase 2: Populate cache
            cached_count = self.market_cache.populate_from_rest(markets)
            logger.info(f"Cached {cached_count} markets")

            # Phase 3: Connect storage
            self.processor.connect_storage()

            # Phase 4: Connect WebSocket
            await self._connect_websocket()

            # Phase 5: Subscribe
            tickers = [m["ticker"] for m in markets]
            await self._subscribe(tickers)

            # Phase 6: Consume
            await self._consume()

    async def _discover_markets(self) -> list[dict]:
        """
        Discover SPORTS markets via REST API.

        IMPORTANT: Only queries series in self.series_tickers (sports only).
        Non-sports markets (politics, economics, etc.) are never fetched.
        """
        all_markets = []

        logger.info(f"Discovering markets for {len(self.series_tickers)} sports series...")

        for series in self.series_tickers:
            try:
                # GET /markets?series_ticker=KXNFL&status=open
                markets = await self._get_markets(series)
                all_markets.extend(markets)
                logger.info(f"  {series}: {len(markets)} open markets")
            except Exception as e:
                logger.error(f"  {series}: ERROR - {e}")

        logger.info(f"Total sports markets discovered: {len(all_markets)}")
        return all_markets

    async def _get_markets(self, series_ticker: str) -> list[dict]:
        """Fetch markets for a series from REST API."""
        path = "/markets"
        params = {
            "series_ticker": series_ticker,
            "status": "open",
            "limit": 1000,
        }

        headers = self.auth.get_headers("GET", f"/trade-api/v2{path}")
        url = f"{self.rest_base_url}{path}"

        async with self._session.get(url, headers=headers, params=params) as resp:
            if resp.status != 200:
                raise Exception(f"REST error: {resp.status}")
            data = await resp.json()
            return data.get("markets", [])

    async def _connect_websocket(self):
        """Connect to WebSocket with authentication."""
        headers = self.auth.get_ws_headers()

        logger.info(f"Connecting to {self.ws_url}")
        self.ws = await self._session.ws_connect(
            self.ws_url,
            headers=headers,
            heartbeat=30.0,
        )
        logger.info("WebSocket connected")

    async def _subscribe(self, tickers: list[str]):
        """Subscribe to channels for given tickers."""
        if not self.ws:
            return

        cmd = {
            "id": 1,
            "cmd": "subscribe",
            "params": {
                "channels": ["ticker", "trade", "orderbook_delta"],
                "market_tickers": tickers,
            }
        }

        logger.info(f"Subscribing to {len(tickers)} markets")
        await self.ws.send_json(cmd)

    async def _consume(self):
        """Main message consumption loop."""
        self._running = True
        logger.info("Starting message consumption")

        try:
            async for msg in self.ws:
                if not self._running:
                    break

                if msg.type == aiohttp.WSMsgType.TEXT:
                    import json
                    data = json.loads(msg.data)
                    self.processor.process_message(data)

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {self.ws.exception()}")
                    break

                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    logger.info("WebSocket closed")
                    break

        except asyncio.CancelledError:
            logger.info("Consumption cancelled")
        except Exception as e:
            logger.error(f"Consumption error: {e}")
        finally:
            self.processor.close()

    def stop(self):
        """Stop the client."""
        self._running = False
        if self.ws:
            asyncio.create_task(self.ws.close())

    def get_stats(self) -> dict:
        """Get combined statistics."""
        stats = {
            "processor": self.processor.get_stats(),
            "cache_size": len(self.market_cache.get_all_active()),
        }
        if self.detector:
            stats["detector"] = self.detector.get_stats()
        return stats
```

---

### Phase 5: Main Runner

**Purpose:** CLI entry point for the integrated system.

**New File:** `app/cli/run_realtime.py`

```python
"""
Real-time arbitrage detection runner.

Entry point for the integrated WebSocket-based detection system.

Usage:
    python -m app.cli.run_realtime

Environment:
    KALSHI_ENV=demo|prod
    KALSHI_KEY_ID=<key>
    KALSHI_PRIVATE_KEY_PATH=<path>
    REDIS_URL=redis://localhost:6379/0
    QUESTDB_ILP_HOST=localhost
    TRADING_PROFILE=STANDARD
    PAPER_TRADING=true
"""

import asyncio
import logging
import os
import signal
import sys
from datetime import datetime, timezone

from app.connectors.kalshi.ws_client import IntegratedWebSocketClient
from app.arb.detector import Signal
from app.execution import ExecutionOrder, ExecutionMode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Global client for signal handling
_client = None

def handle_signal(signal_data: Signal):
    """
    Handle tradeable signal from real-time detector.

    In paper mode: Log the signal
    In live mode: Create and submit order
    """
    paper_mode = os.getenv("PAPER_TRADING", "true").lower() == "true"
    mode = ExecutionMode.PAPER if paper_mode else ExecutionMode.LIVE

    # Create execution order
    order = ExecutionOrder.from_signal(signal_data, mode=mode)

    logger.info(f"\n{'='*60}")
    logger.info(f"SIGNAL: {signal_data.action} on {signal_data.kalshi.ticker}")
    logger.info(f"  Edge: {signal_data.edge.best_edge}c ({signal_data.edge.edge_pct}%)")
    logger.info(f"  Confidence: {signal_data.confidence_score} ({signal_data.confidence_tier})")
    logger.info(f"  Contracts: {signal_data.recommended_contracts}")
    logger.info(f"  Risk: ${signal_data.risk_amount:.2f}")
    logger.info(f"  Mode: {mode.value}")
    logger.info(f"{'='*60}\n")

    if paper_mode:
        logger.info("Paper trade logged (set PAPER_TRADING=false for live)")
    else:
        # TODO: Submit to Kalshi executor
        logger.info("Live execution not yet implemented")

def shutdown(signum, frame):
    """Handle shutdown signals."""
    logger.info("Shutdown requested...")
    if _client:
        _client.stop()

async def main():
    global _client

    logger.info("=" * 60)
    logger.info("KALSHI REAL-TIME ARB DETECTION")
    logger.info("=" * 60)
    logger.info(f"Environment: {os.getenv('KALSHI_ENV', 'demo')}")
    logger.info(f"Profile: {os.getenv('TRADING_PROFILE', 'STANDARD')}")
    logger.info(f"Paper Trading: {os.getenv('PAPER_TRADING', 'true')}")

    # Sports filtering info
    series_env = os.getenv("SPORTS_SERIES_TICKERS", "")
    if series_env:
        logger.info(f"Sports Filter: {series_env}")
    else:
        logger.info("Sports Filter: ALL (KXNFL, KXNBA, KXMLB, KXNHL, KXNCAAF, KXNCAAB)")
    logger.info("NOTE: Only sports markets are ingested (politics, economics excluded)")
    logger.info("=" * 60)

    # Setup signal handlers
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Create client
    _client = IntegratedWebSocketClient.from_env(on_signal=handle_signal)

    try:
        await _client.run()
    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        logger.info("\nFinal Statistics:")
        stats = _client.get_stats()
        for category, values in stats.items():
            logger.info(f"\n{category}:")
            if isinstance(values, dict):
                for k, v in values.items():
                    logger.info(f"  {k}: {v}")
            else:
                logger.info(f"  {values}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## File Structure Summary

```
app/
├── connectors/
│   └── kalshi/
│       ├── client.py           (existing REST client)
│       ├── auth.py             (existing auth)
│       ├── market_cache.py     (NEW - Phase 1)
│       ├── ws_processor.py     (NEW - Phase 2)
│       └── ws_client.py        (NEW - Phase 4)
├── services/
│   ├── arb_pipeline.py         (existing batch pipeline)
│   ├── realtime_detector.py    (NEW - Phase 3)
│   └── odds_ingest.py          (existing)
├── cli/
│   ├── run_ingest.py           (existing)
│   └── run_realtime.py         (NEW - Phase 5)
└── execution/
    ├── models.py               (existing)
    └── circuit_breaker.py      (existing)

scripts/
└── kalshi_ws_sports_only_consumer.py  (existing, can be deprecated)
```

---

## Implementation Order

### Step 1: Create `market_cache.py`
- Thread-safe metadata cache
- Populate from REST, update from WebSocket
- Test: Unit tests for cache operations

### Step 2: Create `ws_processor.py`
- Message routing to QuestDB/Redis
- Statistics tracking
- Test: Mock messages, verify ILP writes

### Step 3: Create `realtime_detector.py`
- Integration with existing EdgeDetector
- Rate limiting per ticker
- Circuit breaker checks
- Test: Mock market updates, verify signals

### Step 4: Create `ws_client.py`
- Full async client
- Market discovery
- WebSocket lifecycle
- Test: Integration with demo API

### Step 5: Create `run_realtime.py`
- CLI entry point
- Signal handling
- Graceful shutdown
- Test: End-to-end with demo API

### Step 6: Update Existing Files
- Add `KalshiAuth` helper class to `auth.py` (if not present)
- Add `get_ws_headers()` method

---

## Testing Strategy

```bash
# Unit tests
pytest tests/test_market_cache.py
pytest tests/test_ws_processor.py
pytest tests/test_realtime_detector.py

# Integration test (requires demo API credentials)
KALSHI_ENV=demo python -m app.cli.run_realtime

# Full system test
docker-compose up -d  # Start QuestDB, Redis
PAPER_TRADING=true python -m app.cli.run_realtime
```

---

## Performance Targets

| Metric | Target | Implementation |
|--------|--------|---------------|
| Message processing | <1ms | Direct ILP writes |
| Detection latency | <50ms | Rate-limited per ticker |
| Signal to log | <100ms | Callback chain |
| Memory (1000 markets) | <50MB | Efficient cache |

---

## Sources

- [Kalshi WebSocket Quick Start](https://docs.kalshi.com/getting_started/quick_start_websockets)
- [Kalshi WebSocket Connection](https://docs.kalshi.com/websockets/websocket-connection)
- [Kalshi API Introduction](https://docs.kalshi.com/welcome)
