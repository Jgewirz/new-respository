# Execution Engine Plan - Kalshi-First Architecture

## Executive Summary

This document provides a comprehensive plan for a **Kalshi-first** arbitrage detection and execution system. The key insight is that event mapping between Kalshi and sportsbooks must be foundational (Phase 1), not polish (Phase 4).

**Critical Change**: The system flow is now:
```
Kalshi Markets → Event Mapping → Targeted Sportsbook Queries → Edge Detection → Execution
```

---

## Architecture Overview

### Kalshi-First Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        KALSHI-FIRST ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   STEP 1: KALSHI MARKET SCAN (Source of Truth)                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  KalshiClient.get_markets(series_ticker="KXNFL", status="open")    │  │
│   │                                                                     │  │
│   │  Returns:                                                           │  │
│   │  ┌──────────────────────────────────────────────────────────────┐  │  │
│   │  │ Market(ticker="KXNFL-26JAN11-BUF",                           │  │  │
│   │  │        title="Buffalo Bills to beat Jacksonville Jaguars",   │  │  │
│   │  │        yes_ask=51, no_ask=51, volume=5000)                   │  │  │
│   │  └──────────────────────────────────────────────────────────────┘  │  │
│   └────────────────────────────┬────────────────────────────────────────┘  │
│                                │                                            │
│                                ▼                                            │
│   STEP 2: EVENT MAPPING (New Component)                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  EventResolver.resolve(kalshi_market)                               │  │
│   │                                                                     │  │
│   │  1. Parse title: "Buffalo Bills to beat Jacksonville Jaguars"      │  │
│   │     → favorite_team = "Buffalo Bills"                               │  │
│   │     → underdog_team = "Jacksonville Jaguars"                        │  │
│   │                                                                     │  │
│   │  2. Extract date from ticker: KXNFL-26JAN11-BUF → Jan 11, 2026     │  │
│   │                                                                     │  │
│   │  3. Match to sportsbook events (fuzzy match on team + date)        │  │
│   │     → sportsbook_event_id = "abc123def456"                         │  │
│   │                                                                     │  │
│   │  4. Return MappedEvent with both sides linked                      │  │
│   └────────────────────────────┬────────────────────────────────────────┘  │
│                                │                                            │
│                                ▼                                            │
│   STEP 3: TARGETED SPORTSBOOK QUERY (Only for Mapped Events)               │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  Option A: Redis Cache Hit (sub-ms)                                 │  │
│   │    redis.get("odds:consensus:abc123def456:buffalo_bills")          │  │
│   │    → 55 (cents)                                                     │  │
│   │                                                                     │  │
│   │  Option B: API Fetch (if stale or missing)                         │  │
│   │    OddsAPIClient.get_odds("americanfootball_nfl")                  │  │
│   │    → Filter to matched event_id                                     │  │
│   └────────────────────────────┬────────────────────────────────────────┘  │
│                                │                                            │
│                                ▼                                            │
│   STEP 4: EDGE DETECTION                                                   │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  EdgeDetector.detect(kalshi_market, sportsbook_consensus)          │  │
│   │                                                                     │  │
│   │  Kalshi YES ask: 51c                                                │  │
│   │  Sportsbook consensus: 55c                                          │  │
│   │  Edge: +4c → BUY YES                                                │  │
│   └────────────────────────────┬────────────────────────────────────────┘  │
│                                │                                            │
│                                ▼                                            │
│   STEP 5: EXECUTION (If Edge Sufficient)                                   │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  CircuitBreaker.check() → KalshiClient.buy_yes(ticker, count, px)  │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Phases

### Phase 1: Event Mapping Foundation (HIGHEST PRIORITY)

This is the critical missing piece. Without mapping, the system cannot connect Kalshi markets to sportsbook odds.

#### New Files to Create

```
app/mapping/
├── __init__.py
├── resolver.py          # Main EventResolver class
├── team_normalizer.py   # Team name standardization
├── cache.py             # Mapping cache (Redis)
└── models.py            # MappedEvent dataclass
```

#### 1.1 Team Name Normalizer (`team_normalizer.py`)

Standardizes team names for matching across platforms.

```python
# NFL team name variations
TEAM_ALIASES = {
    "buffalo_bills": ["buffalo", "bills", "buf"],
    "kansas_city_chiefs": ["kansas city", "chiefs", "kc", "kansas_city"],
    "los_angeles_rams": ["la rams", "rams", "los angeles rams"],
    # ... all 32 NFL teams
    # ... NBA, MLB, NHL teams
}

class TeamNormalizer:
    def normalize(self, name: str) -> str:
        """Convert any team name variant to canonical form."""
        # Lowercase, strip, remove punctuation
        # Fuzzy match against aliases
        # Return canonical name (e.g., "buffalo_bills")
```

#### 1.2 Event Resolver (`resolver.py`)

The core mapping engine.

```python
@dataclass
class MappedEvent:
    """Linked Kalshi ↔ Sportsbook event."""
    kalshi_ticker: str
    kalshi_title: str
    kalshi_yes_ask: int
    kalshi_no_ask: int
    kalshi_volume: int

    sportsbook_event_id: str
    sportsbook_home_team: str
    sportsbook_away_team: str
    sportsbook_commence_time: datetime

    target_team: str           # Team the Kalshi market is betting on
    is_favorite: bool          # Is target team the favorite?

    consensus_prob: int | None # From Redis cache
    confidence: float          # Match confidence (0-1)

class EventResolver:
    def __init__(self, odds_client: OddsAPIClient, redis_store: OddsRedisStore):
        self.odds = odds_client
        self.redis = redis_store
        self.normalizer = TeamNormalizer()
        self._sportsbook_cache: dict[str, list[SportsbookEvent]] = {}
        self._cache_expiry: dict[str, float] = {}

    def resolve_kalshi_market(self, market: Market) -> MappedEvent | None:
        """
        Map a Kalshi market to sportsbook event.

        1. Parse team names from Kalshi title
        2. Extract date from ticker
        3. Find matching sportsbook event
        4. Return MappedEvent with consensus from Redis
        """
        # Parse: "Buffalo Bills to beat Jacksonville Jaguars"
        teams = self._parse_kalshi_title(market.title)
        if not teams:
            return None

        favorite, underdog = teams
        event_date = self._parse_ticker_date(market.ticker)
        sport = self._ticker_to_sport(market.ticker)

        # Find matching sportsbook event
        sportsbook_event = self._find_matching_event(
            sport=sport,
            team1=favorite,
            team2=underdog,
            date=event_date,
        )

        if not sportsbook_event:
            return None

        # Get consensus from Redis
        consensus = self.redis.get_consensus(
            sportsbook_event.event_id,
            self.normalizer.normalize(favorite)
        )

        return MappedEvent(
            kalshi_ticker=market.ticker,
            kalshi_title=market.title,
            kalshi_yes_ask=market.yes_ask,
            kalshi_no_ask=market.no_ask,
            kalshi_volume=market.volume,
            sportsbook_event_id=sportsbook_event.event_id,
            sportsbook_home_team=sportsbook_event.home_team,
            sportsbook_away_team=sportsbook_event.away_team,
            sportsbook_commence_time=sportsbook_event.commence_time,
            target_team=favorite,
            is_favorite=True,
            consensus_prob=consensus,
            confidence=0.95,
        )

    def _parse_kalshi_title(self, title: str) -> tuple[str, str] | None:
        """
        Parse Kalshi market title to extract teams.

        Patterns:
        - "Buffalo Bills to beat Jacksonville Jaguars"
        - "Kansas City Chiefs to win vs Denver Broncos"
        """
        patterns = [
            r"^(.+?)\s+to\s+beat\s+(.+)$",
            r"^(.+?)\s+to\s+win\s+(?:vs\.?|against)\s+(.+)$",
            r"^(.+?)\s+over\s+(.+)$",
        ]
        # ... implementation

    def _parse_ticker_date(self, ticker: str) -> date:
        """
        Parse date from Kalshi ticker.

        KXNFL-26JAN11-BUF → January 11, 2026
        """
        match = re.search(r"-(\d{2})([A-Z]{3})(\d{2})-", ticker)
        if match:
            year, month, day = match.groups()
            # ... conversion logic

    def _find_matching_event(
        self,
        sport: str,
        team1: str,
        team2: str,
        date: date,
    ) -> SportsbookEvent | None:
        """
        Find sportsbook event matching the teams and date.

        Uses fuzzy matching with confidence scoring.
        """
        events = self._get_sportsbook_events(sport)

        best_match = None
        best_score = 0

        for event in events:
            score = self._match_score(event, team1, team2, date)
            if score > best_score and score > 0.8:  # 80% threshold
                best_match = event
                best_score = score

        return best_match
```

#### 1.3 Mapping Cache (`cache.py`)

Persist mappings to avoid repeated lookups.

```python
KEY_MAPPING = "mapping:{kalshi_ticker}"  # Hash: mapped event data
KEY_MAPPING_INDEX = "mapping:index:{sport}"  # Set: all kalshi tickers for sport

class MappingCache:
    def __init__(self, redis_client):
        self.redis = redis_client

    def get(self, kalshi_ticker: str) -> MappedEvent | None:
        """Get cached mapping (sub-ms)."""

    def set(self, mapped_event: MappedEvent, ttl: int = 3600):
        """Cache mapping for 1 hour."""

    def invalidate_sport(self, sport: str):
        """Clear all mappings for a sport (on new event data)."""
```

---

### Phase 2: Kalshi Market Scanner

New service to scan Kalshi for tradeable markets.

#### New File: `app/services/kalshi_scanner.py`

```python
# Sports series on Kalshi
KALSHI_SPORTS_SERIES = {
    "KXNFL": "americanfootball_nfl",
    "KXNCAAF": "americanfootball_ncaaf",
    "KXNBA": "basketball_nba",
    "KXNCAAB": "basketball_ncaab",
    "KXMLB": "baseball_mlb",
    "KXNHL": "icehockey_nhl",
}

class KalshiMarketScanner:
    """
    Scans Kalshi for active sports markets.

    This is the ENTRY POINT for the system.
    Everything flows from Kalshi markets.
    """

    def __init__(
        self,
        kalshi_client: KalshiClient,
        event_resolver: EventResolver,
    ):
        self.kalshi = kalshi_client
        self.resolver = event_resolver

    def scan_all_sports(self) -> list[MappedEvent]:
        """
        Scan all sports series for active markets.

        Returns mapped events ready for edge detection.
        """
        mapped = []

        for series, sport in KALSHI_SPORTS_SERIES.items():
            markets = self.kalshi.get_markets(
                series_ticker=series,
                status="open",
                limit=100,
            )

            for market in markets:
                mapped_event = self.resolver.resolve_kalshi_market(market)
                if mapped_event and mapped_event.consensus_prob:
                    mapped.append(mapped_event)

        return mapped

    def scan_sport(self, series: str) -> list[MappedEvent]:
        """Scan single sport series."""
        # ... similar to above
```

---

### Phase 3: Unified Arbitrage Pipeline

Tie everything together in a single orchestrated pipeline.

#### New File: `app/services/arb_pipeline.py`

```python
class ArbPipeline:
    """
    Unified Kalshi-first arbitrage pipeline.

    Flow:
    1. Scan Kalshi markets
    2. Map to sportsbook events
    3. Get/refresh sportsbook consensus
    4. Detect edges
    5. Generate signals
    """

    def __init__(
        self,
        kalshi_client: KalshiClient,
        odds_client: OddsAPIClient,
        redis_store: OddsRedisStore,
        profile: str = "STANDARD",
    ):
        self.kalshi = kalshi_client
        self.odds = odds_client
        self.redis = redis_store

        self.resolver = EventResolver(odds_client, redis_store)
        self.scanner = KalshiMarketScanner(kalshi_client, self.resolver)
        self.detector = EdgeDetector(profile=profile)

    def run_cycle(self) -> list[Signal]:
        """
        Run one complete detection cycle.

        Target: <100ms total latency
        """
        signals = []

        # Step 1: Scan Kalshi (API call ~50ms)
        mapped_events = self.scanner.scan_all_sports()

        # Step 2: For each mapped event, detect edge
        for event in mapped_events:
            if event.consensus_prob is None:
                continue

            # Build detector inputs
            kalshi_market = KalshiMarket(
                ticker=event.kalshi_ticker,
                title=event.kalshi_title,
                yes_bid=event.kalshi_yes_ask - 2,  # Estimate
                yes_ask=event.kalshi_yes_ask,
                no_bid=event.kalshi_no_ask - 2,
                no_ask=event.kalshi_no_ask,
                volume=event.kalshi_volume,
                status="active",
            )

            sportsbook = SportsbookConsensus(
                event_id=event.sportsbook_event_id,
                team=event.target_team,
                sport=self._get_sport(event.kalshi_ticker),
                consensus_prob=event.consensus_prob,
                book_probs={},  # Could enhance with full breakdown
                book_count=4,
                book_spread=2.0,
                updated_at=datetime.now(timezone.utc),
            )

            # Detect edge
            hours_to_event = self._hours_until(event.sportsbook_commence_time)
            signal = self.detector.detect(kalshi_market, sportsbook, hours_to_event)

            if signal.should_trade:
                signals.append(signal)

        return signals
```

---

### Phase 4: Background Odds Refresh

Keep sportsbook consensus fresh in Redis via background process.

#### Updated: `app/services/odds_ingest.py`

Add targeted refresh based on active Kalshi markets:

```python
def refresh_for_kalshi_markets(self, mapped_events: list[MappedEvent]):
    """
    Refresh ONLY odds for events that have Kalshi markets.

    More efficient than refreshing ALL sportsbook events.
    """
    sports_needed = set()
    event_ids_needed = set()

    for event in mapped_events:
        sport = self._event_id_to_sport(event.sportsbook_event_id)
        sports_needed.add(sport)
        event_ids_needed.add(event.sportsbook_event_id)

    for sport in sports_needed:
        events = self.api_client.get_odds(sport, ["h2h"])
        for event in events:
            if event.event_id in event_ids_needed:
                self.redis_store.store_event(event)
```

---

### Phase 5: Execution Engine (Per Original Plan)

With mapping in place, the execution layer can now work:

```
app/execution/
├── circuit_breaker.py   # Risk controls
├── paper.py             # Paper trading
├── kalshi_executor.py   # Live execution
└── __init__.py
```

#### Updated Main Loop: `app/cli/run_executor.py`

```python
async def main_loop():
    """
    Kalshi-first execution loop.

    100ms cycle time target.
    """
    # Initialize components
    kalshi = KalshiAsyncClient.from_env()
    odds = OddsAPIClient(ODDS_API_KEY)
    redis = OddsRedisStore()

    pipeline = ArbPipeline(kalshi, odds, redis, profile="STANDARD")
    executor = KalshiExecutor(kalshi) if not PAPER_MODE else PaperExecutor()
    breaker = CircuitBreaker()

    # Background odds refresh (every 30s)
    asyncio.create_task(odds_refresh_loop(pipeline))

    while True:
        cycle_start = time.monotonic()

        # Run detection cycle
        signals = pipeline.run_cycle()

        # Execute top signals
        for signal in sorted(signals, key=lambda s: -s.edge.best_edge)[:3]:
            if not breaker.can_trade(signal.risk_amount):
                continue

            result = await executor.execute(signal)
            breaker.record_result(result)
            log_execution(result)

        # Target 100ms cycle
        elapsed = time.monotonic() - cycle_start
        if elapsed < 0.1:
            await asyncio.sleep(0.1 - elapsed)
```

---

## File Dependencies (Updated)

```
run_executor.py (CLI Entry Point)
    │
    ├── arb_pipeline.py (Orchestration)
    │       │
    │       ├── kalshi_scanner.py (Kalshi Market Scan)
    │       │       └── KalshiClient
    │       │
    │       ├── resolver.py (Event Mapping) ◄── NEW CRITICAL
    │       │       ├── team_normalizer.py
    │       │       ├── cache.py
    │       │       └── OddsRedisStore
    │       │
    │       └── detector.py (Edge Detection)
    │
    ├── kalshi_executor.py (Execution)
    │       ├── client.py (Kalshi API)
    │       └── auth.py (RSA Signing)
    │
    └── circuit_breaker.py (Risk Controls)
```

---

## Redis Key Schema (Updated)

```
# Sportsbook Odds (existing)
odds:consensus:{event_id}:{team}     = 55          # Consensus probability
odds:{sport}:{event_id}:{team}       = {hash}      # Full odds data
odds:events:{sport}                  = {set}       # Event IDs by sport

# Event Mapping (NEW)
mapping:{kalshi_ticker}              = {hash}      # MappedEvent data
mapping:index:{sport}                = {set}       # All Kalshi tickers for sport
mapping:reverse:{event_id}           = {set}       # Kalshi tickers for sportsbook event

# Kalshi Cache (NEW)
kalshi:market:{ticker}               = {hash}      # Cached market data
kalshi:markets:{series}              = {set}       # Active tickers by series
```

---

## Latency Budget

| Component | Target | Notes |
|-----------|--------|-------|
| Kalshi API call | <50ms | Get markets for series |
| Redis mapping lookup | <1ms | Cached mappings |
| Redis consensus lookup | <1ms | Per event |
| Edge calculation | <1ms | In-memory |
| **Total detection cycle** | **<100ms** | For all sports |
| Order submission | <100ms | Kalshi API |
| **Total end-to-end** | **<200ms** | Signal to order |

---

## Implementation Priority (Revised)

### Week 1: Foundation
1. `app/mapping/team_normalizer.py` - Team name standardization
2. `app/mapping/models.py` - MappedEvent dataclass
3. `app/mapping/resolver.py` - Core event resolver
4. `app/mapping/cache.py` - Redis mapping cache

### Week 2: Pipeline Integration
5. `app/services/kalshi_scanner.py` - Market scanner
6. `app/services/arb_pipeline.py` - Unified pipeline
7. Update `odds_ingest.py` for targeted refresh

### Week 3: Execution
8. `app/execution/circuit_breaker.py` - Risk controls
9. `app/execution/paper.py` - Paper trading
10. `app/cli/run_executor.py` - Main loop

### Week 4: Production
11. `app/execution/kalshi_executor.py` - Live trading
12. Integration tests
13. Monitoring/alerting

---

## Testing Strategy

### Unit Tests
```python
# test_team_normalizer.py
def test_normalize_buffalo_bills():
    normalizer = TeamNormalizer()
    assert normalizer.normalize("Buffalo Bills") == "buffalo_bills"
    assert normalizer.normalize("BUF") == "buffalo_bills"
    assert normalizer.normalize("buffalo") == "buffalo_bills"

# test_resolver.py
def test_parse_kalshi_title():
    resolver = EventResolver(...)
    teams = resolver._parse_kalshi_title("Buffalo Bills to beat Jacksonville Jaguars")
    assert teams == ("Buffalo Bills", "Jacksonville Jaguars")

def test_parse_ticker_date():
    resolver = EventResolver(...)
    date = resolver._parse_ticker_date("KXNFL-26JAN11-BUF")
    assert date == datetime.date(2026, 1, 11)
```

### Integration Tests
```python
# test_arb_pipeline.py
def test_full_pipeline():
    pipeline = ArbPipeline(...)
    signals = pipeline.run_cycle()

    for signal in signals:
        assert signal.kalshi.ticker is not None
        assert signal.sportsbook.event_id is not None
        assert -100 <= signal.edge.best_edge <= 100
```

---

## Validation Checklist

Before going live:

- [ ] Team normalizer covers all NFL/NBA/MLB/NHL teams
- [ ] Kalshi title parser handles all observed patterns
- [ ] Ticker date parser handles edge cases (year rollover)
- [ ] Mapping cache hit rate > 95%
- [ ] Detection cycle < 100ms (p95)
- [ ] Edge calculations match manual verification
- [ ] Paper trading profitable for 7+ days
- [ ] Circuit breaker triggers correctly
- [ ] Graceful shutdown cancels pending orders

---

## Sources

- [Kalshi API Documentation](https://docs.kalshi.com/welcome)
- [The Odds API Documentation](https://the-odds-api.com/liveapi/guides/v4/)
- [Kalshi Market Structure](https://kalshi.com/markets)
- [RapidFuzz Library](https://github.com/rapidfuzz/RapidFuzz) (for fuzzy matching)
