"""
Kalshi-First Arbitrage Pipeline - Unified Orchestration

This is the MAIN entry point for the arbitrage detection system.
Everything flows from Kalshi markets:

    1. Scan Kalshi for active sports markets
    2. Map each market to sportsbook events
    3. Get/refresh sportsbook consensus from Redis
    4. Detect edges using existing detector
    5. Return signals for execution

Usage:
    from app.services.arb_pipeline import ArbPipeline

    pipeline = ArbPipeline.from_env()
    signals = pipeline.run_cycle()

    for signal in signals:
        if signal.should_trade:
            print(f"Trade: {signal.action} on {signal.kalshi.ticker}")

CLI Test:
    python -m app.services.arb_pipeline
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from app.connectors.kalshi.client import KalshiClient, Market
from app.connectors.odds_api.client import OddsAPIClient
from app.mapping.resolver import EventResolver, MappedEvent, KALSHI_SERIES_TO_SPORT
from app.services.odds_ingest import OddsRedisStore
from app.arb.detector import EdgeDetector, Signal, KalshiMarket, SportsbookConsensus


# =============================================================================
# CONFIGURATION
# =============================================================================

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "dac80126dedbfbe3ff7d1edb216a6c88")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

# Default trading profile
DEFAULT_PROFILE = os.environ.get("TRADING_PROFILE", "STANDARD")

# How often to refresh sportsbook odds (seconds)
ODDS_REFRESH_INTERVAL = 30


# =============================================================================
# PIPELINE
# =============================================================================

@dataclass
class CycleStats:
    """Statistics from a single detection cycle."""
    kalshi_markets_scanned: int = 0
    events_mapped: int = 0
    consensus_found: int = 0
    signals_generated: int = 0
    tradeable_signals: int = 0
    cycle_time_ms: float = 0
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


class ArbPipeline:
    """
    Unified Kalshi-first arbitrage pipeline.

    This orchestrates:
    - KalshiClient for market scanning
    - EventResolver for mapping
    - OddsRedisStore for consensus lookup
    - EdgeDetector for signal generation
    """

    def __init__(
        self,
        kalshi_client: Optional[KalshiClient] = None,
        odds_client: Optional[OddsAPIClient] = None,
        redis_store: Optional[OddsRedisStore] = None,
        profile: str = DEFAULT_PROFILE,
        bankroll: float = 10000,
    ):
        """
        Initialize pipeline with components.

        Args:
            kalshi_client: KalshiClient for market data (optional for testing)
            odds_client: OddsAPIClient for sportsbook odds
            redis_store: OddsRedisStore for consensus cache
            profile: Trading profile (CONSERVATIVE, STANDARD, AGGRESSIVE, etc.)
            bankroll: Starting bankroll for position sizing
        """
        self.kalshi = kalshi_client
        self.odds = odds_client or OddsAPIClient(ODDS_API_KEY)
        self.redis = redis_store or OddsRedisStore(REDIS_URL)

        # Create resolver with our clients
        self.resolver = EventResolver(
            odds_client=self.odds,
            redis_store=self.redis,
        )

        # Create detector with profile
        self.detector = EdgeDetector(
            profile=profile,
            bankroll=bankroll,
        )

        self.profile = profile
        self._last_odds_refresh = 0

    @classmethod
    def from_env(cls, kalshi_client: KalshiClient = None) -> ArbPipeline:
        """
        Create pipeline from environment variables.

        Environment:
            ODDS_API_KEY: The Odds API key
            REDIS_URL: Redis connection URL
            TRADING_PROFILE: Detection profile
            TRADING_BANKROLL: Starting bankroll
        """
        return cls(
            kalshi_client=kalshi_client,
            profile=os.environ.get("TRADING_PROFILE", DEFAULT_PROFILE),
            bankroll=float(os.environ.get("TRADING_BANKROLL", "10000")),
        )

    def run_cycle(
        self,
        sports: list[str] = None,
        refresh_odds: bool = True,
    ) -> tuple[list[Signal], CycleStats]:
        """
        Run one complete detection cycle.

        Args:
            sports: List of Kalshi series to scan (default: all)
            refresh_odds: Whether to refresh sportsbook odds if stale

        Returns:
            (list of Signal objects, CycleStats)
        """
        start_time = time.monotonic()
        stats = CycleStats()

        signals = []

        # Determine which sports to scan
        if sports is None:
            sports = list(KALSHI_SERIES_TO_SPORT.keys())

        # Step 1: Refresh sportsbook odds if needed
        if refresh_odds and self._should_refresh_odds():
            self._refresh_sportsbook_odds(sports)

        # Step 2: Scan Kalshi markets (or use mock data if no client)
        kalshi_markets = self._get_kalshi_markets(sports)
        stats.kalshi_markets_scanned = len(kalshi_markets)

        # Step 3: Map each market and detect edges
        for market in kalshi_markets:
            # Resolve to sportsbook event
            mapped = self.resolver.resolve(market)

            if mapped is None:
                continue

            stats.events_mapped += 1

            if mapped.consensus_prob is None:
                continue

            stats.consensus_found += 1

            # Build detector inputs from mapped event
            kalshi_input = KalshiMarket(
                ticker=mapped.kalshi_ticker,
                title=mapped.kalshi_title,
                yes_bid=mapped.kalshi_yes_bid,
                yes_ask=mapped.kalshi_yes_ask,
                no_bid=mapped.kalshi_no_bid,
                no_ask=mapped.kalshi_no_ask,
                volume=mapped.kalshi_volume,
                status="active",
            )

            # Get full odds data from Redis if available
            odds_data = None
            if self.redis:
                odds_data = self.redis.get_event_odds(
                    mapped.sport,
                    mapped.sportsbook_event_id,
                    mapped.target_team_canonical,
                )

            # Build book_probs from stored data or synthesize from consensus
            if odds_data and "draftkings_prob" in odds_data:
                book_probs = {
                    "draftkings": float(odds_data.get("draftkings_prob", 0)) * 100,
                    "fanduel": float(odds_data.get("fanduel_prob", 0)) * 100,
                    "betmgm": float(odds_data.get("betmgm_prob", 0)) * 100,
                    "caesars": float(odds_data.get("caesars_prob", 0)) * 100,
                }
                book_probs = {k: int(v) for k, v in book_probs.items() if v > 0}
                book_spread = max(book_probs.values()) - min(book_probs.values()) if book_probs else 0
            else:
                # Synthesize from consensus (assume tight agreement)
                c = mapped.consensus_prob
                book_probs = {"draftkings": c, "fanduel": c, "betmgm": c, "caesars": c}
                book_spread = 0.0

            sportsbook_input = SportsbookConsensus(
                event_id=mapped.sportsbook_event_id,
                team=mapped.target_team,
                sport=mapped.sport,
                consensus_prob=mapped.consensus_prob,
                book_probs=book_probs,
                book_count=len(book_probs),
                book_spread=book_spread,
                updated_at=datetime.now(timezone.utc),
            )

            # Calculate hours to event
            hours_to_event = self._hours_until(mapped.sportsbook_commence_time)

            # Run detection
            signal = self.detector.detect(kalshi_input, sportsbook_input, hours_to_event)

            stats.signals_generated += 1

            if signal.should_trade:
                stats.tradeable_signals += 1
                signals.append(signal)

        # Sort by edge (best first)
        signals.sort(key=lambda s: -s.edge.best_edge)

        stats.cycle_time_ms = (time.monotonic() - start_time) * 1000

        return signals, stats

    def run_cycle_without_kalshi(
        self,
        mock_markets: list[dict],
        sports: list[str] = None,
    ) -> tuple[list[Signal], CycleStats]:
        """
        Run detection cycle with mock Kalshi market data.

        Useful for testing when Kalshi API is not available.

        Args:
            mock_markets: List of dicts with market data
            sports: Sports to refresh odds for

        Returns:
            (signals, stats)
        """
        # Convert mock data to Market-like objects
        class MockMarket:
            def __init__(self, data):
                self.ticker = data["ticker"]
                self.title = data["title"]
                self.yes_ask = data.get("yes_ask", 50)
                self.no_ask = data.get("no_ask", 50)
                self.yes_bid = data.get("yes_bid", 48)
                self.no_bid = data.get("no_bid", 48)
                self.volume = data.get("volume", 1000)
                self.status = data.get("status", "open")

        # Temporarily replace _get_kalshi_markets
        original_method = self._get_kalshi_markets

        def mock_get_markets(sports_list):
            return [MockMarket(m) for m in mock_markets]

        self._get_kalshi_markets = mock_get_markets

        try:
            return self.run_cycle(sports=sports, refresh_odds=True)
        finally:
            self._get_kalshi_markets = original_method

    def _get_kalshi_markets(self, sports: list[str]) -> list:
        """
        Get active Kalshi markets for the given sports.
        """
        if self.kalshi is None:
            return []

        markets = []
        for series in sports:
            try:
                series_markets = self.kalshi.get_markets(
                    series_ticker=series,
                    status="open",
                    limit=100,
                )
                markets.extend(series_markets)
            except Exception as e:
                print(f"Error fetching {series} markets: {e}")

        return markets

    def _should_refresh_odds(self) -> bool:
        """Check if sportsbook odds should be refreshed."""
        return time.time() - self._last_odds_refresh > ODDS_REFRESH_INTERVAL

    def _refresh_sportsbook_odds(self, sports: list[str]):
        """
        Refresh sportsbook odds for the given sports.

        Stores consensus in Redis via OddsRedisStore.
        """
        for series in sports:
            sport = KALSHI_SERIES_TO_SPORT.get(series)
            if not sport:
                continue

            try:
                events = self.odds.get_odds(sport, markets=["h2h"])
                for event in events:
                    self.redis.store_event(event)
            except Exception as e:
                print(f"Error refreshing {sport} odds: {e}")

        self._last_odds_refresh = time.time()

    def _hours_until(self, dt: datetime) -> float:
        """Calculate hours until a datetime."""
        now = datetime.now(timezone.utc)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = dt - now
        return max(0, delta.total_seconds() / 3600)

    def get_status(self) -> dict:
        """Get pipeline status."""
        return {
            "profile": self.profile,
            "redis_connected": self.redis.ping() if self.redis else False,
            "odds_api_remaining": self.odds.requests_remaining,
            "resolver_cache_stats": self.resolver.get_cache_stats(),
            "last_odds_refresh": self._last_odds_refresh,
        }

    def close(self):
        """Clean up resources."""
        if self.odds:
            self.odds.close()


# =============================================================================
# CLI TEST
# =============================================================================

def test_pipeline():
    """Test the pipeline with mock Kalshi data."""
    print("=" * 60)
    print("ARB PIPELINE TEST (Kalshi-First)")
    print("=" * 60)
    print()

    # Create pipeline (no Kalshi client needed for test)
    pipeline = ArbPipeline.from_env()

    print(f"Profile: {pipeline.profile}")
    print(f"Redis connected: {pipeline.redis.ping()}")
    print()

    # Create mock Kalshi markets based on real NFL games
    # These simulate what we'd get from Kalshi API
    mock_markets = [
        {
            "ticker": "KXNFL-26JAN11-BUF",
            "title": "Buffalo Bills to beat Jacksonville Jaguars",
            "yes_ask": 82,  # Kalshi thinks 82% chance BUF wins
            "no_ask": 20,
            "yes_bid": 80,
            "no_bid": 18,
            "volume": 5000,
        },
        {
            "ticker": "KXNFL-26JAN11-GB",
            "title": "Green Bay Packers to beat Chicago Bears",
            "yes_ask": 52,  # Close to 50/50
            "no_ask": 50,
            "yes_bid": 50,
            "no_bid": 48,
            "volume": 3000,
        },
        {
            "ticker": "KXNFL-26JAN10-LAR",
            "title": "Los Angeles Rams to beat Carolina Panthers",
            "yes_ask": 85,  # Heavy favorite
            "no_ask": 17,
            "yes_bid": 83,
            "no_bid": 15,
            "volume": 8000,
        },
    ]

    print("Running detection cycle with mock Kalshi data...")
    print()

    signals, stats = pipeline.run_cycle_without_kalshi(
        mock_markets=mock_markets,
        sports=["KXNFL"],
    )

    print("Cycle Statistics:")
    print(f"  Markets scanned: {stats.kalshi_markets_scanned}")
    print(f"  Events mapped: {stats.events_mapped}")
    print(f"  Consensus found: {stats.consensus_found}")
    print(f"  Signals generated: {stats.signals_generated}")
    print(f"  Tradeable signals: {stats.tradeable_signals}")
    print(f"  Cycle time: {stats.cycle_time_ms:.1f}ms")
    print()

    if signals:
        print("Tradeable Signals:")
        for signal in signals:
            print(f"\n  {signal.kalshi.ticker}")
            print(f"    Action: {signal.action}")
            print(f"    Kalshi YES ask: {signal.kalshi.yes_ask}c")
            print(f"    Sportsbook consensus: {signal.sportsbook.consensus_prob}c")
            print(f"    Edge: {signal.edge.best_edge}c ({signal.edge.best_side})")
            print(f"    Confidence: {signal.confidence_score} ({signal.confidence_tier})")
            print(f"    Recommended: {signal.recommended_contracts} contracts")
    else:
        print("No tradeable signals found (edges may be below threshold)")

    print()
    print("Pipeline Status:")
    status = pipeline.get_status()
    for key, value in status.items():
        print(f"  {key}: {value}")

    pipeline.close()
    print()
    print("ARB PIPELINE TEST COMPLETE")


if __name__ == "__main__":
    test_pipeline()
