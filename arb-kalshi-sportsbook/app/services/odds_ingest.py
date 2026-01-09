"""
Odds API Ingestion Service - Low Latency Pipeline

Architecture:
    Odds API ──► Redis (hot cache) ──► QuestDB (cold storage)
                     │
                     └──► Arb Detector (reads from Redis)

Redis provides sub-ms reads for the arb detection loop.
QuestDB stores full history for analytics.

Usage:
    python -m app.services.odds_ingest              # Single fetch
    python -m app.services.odds_ingest --continuous # Poll every 30s
    python -m app.services.odds_ingest --sport nfl  # NFL only
"""

import argparse
import time
from datetime import datetime, timezone
from typing import Optional

import redis

from app.connectors.odds_api import (
    OddsAPIClient,
    SportsbookEvent,
    SUPPORTED_SPORTS,
    TARGET_BOOKMAKERS,
)
from app.data.questdb import QuestDBILPClient


# =============================================================================
# CONFIGURATION
# =============================================================================

ODDS_API_KEY = "dac80126dedbfbe3ff7d1edb216a6c88"

# Redis key patterns
KEY_ODDS = "odds:{sport}:{event_id}:{team}"      # Hash: odds data
KEY_SPORT_EVENTS = "odds:events:{sport}"          # Set: event IDs for sport
KEY_ALL_EVENTS = "odds:events:all"                # Set: all event IDs
KEY_LAST_UPDATE = "odds:last_update"              # String: timestamp
KEY_CONSENSUS = "odds:consensus:{event_id}:{team}" # String: consensus prob

# TTL for Redis keys (2 hours - games can be delayed)
REDIS_TTL = 7200


# =============================================================================
# REDIS STORAGE
# =============================================================================

class OddsRedisStore:
    """
    Redis storage for sportsbook odds.

    Optimized for:
    - Sub-ms reads in arb detection loop
    - Event-based lookups (by sport, by event)
    - Direct comparison to Kalshi prices
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.client = redis.from_url(redis_url, decode_responses=True)

    def ping(self) -> bool:
        """Test connection."""
        try:
            return self.client.ping()
        except redis.ConnectionError:
            return False

    def store_event(self, event: SportsbookEvent) -> int:
        """
        Store event odds in Redis.

        Creates entries for both home and away team.
        Returns number of keys written.
        """
        pipe = self.client.pipeline()
        keys_written = 0

        for team in [event.home_team, event.away_team]:
            # Get consensus probability
            consensus = event.get_consensus_prob(team, "h2h")
            if consensus is None:
                continue

            # Store full odds data
            key = KEY_ODDS.format(
                sport=event.sport_key,
                event_id=event.event_id,
                team=self._normalize_team(team),
            )

            data = event.to_redis_dict(team, "h2h")
            pipe.hset(key, mapping=data)
            pipe.expire(key, REDIS_TTL)

            # Store consensus for quick lookup
            consensus_key = KEY_CONSENSUS.format(
                event_id=event.event_id,
                team=self._normalize_team(team),
            )
            pipe.set(consensus_key, int(round(consensus * 100)))
            pipe.expire(consensus_key, REDIS_TTL)

            # Add to indexes
            pipe.sadd(KEY_SPORT_EVENTS.format(sport=event.sport_key), event.event_id)
            pipe.sadd(KEY_ALL_EVENTS, event.event_id)

            keys_written += 1

        # Update timestamp
        pipe.set(KEY_LAST_UPDATE, datetime.now(timezone.utc).isoformat())

        pipe.execute()
        return keys_written

    def get_consensus(self, event_id: str, team: str) -> Optional[int]:
        """
        Get consensus probability for a team (cents 0-100).

        This is the primary lookup for arb detection.
        Sub-ms latency.
        """
        key = KEY_CONSENSUS.format(
            event_id=event_id,
            team=self._normalize_team(team),
        )
        value = self.client.get(key)
        return int(value) if value else None

    def get_event_odds(self, sport: str, event_id: str, team: str) -> Optional[dict]:
        """Get full odds data for an event/team."""
        key = KEY_ODDS.format(
            sport=sport,
            event_id=event_id,
            team=self._normalize_team(team),
        )
        return self.client.hgetall(key) or None

    def get_all_events(self, sport: str = None) -> list[str]:
        """Get all event IDs, optionally filtered by sport."""
        if sport:
            return list(self.client.smembers(KEY_SPORT_EVENTS.format(sport=sport)))
        return list(self.client.smembers(KEY_ALL_EVENTS))

    def get_last_update(self) -> Optional[str]:
        """Get timestamp of last update."""
        return self.client.get(KEY_LAST_UPDATE)

    def get_stats(self) -> dict:
        """Get storage statistics."""
        return {
            "total_events": self.client.scard(KEY_ALL_EVENTS),
            "last_update": self.get_last_update(),
            "sports": {
                sport: self.client.scard(KEY_SPORT_EVENTS.format(sport=sport))
                for sport in SUPPORTED_SPORTS
            },
        }

    @staticmethod
    def _normalize_team(team: str) -> str:
        """Normalize team name for Redis key."""
        return team.lower().replace(" ", "_").replace(".", "")


# =============================================================================
# QUESTDB STORAGE
# =============================================================================

class OddsQuestDBStore:
    """
    QuestDB storage for sportsbook odds history.

    Used for:
    - Historical analysis
    - P&L calculation
    - Opportunity tracking
    """

    def __init__(self, host: str = "localhost", port: int = 9009):
        self.host = host
        self.port = port

    def store_event(self, event: SportsbookEvent) -> int:
        """
        Store event odds in QuestDB via ILP.

        Returns number of rows written.
        """
        rows = event.to_questdb_rows()

        with QuestDBILPClient(self.host, self.port) as ilp:
            for row in rows:
                self._write_row(ilp, row)

        return len(rows)

    def _write_row(self, ilp: QuestDBILPClient, row: dict):
        """Write single row via ILP."""
        # Build ILP line: sportsbook_odds,tags fields timestamp
        tags = (
            f"event_id={self._escape(row['event_id'])},"
            f"book={self._escape(row['book'])},"
            f"market_type={self._escape(row['market_type'])},"
            f"outcome={self._escape(row['outcome'])}"
        )

        fields = (
            f"odds_decimal={row['odds_decimal']},"
            f"implied_prob={row['implied_prob']}"
        )

        line = f"sportsbook_odds,{tags} {fields} {row['timestamp_ns']}"
        ilp._send(line)

    @staticmethod
    def _escape(value: str) -> str:
        """Escape special characters for ILP tags."""
        return value.replace(" ", "\\ ").replace(",", "\\,").replace("=", "\\=")


# =============================================================================
# INGESTION PIPELINE
# =============================================================================

class OddsIngestionPipeline:
    """
    Main ingestion pipeline: Odds API → Redis + QuestDB.

    Low latency path:
        Odds API → Redis (for arb detection)

    Analytics path:
        Odds API → QuestDB (for history)
    """

    def __init__(
        self,
        api_key: str = ODDS_API_KEY,
        redis_url: str = "redis://localhost:6379/0",
        questdb_host: str = "localhost",
        questdb_port: int = 9009,
        write_questdb: bool = True,
    ):
        self.api_client = OddsAPIClient(api_key)
        self.redis_store = OddsRedisStore(redis_url)
        self.questdb_store = OddsQuestDBStore(questdb_host, questdb_port) if write_questdb else None

    def close(self):
        """Clean up resources."""
        self.api_client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def ingest_sport(self, sport: str, markets: list[str] = None) -> dict:
        """
        Ingest odds for a single sport.

        Args:
            sport: Sport key (e.g., "americanfootball_nfl")
            markets: Markets to fetch (default: h2h)

        Returns:
            Stats dict
        """
        markets = markets or ["h2h"]
        stats = {
            "sport": sport,
            "events": 0,
            "redis_keys": 0,
            "questdb_rows": 0,
            "errors": 0,
        }

        try:
            events = self.api_client.get_odds(sport, markets)
            stats["events"] = len(events)

            for event in events:
                try:
                    # Write to Redis (hot cache)
                    stats["redis_keys"] += self.redis_store.store_event(event)

                    # Write to QuestDB (cold storage)
                    if self.questdb_store:
                        stats["questdb_rows"] += self.questdb_store.store_event(event)

                except Exception as e:
                    print(f"  Error storing event {event.event_id}: {e}")
                    stats["errors"] += 1

        except Exception as e:
            print(f"  Error fetching {sport}: {e}")
            stats["errors"] += 1

        return stats

    def ingest_all(self, sports: list[str] = None, markets: list[str] = None) -> dict:
        """
        Ingest odds for all sports.

        Args:
            sports: Sports to fetch (default: all supported)
            markets: Markets to fetch (default: h2h)

        Returns:
            Aggregated stats
        """
        sports = sports or list(SUPPORTED_SPORTS.keys())
        markets = markets or ["h2h"]

        total_stats = {
            "sports_processed": 0,
            "total_events": 0,
            "total_redis_keys": 0,
            "total_questdb_rows": 0,
            "total_errors": 0,
            "by_sport": {},
        }

        for sport in sports:
            print(f"Fetching {SUPPORTED_SPORTS.get(sport, sport)}...")
            stats = self.ingest_sport(sport, markets)

            total_stats["sports_processed"] += 1
            total_stats["total_events"] += stats["events"]
            total_stats["total_redis_keys"] += stats["redis_keys"]
            total_stats["total_questdb_rows"] += stats["questdb_rows"]
            total_stats["total_errors"] += stats["errors"]
            total_stats["by_sport"][sport] = stats

            print(f"  -> {stats['events']} events, {stats['redis_keys']} Redis keys")

            # Small delay between sports to be nice to API
            time.sleep(0.2)

        total_stats["api_requests_remaining"] = self.api_client.requests_remaining
        return total_stats


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Odds API ingestion")
    parser.add_argument("--sport", help="Single sport to fetch (e.g., nfl, nba)")
    parser.add_argument("--continuous", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=30, help="Poll interval (seconds)")
    parser.add_argument("--redis-url", default="redis://localhost:6379/0")
    parser.add_argument("--no-questdb", action="store_true", help="Skip QuestDB writes")
    parser.add_argument("--stats", action="store_true", help="Show Redis stats only")
    args = parser.parse_args()

    print("Odds API Ingestion Pipeline")
    print("=" * 50)
    print(f"API Key: {ODDS_API_KEY[:8]}...")
    print(f"Redis: {args.redis_url}")
    print(f"QuestDB: {'disabled' if args.no_questdb else 'localhost:9009'}")
    print()

    # Stats only mode
    if args.stats:
        store = OddsRedisStore(args.redis_url)
        if not store.ping():
            print("ERROR: Cannot connect to Redis")
            return 1

        stats = store.get_stats()
        print(f"Total events: {stats['total_events']}")
        print(f"Last update: {stats['last_update']}")
        print("By sport:")
        for sport, count in stats["sports"].items():
            if count > 0:
                print(f"  {sport}: {count}")
        return 0

    # Map short sport names
    sport_map = {
        "nfl": "americanfootball_nfl",
        "ncaaf": "americanfootball_ncaaf",
        "nba": "basketball_nba",
        "ncaab": "basketball_ncaab",
        "mlb": "baseball_mlb",
        "nhl": "icehockey_nhl",
        "mma": "mma_mixed_martial_arts",
        "mls": "soccer_usa_mls",
    }

    sports = None
    if args.sport:
        sport_key = sport_map.get(args.sport.lower(), args.sport)
        sports = [sport_key]

    # Run pipeline
    with OddsIngestionPipeline(
        redis_url=args.redis_url,
        write_questdb=not args.no_questdb,
    ) as pipeline:
        if args.continuous:
            print(f"Running continuously every {args.interval}s (Ctrl+C to stop)")
            while True:
                try:
                    stats = pipeline.ingest_all(sports)
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
                          f"Events: {stats['total_events']}, "
                          f"Redis: {stats['total_redis_keys']}, "
                          f"API remaining: {stats['api_requests_remaining']}")
                    time.sleep(args.interval)
                except KeyboardInterrupt:
                    print("\nStopped.")
                    break
        else:
            stats = pipeline.ingest_all(sports)
            print(f"\nDone!")
            print(f"  Events: {stats['total_events']}")
            print(f"  Redis keys: {stats['total_redis_keys']}")
            print(f"  QuestDB rows: {stats['total_questdb_rows']}")
            print(f"  API remaining: {stats['api_requests_remaining']}")


if __name__ == "__main__":
    main()
