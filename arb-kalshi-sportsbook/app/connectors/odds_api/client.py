"""
The Odds API Client - Low Latency Sportsbook Odds Fetcher

Fetches odds from major US sportsbooks and normalizes to implied probabilities
for comparison against Kalshi prediction market prices.

Target sportsbooks (consensus pricing):
- DraftKings
- FanDuel
- BetMGM
- Caesars

Usage:
    client = OddsAPIClient(api_key="...")

    # Fetch NFL odds
    odds = client.get_odds("americanfootball_nfl", markets=["h2h", "spreads"])

    # Get normalized events ready for arb detection
    events = client.get_normalized_events("americanfootball_nfl")
"""

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
import httpx


# =============================================================================
# CONFIGURATION
# =============================================================================

ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# Primary US sportsbooks for consensus pricing
TARGET_BOOKMAKERS = frozenset({
    "draftkings",
    "fanduel",
    "betmgm",
    "caesars",
})

# Sports we care about (Kalshi sports markets)
SUPPORTED_SPORTS = {
    "americanfootball_nfl": "NFL",
    "americanfootball_ncaaf": "NCAAF",
    "basketball_nba": "NBA",
    "basketball_ncaab": "NCAAB",
    "baseball_mlb": "MLB",
    "icehockey_nhl": "NHL",
    "mma_mixed_martial_arts": "MMA",
    "soccer_usa_mls": "MLS",
}


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class Outcome:
    """Single betting outcome with odds."""
    name: str                    # Team name or "Over"/"Under"
    price: int                   # American odds (-110, +150, etc.)
    point: Optional[float] = None  # Spread or total line

    @property
    def decimal_odds(self) -> float:
        """Convert American odds to decimal."""
        if self.price > 0:
            return (self.price / 100) + 1
        else:
            return (100 / abs(self.price)) + 1

    @property
    def implied_prob(self) -> float:
        """Convert American odds to implied probability (0-1)."""
        if self.price > 0:
            return 100 / (self.price + 100)
        else:
            return abs(self.price) / (abs(self.price) + 100)

    @property
    def implied_prob_cents(self) -> int:
        """Implied probability as cents (0-100) for Kalshi comparison."""
        return int(round(self.implied_prob * 100))


@dataclass
class BookmakerOdds:
    """Odds from a single bookmaker for one market type."""
    key: str                     # "draftkings", "fanduel", etc.
    title: str                   # Display name
    market_type: str             # "h2h", "spreads", "totals"
    outcomes: list[Outcome]
    last_update: datetime

    def get_outcome(self, name: str) -> Optional[Outcome]:
        """Get outcome by team name."""
        for o in self.outcomes:
            if o.name.lower() == name.lower():
                return o
        return None


@dataclass
class SportsbookEvent:
    """
    Normalized sportsbook event with odds from multiple books.

    Designed for direct comparison against Kalshi markets.
    """
    event_id: str                # Odds API event ID
    sport_key: str               # "americanfootball_nfl"
    sport_title: str             # "NFL"
    home_team: str
    away_team: str
    commence_time: datetime
    bookmakers: dict[str, list[BookmakerOdds]] = field(default_factory=dict)

    # Computed consensus (average of target bookmakers)
    _consensus_cache: dict = field(default_factory=dict, repr=False)

    def get_consensus_prob(self, team: str, market: str = "h2h") -> Optional[float]:
        """
        Get consensus implied probability across target bookmakers.

        This is the "true" probability estimate from sharp books.
        Compare against Kalshi yes_ask/100 to find edge.

        Args:
            team: Team name (home or away)
            market: "h2h", "spreads", or "totals"

        Returns:
            Average implied probability (0-1) or None if not available
        """
        cache_key = f"{team}:{market}"
        if cache_key in self._consensus_cache:
            return self._consensus_cache[cache_key]

        probs = []
        for book_key, markets in self.bookmakers.items():
            if book_key not in TARGET_BOOKMAKERS:
                continue
            for book_odds in markets:
                if book_odds.market_type != market:
                    continue
                outcome = book_odds.get_outcome(team)
                if outcome:
                    probs.append(outcome.implied_prob)

        if not probs:
            return None

        consensus = sum(probs) / len(probs)
        self._consensus_cache[cache_key] = consensus
        return consensus

    def get_consensus_prob_cents(self, team: str, market: str = "h2h") -> Optional[int]:
        """Consensus probability as cents (0-100) for Kalshi comparison."""
        prob = self.get_consensus_prob(team, market)
        return int(round(prob * 100)) if prob else None

    def get_all_book_probs(self, team: str, market: str = "h2h") -> dict[str, float]:
        """Get implied probability from each bookmaker."""
        result = {}
        for book_key, markets in self.bookmakers.items():
            for book_odds in markets:
                if book_odds.market_type != market:
                    continue
                outcome = book_odds.get_outcome(team)
                if outcome:
                    result[book_key] = outcome.implied_prob
        return result

    def to_redis_dict(self, team: str, market: str = "h2h") -> dict:
        """Convert to flat dict for Redis storage."""
        consensus = self.get_consensus_prob(team, market)
        all_probs = self.get_all_book_probs(team, market)

        return {
            "event_id": self.event_id,
            "sport": self.sport_key,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "team": team,
            "market": market,
            "commence_time": self.commence_time.isoformat(),
            "consensus_prob": consensus or 0,
            "consensus_cents": int(round((consensus or 0) * 100)),
            "draftkings_prob": all_probs.get("draftkings", 0),
            "fanduel_prob": all_probs.get("fanduel", 0),
            "betmgm_prob": all_probs.get("betmgm", 0),
            "caesars_prob": all_probs.get("caesars", 0),
            "book_count": len(all_probs),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    def to_questdb_rows(self) -> list[dict]:
        """
        Convert to rows for QuestDB sportsbook_odds table.

        Schema: event_id, book, market_type, outcome, odds_decimal, implied_prob, timestamp
        """
        rows = []
        timestamp_ns = int(datetime.now(timezone.utc).timestamp() * 1e9)

        for book_key, markets in self.bookmakers.items():
            for book_odds in markets:
                for outcome in book_odds.outcomes:
                    rows.append({
                        "event_id": self.event_id,
                        "book": book_key,
                        "market_type": book_odds.market_type,
                        "outcome": outcome.name,
                        "odds_decimal": outcome.decimal_odds,
                        "implied_prob": outcome.implied_prob,
                        "timestamp_ns": timestamp_ns,
                    })

        return rows


# =============================================================================
# API CLIENT
# =============================================================================

class OddsAPIClient:
    """
    Low-latency client for The Odds API.

    Designed for high-frequency polling with:
    - Connection pooling (reuse HTTP client)
    - Minimal parsing overhead
    - Direct conversion to comparison-ready format
    """

    def __init__(
        self,
        api_key: str,
        timeout: float = 10.0,
        regions: str = "us",
    ):
        self.api_key = api_key
        self.regions = regions
        self.client = httpx.Client(
            timeout=timeout,
        )

        # Track API usage
        self.requests_remaining: Optional[int] = None
        self.requests_used: Optional[int] = None

    def close(self):
        """Close the HTTP client."""
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _request(self, endpoint: str, params: dict = None) -> dict | list | None:
        """Make API request and track usage."""
        url = f"{ODDS_API_BASE}{endpoint}"
        params = params or {}
        params["apiKey"] = self.api_key

        try:
            resp = self.client.get(url, params=params)

            # Track quota
            self.requests_remaining = resp.headers.get("x-requests-remaining")
            self.requests_used = resp.headers.get("x-requests-used")

            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                print(f"Rate limited. Remaining: {self.requests_remaining}")
                return None
            else:
                print(f"API error: {resp.status_code} - {resp.text[:200]}")
                return None

        except httpx.RequestError as e:
            print(f"Request failed: {e}")
            return None

    def get_sports(self) -> list[dict]:
        """Get list of available sports."""
        return self._request("/sports") or []

    def get_in_season_sports(self) -> list[str]:
        """Get sport keys that are currently in season."""
        sports = self.get_sports()
        return [s["key"] for s in sports if not s.get("has_outrights", True)]

    def get_odds_raw(
        self,
        sport: str,
        markets: list[str] = None,
        bookmakers: list[str] = None,
    ) -> list[dict]:
        """
        Fetch raw odds data from API.

        Args:
            sport: Sport key (e.g., "americanfootball_nfl")
            markets: List of markets ["h2h", "spreads", "totals"]
            bookmakers: Specific bookmakers to fetch (default: all US)

        Returns:
            Raw API response
        """
        params = {
            "regions": self.regions,
            "markets": ",".join(markets or ["h2h"]),
            "oddsFormat": "american",
        }

        if bookmakers:
            params["bookmakers"] = ",".join(bookmakers)

        return self._request(f"/sports/{sport}/odds", params) or []

    def get_odds(
        self,
        sport: str,
        markets: list[str] = None,
    ) -> list[SportsbookEvent]:
        """
        Fetch and parse odds into SportsbookEvent objects.

        Args:
            sport: Sport key
            markets: Markets to fetch (default: h2h only for speed)

        Returns:
            List of SportsbookEvent with normalized odds
        """
        markets = markets or ["h2h"]
        raw = self.get_odds_raw(sport, markets)

        events = []
        for event_data in raw:
            event = self._parse_event(event_data)
            if event:
                events.append(event)

        return events

    def _parse_event(self, data: dict) -> Optional[SportsbookEvent]:
        """Parse raw API event into SportsbookEvent."""
        try:
            commence_time = datetime.fromisoformat(
                data["commence_time"].replace("Z", "+00:00")
            )

            event = SportsbookEvent(
                event_id=data["id"],
                sport_key=data["sport_key"],
                sport_title=data.get("sport_title", ""),
                home_team=data["home_team"],
                away_team=data["away_team"],
                commence_time=commence_time,
            )

            # Parse bookmakers
            for book_data in data.get("bookmakers", []):
                book_key = book_data["key"]

                if book_key not in event.bookmakers:
                    event.bookmakers[book_key] = []

                for market_data in book_data.get("markets", []):
                    outcomes = []
                    for o in market_data.get("outcomes", []):
                        outcomes.append(Outcome(
                            name=o["name"],
                            price=o["price"],
                            point=o.get("point"),
                        ))

                    event.bookmakers[book_key].append(BookmakerOdds(
                        key=book_key,
                        title=book_data.get("title", book_key),
                        market_type=market_data["key"],
                        outcomes=outcomes,
                        last_update=datetime.fromisoformat(
                            book_data["last_update"].replace("Z", "+00:00")
                        ),
                    ))

            return event

        except (KeyError, ValueError) as e:
            print(f"Failed to parse event: {e}")
            return None

    def get_all_sports_odds(
        self,
        sports: list[str] = None,
        markets: list[str] = None,
    ) -> dict[str, list[SportsbookEvent]]:
        """
        Fetch odds for multiple sports.

        Args:
            sports: List of sport keys (default: all supported)
            markets: Markets to fetch

        Returns:
            Dict mapping sport_key -> list of events
        """
        sports = sports or list(SUPPORTED_SPORTS.keys())
        markets = markets or ["h2h"]

        result = {}
        for sport in sports:
            events = self.get_odds(sport, markets)
            if events:
                result[sport] = events
            time.sleep(0.1)  # Small delay between requests

        return result


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def american_to_implied_prob(american: int) -> float:
    """Convert American odds to implied probability."""
    if american > 0:
        return 100 / (american + 100)
    else:
        return abs(american) / (abs(american) + 100)


def remove_vig(prob_a: float, prob_b: float) -> tuple[float, float]:
    """
    Remove vig/juice from two-way market probabilities.

    Sportsbooks add ~10% vig, so probs sum to ~1.05-1.10.
    This normalizes them to sum to 1.0.
    """
    total = prob_a + prob_b
    return prob_a / total, prob_b / total


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    import sys

    API_KEY = "dac80126dedbfbe3ff7d1edb216a6c88"

    print("The Odds API Client Test")
    print("=" * 50)

    with OddsAPIClient(API_KEY) as client:
        # Get available sports
        print("\nAvailable sports:")
        for sport in client.get_in_season_sports()[:10]:
            print(f"  - {sport}")

        # Fetch NFL odds
        print("\nFetching NFL odds...")
        events = client.get_odds("americanfootball_nfl", markets=["h2h"])

        print(f"\nFound {len(events)} NFL events:")
        for event in events[:5]:
            print(f"\n  {event.away_team} @ {event.home_team}")
            print(f"  Commence: {event.commence_time}")

            # Show consensus
            home_prob = event.get_consensus_prob_cents(event.home_team)
            away_prob = event.get_consensus_prob_cents(event.away_team)

            print(f"  Consensus: {event.home_team}={home_prob}c, {event.away_team}={away_prob}c")

            # Show individual books
            print("  Books:")
            for book, prob in event.get_all_book_probs(event.home_team).items():
                print(f"    {book}: {int(prob*100)}%")

        print(f"\nAPI requests remaining: {client.requests_remaining}")
