"""
Event Mapping - Kalshi ↔ Sportsbook Resolution (Consolidated Module)

This is the FOUNDATION of the Kalshi-first architecture.
Maps Kalshi market tickers to sportsbook event IDs for targeted
odds queries and accurate edge detection.

Contains:
    - MappedEvent: Dataclass linking both sides
    - TeamNormalizer: Standardizes team names across platforms
    - EventResolver: Core mapping engine with built-in caching

Usage:
    from app.mapping.resolver import EventResolver, MappedEvent

    resolver = EventResolver(odds_client, redis_store)
    mapped = resolver.resolve(kalshi_market)

    if mapped and mapped.consensus_prob:
        edge = mapped.consensus_prob - mapped.kalshi_yes_ask

CLI Test:
    python -m app.mapping.resolver
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime, date, timezone
from typing import Optional

from rapidfuzz import fuzz, process

# =============================================================================
# KALSHI SPORTS SERIES MAPPING
# =============================================================================

KALSHI_SERIES_TO_SPORT = {
    "KXNFL": "americanfootball_nfl",
    "KXNCAAF": "americanfootball_ncaaf",
    "KXNBA": "basketball_nba",
    "KXNCAAB": "basketball_ncaab",
    "KXMLB": "baseball_mlb",
    "KXNHL": "icehockey_nhl",
}

SPORT_TO_KALSHI_SERIES = {v: k for k, v in KALSHI_SERIES_TO_SPORT.items()}

# =============================================================================
# NFL TEAM DATABASE (All 32 teams with aliases)
# =============================================================================

NFL_TEAMS = {
    "arizona_cardinals": ["arizona", "cardinals", "ari", "cards"],
    "atlanta_falcons": ["atlanta", "falcons", "atl"],
    "baltimore_ravens": ["baltimore", "ravens", "bal", "balt"],
    "buffalo_bills": ["buffalo", "bills", "buf"],
    "carolina_panthers": ["carolina", "panthers", "car"],
    "chicago_bears": ["chicago", "bears", "chi"],
    "cincinnati_bengals": ["cincinnati", "bengals", "cin", "cincy"],
    "cleveland_browns": ["cleveland", "browns", "cle"],
    "dallas_cowboys": ["dallas", "cowboys", "dal"],
    "denver_broncos": ["denver", "broncos", "den"],
    "detroit_lions": ["detroit", "lions", "det"],
    "green_bay_packers": ["green bay", "packers", "gb", "greenbay"],
    "houston_texans": ["houston", "texans", "hou"],
    "indianapolis_colts": ["indianapolis", "colts", "ind", "indy"],
    "jacksonville_jaguars": ["jacksonville", "jaguars", "jax", "jags"],
    "kansas_city_chiefs": ["kansas city", "chiefs", "kc", "kansascity"],
    "las_vegas_raiders": ["las vegas", "raiders", "lv", "vegas", "oak", "oakland"],
    "los_angeles_chargers": ["la chargers", "chargers", "lac", "san diego"],
    "los_angeles_rams": ["la rams", "rams", "lar"],
    "miami_dolphins": ["miami", "dolphins", "mia"],
    "minnesota_vikings": ["minnesota", "vikings", "min"],
    "new_england_patriots": ["new england", "patriots", "ne", "pats"],
    "new_orleans_saints": ["new orleans", "saints", "no", "nola"],
    "new_york_giants": ["ny giants", "giants", "nyg"],
    "new_york_jets": ["ny jets", "jets", "nyj"],
    "philadelphia_eagles": ["philadelphia", "eagles", "phi", "philly"],
    "pittsburgh_steelers": ["pittsburgh", "steelers", "pit"],
    "san_francisco_49ers": ["san francisco", "49ers", "sf", "niners"],
    "seattle_seahawks": ["seattle", "seahawks", "sea"],
    "tampa_bay_buccaneers": ["tampa bay", "buccaneers", "tb", "bucs", "tampa"],
    "tennessee_titans": ["tennessee", "titans", "ten"],
    "washington_commanders": ["washington", "commanders", "was", "wsh", "redskins"],
}

# NBA Teams (30 teams)
NBA_TEAMS = {
    "atlanta_hawks": ["atlanta", "hawks", "atl"],
    "boston_celtics": ["boston", "celtics", "bos"],
    "brooklyn_nets": ["brooklyn", "nets", "bkn", "nj nets", "new jersey"],
    "charlotte_hornets": ["charlotte", "hornets", "cha", "bobcats"],
    "chicago_bulls": ["chicago", "bulls", "chi"],
    "cleveland_cavaliers": ["cleveland", "cavaliers", "cle", "cavs"],
    "dallas_mavericks": ["dallas", "mavericks", "dal", "mavs"],
    "denver_nuggets": ["denver", "nuggets", "den"],
    "detroit_pistons": ["detroit", "pistons", "det"],
    "golden_state_warriors": ["golden state", "warriors", "gs", "gsw"],
    "houston_rockets": ["houston", "rockets", "hou"],
    "indiana_pacers": ["indiana", "pacers", "ind"],
    "los_angeles_clippers": ["la clippers", "clippers", "lac"],
    "los_angeles_lakers": ["la lakers", "lakers", "lal"],
    "memphis_grizzlies": ["memphis", "grizzlies", "mem"],
    "miami_heat": ["miami", "heat", "mia"],
    "milwaukee_bucks": ["milwaukee", "bucks", "mil"],
    "minnesota_timberwolves": ["minnesota", "timberwolves", "min", "wolves"],
    "new_orleans_pelicans": ["new orleans", "pelicans", "nop", "nola"],
    "new_york_knicks": ["new york", "knicks", "ny", "nyk"],
    "oklahoma_city_thunder": ["oklahoma city", "thunder", "okc"],
    "orlando_magic": ["orlando", "magic", "orl"],
    "philadelphia_76ers": ["philadelphia", "76ers", "phi", "sixers"],
    "phoenix_suns": ["phoenix", "suns", "phx"],
    "portland_trail_blazers": ["portland", "trail blazers", "por", "blazers"],
    "sacramento_kings": ["sacramento", "kings", "sac"],
    "san_antonio_spurs": ["san antonio", "spurs", "sa"],
    "toronto_raptors": ["toronto", "raptors", "tor"],
    "utah_jazz": ["utah", "jazz", "uta"],
    "washington_wizards": ["washington", "wizards", "was", "wsh"],
}

# Combine all teams by sport
TEAMS_BY_SPORT = {
    "americanfootball_nfl": NFL_TEAMS,
    "basketball_nba": NBA_TEAMS,
    # Add MLB, NHL as needed
}

# Build reverse lookup: alias -> canonical name
ALIAS_TO_CANONICAL = {}
for sport, teams in TEAMS_BY_SPORT.items():
    for canonical, aliases in teams.items():
        # Add canonical name itself
        ALIAS_TO_CANONICAL[canonical.lower().replace("_", " ")] = (canonical, sport)
        for alias in aliases:
            ALIAS_TO_CANONICAL[alias.lower()] = (canonical, sport)


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class MappedEvent:
    """
    Linked Kalshi ↔ Sportsbook event.

    This is the output of EventResolver.resolve() and contains
    all data needed for edge detection.
    """
    # Kalshi side
    kalshi_ticker: str
    kalshi_title: str
    kalshi_yes_ask: int           # cents 0-100
    kalshi_no_ask: int
    kalshi_yes_bid: int
    kalshi_no_bid: int
    kalshi_volume: int

    # Sportsbook side
    sportsbook_event_id: str
    sportsbook_home_team: str
    sportsbook_away_team: str
    sportsbook_commence_time: datetime

    # Mapping metadata
    target_team: str              # Team the Kalshi YES contract is betting on
    target_team_canonical: str    # Normalized team name
    opponent_team: str
    sport: str

    # Sportsbook consensus (from Redis)
    consensus_prob: Optional[int] = None  # cents 0-100

    # Match quality
    match_confidence: float = 0.0  # 0-1, how confident in the mapping

    @property
    def edge_yes(self) -> int:
        """Edge if buying YES: positive = Kalshi underpriced."""
        if self.consensus_prob is None:
            return 0
        return self.consensus_prob - self.kalshi_yes_ask

    @property
    def edge_no(self) -> int:
        """Edge if buying NO: positive = Kalshi underpriced."""
        if self.consensus_prob is None:
            return 0
        return (100 - self.consensus_prob) - self.kalshi_no_ask

    @property
    def best_edge(self) -> int:
        """Best edge available (YES or NO)."""
        return max(self.edge_yes, self.edge_no)

    @property
    def best_side(self) -> str:
        """Which side has the best edge."""
        return "yes" if self.edge_yes >= self.edge_no else "no"

    def to_dict(self) -> dict:
        """Convert to dict for Redis storage."""
        return {
            "kalshi_ticker": self.kalshi_ticker,
            "kalshi_title": self.kalshi_title,
            "kalshi_yes_ask": self.kalshi_yes_ask,
            "kalshi_no_ask": self.kalshi_no_ask,
            "kalshi_volume": self.kalshi_volume,
            "sportsbook_event_id": self.sportsbook_event_id,
            "sportsbook_home_team": self.sportsbook_home_team,
            "sportsbook_away_team": self.sportsbook_away_team,
            "sportsbook_commence_time": self.sportsbook_commence_time.isoformat(),
            "target_team": self.target_team,
            "target_team_canonical": self.target_team_canonical,
            "sport": self.sport,
            "consensus_prob": self.consensus_prob,
            "match_confidence": self.match_confidence,
        }


# =============================================================================
# TEAM NORMALIZER
# =============================================================================

class TeamNormalizer:
    """
    Normalizes team names across Kalshi and sportsbook platforms.

    Handles variations like:
    - "Buffalo Bills" -> "buffalo_bills"
    - "BUF" -> "buffalo_bills"
    - "buffalo" -> "buffalo_bills"
    """

    def __init__(self):
        self._cache: dict[str, str] = {}

    def normalize(self, name: str, sport: str = None) -> Optional[str]:
        """
        Convert any team name variant to canonical form.

        Args:
            name: Raw team name (e.g., "Buffalo Bills", "BUF", "Bills")
            sport: Optional sport hint for disambiguation

        Returns:
            Canonical name (e.g., "buffalo_bills") or None if not found
        """
        cache_key = f"{name}:{sport}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Clean input
        clean = name.lower().strip()
        clean = re.sub(r'[^\w\s]', '', clean)  # Remove punctuation

        # Direct lookup
        if clean in ALIAS_TO_CANONICAL:
            canonical, found_sport = ALIAS_TO_CANONICAL[clean]
            if sport is None or found_sport == sport:
                self._cache[cache_key] = canonical
                return canonical

        # Try with underscores replaced by spaces
        clean_spaces = clean.replace("_", " ")
        if clean_spaces in ALIAS_TO_CANONICAL:
            canonical, found_sport = ALIAS_TO_CANONICAL[clean_spaces]
            if sport is None or found_sport == sport:
                self._cache[cache_key] = canonical
                return canonical

        # Fuzzy match if direct lookup fails
        if sport and sport in TEAMS_BY_SPORT:
            teams = TEAMS_BY_SPORT[sport]
            all_names = []
            for canonical, aliases in teams.items():
                all_names.append((canonical.replace("_", " "), canonical))
                for alias in aliases:
                    all_names.append((alias, canonical))

            # Find best fuzzy match
            choices = [n[0] for n in all_names]
            result = process.extractOne(clean, choices, scorer=fuzz.ratio)

            if result and result[1] >= 80:  # 80% confidence threshold
                matched_name = result[0]
                for display, canonical in all_names:
                    if display == matched_name:
                        self._cache[cache_key] = canonical
                        return canonical

        return None

    def get_display_name(self, canonical: str) -> str:
        """Convert canonical name back to display format."""
        return canonical.replace("_", " ").title()


# =============================================================================
# EVENT RESOLVER
# =============================================================================

class EventResolver:
    """
    Core mapping engine with built-in caching.

    Resolves Kalshi markets to sportsbook events by:
    1. Parsing team names from Kalshi title
    2. Extracting event date from ticker
    3. Matching to sportsbook events via team + date
    4. Looking up consensus probability from Redis
    """

    # Kalshi title patterns
    TITLE_PATTERNS = [
        r"^(.+?)\s+to\s+beat\s+(.+?)(?:\s*\?)?$",
        r"^(.+?)\s+to\s+win\s+(?:vs\.?|against|over)\s+(.+?)(?:\s*\?)?$",
        r"^(.+?)\s+over\s+(.+?)(?:\s*\?)?$",
        r"^Will\s+(.+?)\s+beat\s+(.+?)(?:\s*\?)?$",
    ]

    # Month abbreviations for ticker parsing
    MONTH_MAP = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
        "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
        "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
    }

    def __init__(
        self,
        odds_client=None,
        redis_store=None,
        cache_ttl: int = 300,  # 5 minute cache
    ):
        """
        Initialize resolver.

        Args:
            odds_client: OddsAPIClient for fetching sportsbook events
            redis_store: OddsRedisStore for consensus lookups
            cache_ttl: Seconds to cache sportsbook events
        """
        self.odds = odds_client
        self.redis = redis_store
        self.normalizer = TeamNormalizer()
        self.cache_ttl = cache_ttl

        # In-memory cache for sportsbook events
        self._sportsbook_cache: dict[str, list] = {}
        self._cache_timestamps: dict[str, float] = {}

        # Mapping cache (kalshi_ticker -> MappedEvent)
        self._mapping_cache: dict[str, MappedEvent] = {}

    def resolve(self, kalshi_market) -> Optional[MappedEvent]:
        """
        Resolve a Kalshi market to a MappedEvent.

        Args:
            kalshi_market: Market object from KalshiClient

        Returns:
            MappedEvent if mapping found, None otherwise
        """
        ticker = kalshi_market.ticker

        # Check mapping cache first
        if ticker in self._mapping_cache:
            cached = self._mapping_cache[ticker]
            # Refresh consensus from Redis
            if self.redis and cached.sportsbook_event_id:
                consensus = self.redis.get_consensus(
                    cached.sportsbook_event_id,
                    self.normalizer.normalize(cached.target_team, cached.sport) or ""
                )
                cached.consensus_prob = consensus
            return cached

        # Parse Kalshi market
        teams = self._parse_title(kalshi_market.title)
        if not teams:
            return None

        favorite, underdog = teams
        event_date = self._parse_ticker_date(ticker)
        sport = self._ticker_to_sport(ticker)

        if not sport:
            return None

        # Normalize team names
        favorite_canonical = self.normalizer.normalize(favorite, sport)
        underdog_canonical = self.normalizer.normalize(underdog, sport)

        if not favorite_canonical:
            return None

        # Find matching sportsbook event
        sportsbook_event, confidence = self._find_matching_event(
            sport=sport,
            team1=favorite,
            team2=underdog,
            event_date=event_date,
        )

        if not sportsbook_event:
            return None

        # Get consensus from Redis
        consensus = None
        if self.redis:
            # Try normalized name first
            consensus = self.redis.get_consensus(
                sportsbook_event.event_id,
                favorite_canonical.replace("_", " ") if favorite_canonical else favorite.lower()
            )
            # Try display name if that fails
            if consensus is None:
                consensus = self.redis.get_consensus(
                    sportsbook_event.event_id,
                    self.normalizer.get_display_name(favorite_canonical) if favorite_canonical else favorite
                )

        # Build MappedEvent
        mapped = MappedEvent(
            kalshi_ticker=ticker,
            kalshi_title=kalshi_market.title,
            kalshi_yes_ask=kalshi_market.yes_ask,
            kalshi_no_ask=kalshi_market.no_ask,
            kalshi_yes_bid=kalshi_market.yes_bid,
            kalshi_no_bid=kalshi_market.no_bid,
            kalshi_volume=kalshi_market.volume,
            sportsbook_event_id=sportsbook_event.event_id,
            sportsbook_home_team=sportsbook_event.home_team,
            sportsbook_away_team=sportsbook_event.away_team,
            sportsbook_commence_time=sportsbook_event.commence_time,
            target_team=favorite,
            target_team_canonical=favorite_canonical or favorite.lower().replace(" ", "_"),
            opponent_team=underdog,
            sport=sport,
            consensus_prob=consensus,
            match_confidence=confidence,
        )

        # Cache the mapping
        self._mapping_cache[ticker] = mapped

        return mapped

    def _parse_title(self, title: str) -> Optional[tuple[str, str]]:
        """
        Parse Kalshi market title to extract teams.

        Returns:
            (favorite_team, underdog_team) or None
        """
        for pattern in self.TITLE_PATTERNS:
            match = re.match(pattern, title, re.IGNORECASE)
            if match:
                team1 = match.group(1).strip()
                team2 = match.group(2).strip()
                return (team1, team2)
        return None

    def _parse_ticker_date(self, ticker: str) -> Optional[date]:
        """
        Parse date from Kalshi ticker.

        Format: KXNFL-26JAN11-BUF
        - 26 = year (2026)
        - JAN = month
        - 11 = day
        """
        match = re.search(r"-(\d{2})([A-Z]{3})(\d{2})-", ticker)
        if not match:
            return None

        year_short, month_str, day_str = match.groups()

        year = 2000 + int(year_short)
        month = self.MONTH_MAP.get(month_str.upper())
        day = int(day_str)

        if not month:
            return None

        try:
            return date(year, month, day)
        except ValueError:
            return None

    def _ticker_to_sport(self, ticker: str) -> Optional[str]:
        """Extract sport from Kalshi ticker."""
        for series, sport in KALSHI_SERIES_TO_SPORT.items():
            if ticker.startswith(series):
                return sport
        return None

    def _find_matching_event(
        self,
        sport: str,
        team1: str,
        team2: str,
        event_date: Optional[date],
    ) -> tuple[Optional[object], float]:
        """
        Find sportsbook event matching the teams and date.

        Returns:
            (SportsbookEvent, confidence_score) or (None, 0)
        """
        events = self._get_sportsbook_events(sport)
        if not events:
            return (None, 0)

        best_match = None
        best_score = 0

        team1_lower = team1.lower()
        team2_lower = team2.lower()

        for event in events:
            home_lower = event.home_team.lower()
            away_lower = event.away_team.lower()

            # Score team matching
            team_score = 0

            # Check if team1 matches either home or away
            t1_home = fuzz.partial_ratio(team1_lower, home_lower)
            t1_away = fuzz.partial_ratio(team1_lower, away_lower)
            t1_best = max(t1_home, t1_away)

            # Check if team2 matches the other team
            if t1_home >= t1_away:
                t2_score = fuzz.partial_ratio(team2_lower, away_lower)
            else:
                t2_score = fuzz.partial_ratio(team2_lower, home_lower)

            team_score = (t1_best + t2_score) / 2

            # Score date matching
            date_score = 100
            if event_date and hasattr(event, 'commence_time'):
                event_day = event.commence_time.date()
                days_diff = abs((event_day - event_date).days)
                if days_diff == 0:
                    date_score = 100
                elif days_diff == 1:
                    date_score = 80  # Allow 1 day tolerance
                else:
                    date_score = max(0, 100 - days_diff * 20)

            # Combined score
            combined = (team_score * 0.7 + date_score * 0.3)

            if combined > best_score:
                best_score = combined
                best_match = event

        # Require minimum 70% confidence
        if best_score >= 70:
            return (best_match, best_score / 100)

        return (None, 0)

    def _get_sportsbook_events(self, sport: str) -> list:
        """
        Get sportsbook events for a sport (with caching).
        """
        now = time.time()

        # Check cache
        if sport in self._sportsbook_cache:
            if now - self._cache_timestamps.get(sport, 0) < self.cache_ttl:
                return self._sportsbook_cache[sport]

        # Fetch fresh data
        if self.odds:
            try:
                events = self.odds.get_odds(sport, markets=["h2h"])
                self._sportsbook_cache[sport] = events
                self._cache_timestamps[sport] = now
                return events
            except Exception as e:
                print(f"Error fetching {sport} odds: {e}")
                return self._sportsbook_cache.get(sport, [])

        return []

    def clear_cache(self):
        """Clear all caches."""
        self._sportsbook_cache.clear()
        self._cache_timestamps.clear()
        self._mapping_cache.clear()

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "sportsbook_events_cached": sum(len(v) for v in self._sportsbook_cache.values()),
            "sports_cached": list(self._sportsbook_cache.keys()),
            "mappings_cached": len(self._mapping_cache),
        }


# =============================================================================
# REDIS KEY PATTERNS (for reference)
# =============================================================================

# These match the patterns in odds_ingest.py
KEY_MAPPING = "mapping:{ticker}"           # Hash: MappedEvent data
KEY_MAPPING_INDEX = "mapping:sport:{sport}"  # Set: tickers for sport


# =============================================================================
# CLI TEST
# =============================================================================

def test_resolver():
    """Test the resolver with mock data."""
    print("=" * 60)
    print("EVENT RESOLVER TEST")
    print("=" * 60)
    print()

    # Test TeamNormalizer
    print("Testing TeamNormalizer:")
    normalizer = TeamNormalizer()

    test_names = [
        ("Buffalo Bills", "americanfootball_nfl"),
        ("BUF", "americanfootball_nfl"),
        ("bills", "americanfootball_nfl"),
        ("buffalo", "americanfootball_nfl"),
        ("Kansas City Chiefs", "americanfootball_nfl"),
        ("KC", "americanfootball_nfl"),
        ("Los Angeles Lakers", "basketball_nba"),
        ("lakers", "basketball_nba"),
    ]

    for name, sport in test_names:
        canonical = normalizer.normalize(name, sport)
        print(f"  '{name}' -> {canonical}")

    print()

    # Test title parsing
    print("Testing title parsing:")
    resolver = EventResolver()

    test_titles = [
        "Buffalo Bills to beat Jacksonville Jaguars",
        "Kansas City Chiefs to beat Denver Broncos?",
        "Will Los Angeles Rams beat Carolina Panthers?",
        "Green Bay Packers over Chicago Bears",
        "Miami Dolphins to win vs New York Jets",
    ]

    for title in test_titles:
        teams = resolver._parse_title(title)
        print(f"  '{title[:40]}...' -> {teams}")

    print()

    # Test ticker date parsing
    print("Testing ticker date parsing:")
    test_tickers = [
        "KXNFL-26JAN11-BUF",
        "KXNFL-26JAN10-CAR",
        "KXNBA-26FEB15-LAL",
    ]

    for ticker in test_tickers:
        event_date = resolver._parse_ticker_date(ticker)
        sport = resolver._ticker_to_sport(ticker)
        print(f"  {ticker} -> date={event_date}, sport={sport}")

    print()
    print("RESOLVER TEST COMPLETE")


if __name__ == "__main__":
    test_resolver()
