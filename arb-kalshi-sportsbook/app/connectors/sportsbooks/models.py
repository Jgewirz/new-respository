"""
Sportsbook data models for arb detection.

Core models for comparing sportsbook odds against Kalshi prices.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class NormalizedOdds:
    """
    Normalized odds for comparison against Kalshi.

    All probabilities expressed as 0-100 (same as Kalshi cents).
    """
    event_id: str
    sport: str
    team: str
    market_type: str              # "h2h", "spreads", "totals"

    # Consensus from major books (DraftKings, FanDuel, BetMGM, Caesars)
    consensus_prob: int           # 0-100, comparable to Kalshi yes_ask

    # Individual book probabilities
    draftkings_prob: Optional[int] = None
    fanduel_prob: Optional[int] = None
    betmgm_prob: Optional[int] = None
    caesars_prob: Optional[int] = None

    # Metadata
    book_count: int = 0
    updated_at: Optional[datetime] = None

    def edge_vs_kalshi(self, kalshi_yes_ask: int) -> int:
        """
        Calculate edge (in cents) vs Kalshi price.

        Positive = Kalshi is cheap (buy YES)
        Negative = Kalshi is expensive (buy NO / sell YES)

        Args:
            kalshi_yes_ask: Kalshi YES ask price (0-100)

        Returns:
            Edge in cents. E.g., +5 means 5 cent edge.
        """
        return self.consensus_prob - kalshi_yes_ask

    def should_buy_kalshi_yes(self, kalshi_yes_ask: int, min_edge: int = 3) -> bool:
        """
        Should we buy YES on Kalshi?

        True if sportsbooks imply higher probability than Kalshi price.
        """
        return self.edge_vs_kalshi(kalshi_yes_ask) >= min_edge

    def should_buy_kalshi_no(self, kalshi_no_ask: int, min_edge: int = 3) -> bool:
        """
        Should we buy NO on Kalshi?

        True if sportsbooks imply lower probability (higher NO prob).
        """
        consensus_no = 100 - self.consensus_prob
        return consensus_no - kalshi_no_ask >= min_edge


@dataclass
class ArbSignal:
    """
    Arbitrage signal when Kalshi price differs from sportsbook consensus.

    This is the output of the detection system.
    """
    # Identifiers
    kalshi_ticker: str
    sportsbook_event_id: str
    sport: str
    team: str

    # Pricing (all in cents 0-100)
    kalshi_yes_ask: int
    kalshi_no_ask: int
    sportsbook_consensus: int

    # Calculated edge
    edge_cents: int              # consensus - kalshi_price
    edge_pct: float              # edge as percentage

    # Recommendation
    side: str                    # "yes" or "no"
    confidence: str              # "high", "medium", "low"

    # Metadata
    book_count: int
    detected_at: datetime

    @classmethod
    def from_comparison(
        cls,
        kalshi_ticker: str,
        kalshi_yes_ask: int,
        kalshi_no_ask: int,
        sportsbook_odds: NormalizedOdds,
    ) -> Optional["ArbSignal"]:
        """
        Create ArbSignal if edge exists.

        Returns None if no actionable edge.
        """
        # Check YES side
        yes_edge = sportsbook_odds.consensus_prob - kalshi_yes_ask

        # Check NO side
        sportsbook_no = 100 - sportsbook_odds.consensus_prob
        no_edge = sportsbook_no - kalshi_no_ask

        # Take the better side
        if yes_edge >= 3 and yes_edge > no_edge:
            side = "yes"
            edge = yes_edge
            kalshi_price = kalshi_yes_ask
        elif no_edge >= 3:
            side = "no"
            edge = no_edge
            kalshi_price = kalshi_no_ask
        else:
            return None  # No actionable edge

        # Determine confidence based on edge and book count
        if edge >= 7 and sportsbook_odds.book_count >= 4:
            confidence = "high"
        elif edge >= 5 and sportsbook_odds.book_count >= 3:
            confidence = "medium"
        else:
            confidence = "low"

        return cls(
            kalshi_ticker=kalshi_ticker,
            sportsbook_event_id=sportsbook_odds.event_id,
            sport=sportsbook_odds.sport,
            team=sportsbook_odds.team,
            kalshi_yes_ask=kalshi_yes_ask,
            kalshi_no_ask=kalshi_no_ask,
            sportsbook_consensus=sportsbook_odds.consensus_prob,
            edge_cents=edge,
            edge_pct=edge / kalshi_price * 100 if kalshi_price else 0,
            side=side,
            confidence=confidence,
            book_count=sportsbook_odds.book_count,
            detected_at=datetime.now(),
        )
