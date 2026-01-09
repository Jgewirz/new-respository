"""
Edge Detection Algorithm - Core Detection Module

Implements the strategy from ALGORITHM_STRATEGY.md:
1. Calculate edge between Kalshi and sportsbook consensus
2. Apply entry conditions (min edge, book consensus, liquidity)
3. Score confidence
4. Generate actionable signals

Usage:
    detector = EdgeDetector()
    signals = detector.scan_all()

    for signal in signals:
        if signal.should_trade:
            execute(signal)
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
import json


# =============================================================================
# CONFIGURATION - Entry Conditions from ALGORITHM_STRATEGY.md
# =============================================================================

# Trading profiles: CONSERVATIVE → VERY_AGGRESSIVE
# More aggressive = lower edge requirements, more trades, higher variance

PROFILES = {
    # Original settings - safe, proven
    "CONSERVATIVE": {
        "min_edge_cents": 6,              # Require 6c edge minimum
        "min_edge_high_conf": 4,          # 4c edge if 4/4 books agree
        "min_books": 3,
        "max_book_spread": 2.5,           # Tight consensus required
        "target_books": ["draftkings", "fanduel", "betmgm", "caesars"],
        "max_kalshi_spread": 8,           # Tighter Kalshi spread
        "min_kalshi_volume": 1000,        # Higher volume requirement
        "min_hours_to_event": 3,
        "max_hours_to_event": 36,
        "min_price": 30,
        "max_price": 80,
        "kelly_fraction": 0.15,           # Very conservative sizing
        "confidence_thresholds": {"HIGH": 75, "MEDIUM": 55, "LOW": 35},
    },

    # Standard settings - balanced
    "STANDARD": {
        "min_edge_cents": 5,              # 5c edge minimum
        "min_edge_high_conf": 3,          # 3c edge if 4/4 books agree
        "min_books": 3,
        "max_book_spread": 3.0,
        "target_books": ["draftkings", "fanduel", "betmgm", "caesars"],
        "max_kalshi_spread": 10,
        "min_kalshi_volume": 500,
        "min_hours_to_event": 2,
        "max_hours_to_event": 48,
        "min_price": 25,
        "max_price": 85,
        "kelly_fraction": 0.25,
        "confidence_thresholds": {"HIGH": 70, "MEDIUM": 50, "LOW": 30},
    },

    # Aggressive settings - more trades, more risk
    "AGGRESSIVE": {
        "min_edge_cents": 3,              # Lower to 3c edge
        "min_edge_high_conf": 2,          # 2c edge if 4/4 books agree
        "min_books": 3,
        "max_book_spread": 4.0,           # Allow more book disagreement
        "target_books": ["draftkings", "fanduel", "betmgm", "caesars"],
        "max_kalshi_spread": 12,          # Allow wider Kalshi spread
        "min_kalshi_volume": 300,         # Lower volume threshold
        "min_hours_to_event": 1,          # Trade closer to event
        "max_hours_to_event": 72,         # Trade further out too
        "min_price": 20,                  # Include more longshots
        "max_price": 88,                  # Include heavier favorites
        "kelly_fraction": 0.35,           # Larger positions
        "confidence_thresholds": {"HIGH": 60, "MEDIUM": 40, "LOW": 25},
    },

    # Very aggressive - maximum opportunity capture
    "VERY_AGGRESSIVE": {
        "min_edge_cents": 2,              # 2c edge minimum (very thin)
        "min_edge_high_conf": 1,          # Even 1c if perfect consensus
        "min_books": 2,                   # Accept 2 book consensus
        "max_book_spread": 5.0,           # Accept wide disagreement
        "target_books": ["draftkings", "fanduel", "betmgm", "caesars"],
        "max_kalshi_spread": 15,          # Accept very wide Kalshi spread
        "min_kalshi_volume": 100,         # Very low volume OK
        "min_hours_to_event": 0.5,        # 30 min before event
        "max_hours_to_event": 96,         # 4 days out
        "min_price": 15,                  # Deep longshots
        "max_price": 92,                  # Very heavy favorites
        "kelly_fraction": 0.50,           # Half Kelly
        "confidence_thresholds": {"HIGH": 50, "MEDIUM": 30, "LOW": 15},
    },

    # ANY EDGE - Trade on ANY positive edge (maximum risk, maximum opportunity)
    # WARNING: This will generate many trades, some with very thin margins
    "ANY_EDGE": {
        "min_edge_cents": 1,              # 1c edge = trade (essentially any edge)
        "min_edge_high_conf": 0,          # 0c = any positive edge with consensus
        "min_books": 2,                   # Need at least 2 books for signal
        "max_book_spread": 10.0,          # Accept ANY book disagreement
        "target_books": ["draftkings", "fanduel", "betmgm", "caesars"],
        "max_kalshi_spread": 20,          # Accept any Kalshi spread
        "min_kalshi_volume": 50,          # Very low volume OK
        "min_hours_to_event": 0.25,       # 15 min before event
        "max_hours_to_event": 168,        # 7 days out
        "min_price": 5,                   # Deep longshots OK
        "max_price": 98,                  # Heavy favorites OK
        "kelly_fraction": 0.50,           # Full half-Kelly on thin edges
        "confidence_thresholds": {"HIGH": 30, "MEDIUM": 15, "LOW": 5},
    },

    # SCALP - Ultra-tight edges with high volume for quick flips
    "SCALP": {
        "min_edge_cents": 1,              # 1c edge minimum
        "min_edge_high_conf": 1,          # Always require 1c
        "min_books": 3,                   # Need good consensus for scalps
        "max_book_spread": 2.0,           # Tight consensus for confidence
        "target_books": ["draftkings", "fanduel", "betmgm", "caesars"],
        "max_kalshi_spread": 6,           # Tight spread for entry/exit
        "min_kalshi_volume": 2000,        # High volume for liquidity
        "min_hours_to_event": 0.5,        # Can trade close to event
        "max_hours_to_event": 24,         # Only near-term events
        "min_price": 30,                  # Avoid extremes
        "max_price": 70,                  # Stick to middle
        "kelly_fraction": 0.75,           # Larger size on tight edges
        "confidence_thresholds": {"HIGH": 40, "MEDIUM": 25, "LOW": 10},
    },
}

# Default profile
DEFAULT_PROFILE = "STANDARD"

# Legacy support - ENTRY_CONDITIONS still works
ENTRY_CONDITIONS = PROFILES["STANDARD"].copy()

CONFIDENCE_THRESHOLDS = PROFILES["STANDARD"]["confidence_thresholds"]


def get_profile(name: str = None) -> dict:
    """Get trading profile by name."""
    name = (name or DEFAULT_PROFILE).upper()
    if name not in PROFILES:
        raise ValueError(f"Unknown profile: {name}. Available: {list(PROFILES.keys())}")
    return PROFILES[name].copy()


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class KalshiMarket:
    """Kalshi market data for edge detection."""
    ticker: str
    title: str
    yes_bid: int          # Cents 0-100
    yes_ask: int
    no_bid: int
    no_ask: int
    volume: int
    status: str
    event_time: Optional[datetime] = None

    @property
    def spread(self) -> int:
        """Bid-ask spread in cents."""
        return self.yes_ask - self.yes_bid

    @property
    def midpoint(self) -> float:
        """Midpoint price."""
        return (self.yes_bid + self.yes_ask) / 2


@dataclass
class SportsbookConsensus:
    """Aggregated sportsbook odds for an event."""
    event_id: str
    team: str
    sport: str
    consensus_prob: int    # Cents 0-100 (average of books)
    book_probs: dict       # {"draftkings": 53, "fanduel": 52, ...}
    book_count: int
    book_spread: float     # Max - min among books
    updated_at: datetime

    @property
    def is_consensus_tight(self) -> bool:
        """Books agree within 3c."""
        return self.book_spread <= 3.0

    @property
    def has_enough_books(self) -> bool:
        """At least 3 target books reporting."""
        target = set(ENTRY_CONDITIONS["target_books"])
        reporting = set(self.book_probs.keys())
        return len(target & reporting) >= 3


@dataclass
class EdgeCalculation:
    """Result of edge calculation."""
    yes_edge: int          # Consensus - Kalshi YES ask
    no_edge: int           # (100 - Consensus) - Kalshi NO ask
    best_side: str         # "yes" or "no"
    best_edge: int         # The larger edge
    edge_pct: float        # Edge as percentage of price

    @classmethod
    def calculate(cls, kalshi: KalshiMarket, sportsbook: SportsbookConsensus) -> "EdgeCalculation":
        """
        Calculate edge between Kalshi and sportsbook.

        Positive edge = Kalshi is cheap relative to sportsbook consensus
        """
        # YES side: sportsbook thinks YES is more likely
        yes_edge = sportsbook.consensus_prob - kalshi.yes_ask

        # NO side: sportsbook thinks NO is more likely
        sportsbook_no = 100 - sportsbook.consensus_prob
        no_edge = sportsbook_no - kalshi.no_ask

        # Best side
        if yes_edge >= no_edge:
            best_side = "yes"
            best_edge = yes_edge
            price = kalshi.yes_ask
        else:
            best_side = "no"
            best_edge = no_edge
            price = kalshi.no_ask

        edge_pct = (best_edge / price * 100) if price > 0 else 0

        return cls(
            yes_edge=yes_edge,
            no_edge=no_edge,
            best_side=best_side,
            best_edge=best_edge,
            edge_pct=round(edge_pct, 2),
        )


@dataclass
class ConditionCheck:
    """Single condition validation result."""
    name: str
    required: any
    actual: any
    passed: bool
    reason: str = ""


@dataclass
class Signal:
    """
    Trading signal output.

    This is the final output of the detection algorithm.
    """
    # Identifiers
    signal_id: str
    timestamp: datetime

    # Action
    action: str            # "BUY_YES", "BUY_NO", "NO_TRADE"
    should_trade: bool

    # Market data
    kalshi: KalshiMarket
    sportsbook: SportsbookConsensus

    # Edge
    edge: EdgeCalculation

    # Confidence
    confidence_score: int
    confidence_tier: str   # "HIGH", "MEDIUM", "LOW", "SKIP"
    confidence_reasons: list

    # Validation
    conditions_met: bool
    condition_checks: list

    # Position sizing (if trading)
    recommended_contracts: int = 0
    max_price: int = 0
    risk_amount: float = 0
    potential_profit: float = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON output."""
        return {
            "signal": {
                "id": self.signal_id,
                "timestamp": self.timestamp.isoformat(),
                "action": self.action,
                "should_trade": self.should_trade,

                "kalshi": {
                    "ticker": self.kalshi.ticker,
                    "title": self.kalshi.title,
                    "yes_ask": self.kalshi.yes_ask,
                    "no_ask": self.kalshi.no_ask,
                    "spread": self.kalshi.spread,
                    "volume": self.kalshi.volume,
                },

                "sportsbook": {
                    "consensus": self.sportsbook.consensus_prob,
                    "books": self.sportsbook.book_probs,
                    "book_count": self.sportsbook.book_count,
                    "book_spread": self.sportsbook.book_spread,
                },

                "edge": {
                    "cents": self.edge.best_edge,
                    "percent": self.edge.edge_pct,
                    "side": self.edge.best_side,
                    "yes_edge": self.edge.yes_edge,
                    "no_edge": self.edge.no_edge,
                },

                "confidence": {
                    "score": self.confidence_score,
                    "tier": self.confidence_tier,
                    "reasons": self.confidence_reasons,
                },

                "position": {
                    "recommended_contracts": self.recommended_contracts,
                    "max_price": self.max_price,
                    "risk_amount": self.risk_amount,
                    "potential_profit": self.potential_profit,
                },
            },

            "validation": {
                "all_conditions_met": self.conditions_met,
                "conditions": {
                    c.name: {
                        "required": c.required,
                        "actual": c.actual,
                        "pass": c.passed,
                    }
                    for c in self.condition_checks
                },
            },
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


# =============================================================================
# EDGE DETECTOR
# =============================================================================

class EdgeDetector:
    """
    Core edge detection algorithm.

    Scans Kalshi markets against sportsbook consensus to find
    profitable trading opportunities.

    Profiles:
        - CONSERVATIVE: 6c min edge, tight consensus, lower variance
        - STANDARD: 5c min edge, balanced approach
        - AGGRESSIVE: 3c min edge, more trades
        - VERY_AGGRESSIVE: 2c min edge, high variance
        - ANY_EDGE: 1c min edge, trade on any positive edge
        - SCALP: 1c edge with high volume for quick flips
    """

    def __init__(
        self,
        profile: str = None,
        config: dict = None,
        bankroll: float = 10000,
        kelly_fraction: float = None,
        max_position: int = 500,
    ):
        # Load profile or use custom config
        if profile:
            self.config = get_profile(profile)
            self.profile_name = profile.upper()
        elif config:
            self.config = config
            self.profile_name = "CUSTOM"
        else:
            self.config = ENTRY_CONDITIONS
            self.profile_name = DEFAULT_PROFILE

        self.bankroll = bankroll
        # Use profile's kelly_fraction if not explicitly provided
        self.kelly_fraction = kelly_fraction or self.config.get("kelly_fraction", 0.25)
        self.max_position = max_position

        # Load confidence thresholds from profile
        self.confidence_thresholds = self.config.get(
            "confidence_thresholds",
            CONFIDENCE_THRESHOLDS
        )

    def detect(
        self,
        kalshi: KalshiMarket,
        sportsbook: SportsbookConsensus,
        hours_to_event: float = 12,
    ) -> Signal:
        """
        Detect edge and generate signal.

        Args:
            kalshi: Kalshi market data
            sportsbook: Sportsbook consensus data
            hours_to_event: Hours until event starts

        Returns:
            Signal with trade recommendation
        """
        timestamp = datetime.now(timezone.utc)
        signal_id = f"sig_{timestamp.strftime('%Y%m%d%H%M%S')}_{kalshi.ticker}"

        # 1. Calculate edge
        edge = EdgeCalculation.calculate(kalshi, sportsbook)

        # 2. Check all conditions
        conditions = self._check_conditions(kalshi, sportsbook, edge, hours_to_event)
        conditions_met = all(c.passed for c in conditions)

        # 3. Calculate confidence
        confidence_score, confidence_reasons = self._calculate_confidence(
            edge, sportsbook, kalshi, hours_to_event
        )
        confidence_tier = self._get_confidence_tier(confidence_score)

        # 4. Determine action
        should_trade = conditions_met and confidence_tier != "SKIP"
        action = f"BUY_{edge.best_side.upper()}" if should_trade else "NO_TRADE"

        # 5. Calculate position size if trading
        if should_trade:
            position = self._calculate_position(edge, confidence_score, kalshi)
        else:
            position = {"contracts": 0, "max_price": 0, "risk": 0, "profit": 0}

        return Signal(
            signal_id=signal_id,
            timestamp=timestamp,
            action=action,
            should_trade=should_trade,
            kalshi=kalshi,
            sportsbook=sportsbook,
            edge=edge,
            confidence_score=confidence_score,
            confidence_tier=confidence_tier,
            confidence_reasons=confidence_reasons,
            conditions_met=conditions_met,
            condition_checks=conditions,
            recommended_contracts=position["contracts"],
            max_price=position["max_price"],
            risk_amount=position["risk"],
            potential_profit=position["profit"],
        )

    def _check_conditions(
        self,
        kalshi: KalshiMarket,
        sportsbook: SportsbookConsensus,
        edge: EdgeCalculation,
        hours_to_event: float,
    ) -> list[ConditionCheck]:
        """Check all entry conditions."""
        checks = []

        # Edge check (dynamic based on book consensus)
        if sportsbook.book_count >= 4 and sportsbook.book_spread <= 2:
            min_edge = self.config["min_edge_high_conf"]
        else:
            min_edge = self.config["min_edge_cents"]

        checks.append(ConditionCheck(
            name="min_edge",
            required=min_edge,
            actual=edge.best_edge,
            passed=edge.best_edge >= min_edge,
        ))

        # Book count
        target_books = set(self.config["target_books"])
        reporting_target = len(target_books & set(sportsbook.book_probs.keys()))
        checks.append(ConditionCheck(
            name="min_books",
            required=self.config["min_books"],
            actual=reporting_target,
            passed=reporting_target >= self.config["min_books"],
        ))

        # Book spread
        checks.append(ConditionCheck(
            name="max_book_spread",
            required=self.config["max_book_spread"],
            actual=sportsbook.book_spread,
            passed=sportsbook.book_spread <= self.config["max_book_spread"],
        ))

        # Kalshi spread
        checks.append(ConditionCheck(
            name="max_kalshi_spread",
            required=self.config["max_kalshi_spread"],
            actual=kalshi.spread,
            passed=kalshi.spread <= self.config["max_kalshi_spread"],
        ))

        # Kalshi volume
        checks.append(ConditionCheck(
            name="min_volume",
            required=self.config["min_kalshi_volume"],
            actual=kalshi.volume,
            passed=kalshi.volume >= self.config["min_kalshi_volume"],
        ))

        # Price range
        price = kalshi.yes_ask if edge.best_side == "yes" else kalshi.no_ask
        in_range = self.config["min_price"] <= price <= self.config["max_price"]
        checks.append(ConditionCheck(
            name="price_range",
            required=f"{self.config['min_price']}-{self.config['max_price']}",
            actual=price,
            passed=in_range,
        ))

        # Timing
        timing_ok = (
            self.config["min_hours_to_event"] <= hours_to_event <= self.config["max_hours_to_event"]
        )
        checks.append(ConditionCheck(
            name="timing",
            required=f"{self.config['min_hours_to_event']}-{self.config['max_hours_to_event']}h",
            actual=f"{hours_to_event}h",
            passed=timing_ok,
        ))

        return checks

    def _calculate_confidence(
        self,
        edge: EdgeCalculation,
        sportsbook: SportsbookConsensus,
        kalshi: KalshiMarket,
        hours_to_event: float,
    ) -> tuple[int, list[str]]:
        """
        Calculate confidence score (0-100).

        Higher score = more confident in the edge.
        """
        score = 0
        reasons = []

        # Edge size (0-30 points)
        if edge.best_edge >= 10:
            score += 30
            reasons.append(f"Large edge: {edge.best_edge}c")
        elif edge.best_edge >= 7:
            score += 25
            reasons.append(f"Strong edge: {edge.best_edge}c")
        elif edge.best_edge >= 5:
            score += 20
            reasons.append(f"Solid edge: {edge.best_edge}c")
        elif edge.best_edge >= 3:
            score += 10
            reasons.append(f"Minimal edge: {edge.best_edge}c")

        # Book consensus quality (0-30 points)
        target_count = len(set(self.config["target_books"]) & set(sportsbook.book_probs.keys()))
        if target_count == 4 and sportsbook.book_spread <= 1:
            score += 30
            reasons.append("Perfect consensus: 4/4 books within 1c")
        elif target_count >= 3 and sportsbook.book_spread <= 2:
            score += 25
            reasons.append(f"Strong consensus: {target_count}/4 books within 2c")
        elif target_count >= 3 and sportsbook.book_spread <= 3:
            score += 15
            reasons.append(f"Moderate consensus: {target_count}/4 books")

        # Kalshi liquidity (0-20 points)
        if kalshi.volume >= 5000:
            score += 20
            reasons.append(f"High liquidity: {kalshi.volume:,} volume")
        elif kalshi.volume >= 2000:
            score += 15
            reasons.append(f"Good liquidity: {kalshi.volume:,} volume")
        elif kalshi.volume >= 500:
            score += 10
            reasons.append(f"Adequate liquidity: {kalshi.volume:,} volume")

        # Timing (0-20 points)
        if 6 <= hours_to_event <= 24:
            score += 20
            reasons.append(f"Optimal timing: {hours_to_event:.1f}h to event")
        elif 2 <= hours_to_event <= 48:
            score += 10
            reasons.append(f"Acceptable timing: {hours_to_event:.1f}h to event")

        return score, reasons

    def _get_confidence_tier(self, score: int) -> str:
        """Convert score to tier using profile thresholds."""
        thresholds = self.confidence_thresholds
        if score >= thresholds["HIGH"]:
            return "HIGH"
        elif score >= thresholds["MEDIUM"]:
            return "MEDIUM"
        elif score >= thresholds["LOW"]:
            return "LOW"
        else:
            return "SKIP"

    def _calculate_position(
        self,
        edge: EdgeCalculation,
        confidence_score: int,
        kalshi: KalshiMarket,
    ) -> dict:
        """
        Calculate position size using modified Kelly criterion.

        f* = (p * b - q) / b
        where:
            p = win probability (consensus / 100)
            q = lose probability (1 - p)
            b = odds (payout / risk)
        """
        # Determine which side we're betting
        if edge.best_side == "yes":
            price = kalshi.yes_ask
            fair_value = kalshi.yes_ask + edge.best_edge
        else:
            price = kalshi.no_ask
            fair_value = kalshi.no_ask + edge.best_edge

        # Win probability estimate (from sportsbook consensus)
        p_win = fair_value / 100
        p_lose = 1 - p_win

        # Payout odds: pay `price`, win `100 - price` if correct
        win_amount = 100 - price
        lose_amount = price

        if win_amount <= 0:
            return {"contracts": 0, "max_price": 0, "risk": 0, "profit": 0}

        # Kelly fraction: (p * win - q * lose) / win
        kelly = (p_win * win_amount - p_lose * lose_amount) / win_amount
        kelly = max(0, kelly)  # Can't be negative

        # Apply fractional Kelly
        position_frac = kelly * self.kelly_fraction

        # Confidence adjustment
        if confidence_score >= 70:
            conf_mult = 1.0
        elif confidence_score >= 50:
            conf_mult = 0.5
        else:
            conf_mult = 0.25

        position_frac *= conf_mult

        # Calculate dollar amount
        position_dollars = self.bankroll * position_frac

        # Convert to contracts (each contract is $1 risk at the price)
        contracts = int(position_dollars / (price / 100))

        # Apply max position limit
        contracts = min(contracts, self.max_position)

        # Calculate risk and profit
        risk = contracts * (price / 100)
        profit = contracts * ((100 - price) / 100)

        return {
            "contracts": contracts,
            "max_price": price + 2,  # Allow 2c slippage
            "risk": round(risk, 2),
            "profit": round(profit, 2),
        }


# =============================================================================
# VALIDATION / TEST
# =============================================================================

def validate_detector():
    """
    Validate the detector with real-world examples across all profiles.
    """
    print("=" * 70)
    print("EDGE DETECTOR VALIDATION - ALL PROFILES")
    print("=" * 70)
    print()

    # Test case 1: Small edge (2c) - only passes with aggressive profiles
    test_cases = [
        {
            "name": "Small Edge (2c)",
            "kalshi": KalshiMarket(
                ticker="KXNFL-26JAN11-BUF",
                title="Buffalo Bills to beat Jacksonville Jaguars",
                yes_bid=49,
                yes_ask=51,  # 2c edge vs 53c consensus
                no_bid=49,
                no_ask=51,
                volume=5000,
                status="active",
            ),
            "sportsbook": SportsbookConsensus(
                event_id="test1",
                team="Buffalo Bills",
                sport="nfl",
                consensus_prob=53,
                book_probs={"draftkings": 53, "fanduel": 53, "betmgm": 53, "caesars": 53},
                book_count=4,
                book_spread=0.0,
                updated_at=datetime.now(timezone.utc),
            ),
            "hours": 12,
        },
        {
            "name": "Medium Edge (5c)",
            "kalshi": KalshiMarket(
                ticker="KXNFL-26JAN11-GB",
                title="Green Bay Packers to beat Chicago Bears",
                yes_bid=46,
                yes_ask=50,  # 5c edge vs 55c consensus
                no_bid=50,
                no_ask=54,
                volume=8000,
                status="active",
            ),
            "sportsbook": SportsbookConsensus(
                event_id="test2",
                team="Green Bay Packers",
                sport="nfl",
                consensus_prob=55,
                book_probs={"draftkings": 54, "fanduel": 55, "betmgm": 56, "caesars": 55},
                book_count=4,
                book_spread=2.0,
                updated_at=datetime.now(timezone.utc),
            ),
            "hours": 24,
        },
        {
            "name": "Large Edge (10c)",
            "kalshi": KalshiMarket(
                ticker="KXNFL-26JAN11-KC",
                title="Kansas City Chiefs to beat Denver Broncos",
                yes_bid=55,
                yes_ask=58,  # 10c edge vs 68c consensus
                no_bid=42,
                no_ask=45,
                volume=15000,
                status="active",
            ),
            "sportsbook": SportsbookConsensus(
                event_id="test3",
                team="Kansas City Chiefs",
                sport="nfl",
                consensus_prob=68,
                book_probs={"draftkings": 68, "fanduel": 67, "betmgm": 69, "caesars": 68},
                book_count=4,
                book_spread=2.0,
                updated_at=datetime.now(timezone.utc),
            ),
            "hours": 6,
        },
    ]

    profiles_to_test = ["CONSERVATIVE", "STANDARD", "AGGRESSIVE", "ANY_EDGE"]

    print("PROFILE COMPARISON")
    print("-" * 70)
    print(f"{'Profile':<18} {'2c Edge':<12} {'5c Edge':<12} {'10c Edge':<12}")
    print("-" * 70)

    results = {}
    for profile in profiles_to_test:
        detector = EdgeDetector(profile=profile, bankroll=10000)
        results[profile] = []

        for tc in test_cases:
            signal = detector.detect(tc["kalshi"], tc["sportsbook"], tc["hours"])
            results[profile].append(signal.should_trade)

    for profile in profiles_to_test:
        row = f"{profile:<18}"
        for should_trade in results[profile]:
            status = "TRADE" if should_trade else "SKIP"
            row += f" {status:<12}"
        print(row)

    print("-" * 70)
    print()

    # Show detailed output for ANY_EDGE profile with small edge
    print("=" * 70)
    print("DETAILED: ANY_EDGE Profile with 2c Edge")
    print("=" * 70)
    print()

    detector = EdgeDetector(profile="ANY_EDGE", bankroll=10000)
    tc = test_cases[0]  # 2c edge
    signal = detector.detect(tc["kalshi"], tc["sportsbook"], tc["hours"])

    print(f"Profile: {detector.profile_name}")
    print(f"Kelly Fraction: {detector.kelly_fraction}")
    print(f"Confidence Thresholds: {detector.confidence_thresholds}")
    print()

    print(f"Edge: {signal.edge.best_edge}c on {signal.edge.best_side.upper()} side")
    print(f"Action: {signal.action}")
    print(f"Should Trade: {signal.should_trade}")
    print(f"Confidence: {signal.confidence_score} ({signal.confidence_tier})")
    print()

    print("Condition Checks:")
    for check in signal.condition_checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"  [{status}] {check.name}: required {check.required}, actual {check.actual}")
    print()

    if signal.should_trade:
        print(f"Position: {signal.recommended_contracts} contracts @ max {signal.max_price}c")
        print(f"Risk: ${signal.risk_amount:.2f} | Potential Profit: ${signal.potential_profit:.2f}")

    print()
    print("=" * 70)
    print("PROFILE SETTINGS SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Profile':<18} {'Min Edge':<10} {'Min Vol':<10} {'Kelly':<8} {'Price Range'}")
    print("-" * 70)
    for profile_name in PROFILES:
        p = PROFILES[profile_name]
        print(f"{profile_name:<18} {p['min_edge_cents']}c{'':<7} {p['min_kalshi_volume']:<10} {p['kelly_fraction']:<8} {p['min_price']}-{p['max_price']}c")

    return signal


if __name__ == "__main__":
    validate_detector()
