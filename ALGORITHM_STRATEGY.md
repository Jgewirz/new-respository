# Kalshi vs Sportsbook Edge Detection Algorithm

## Executive Summary

This document defines the optimal algorithm for capturing positive expected value (EV) by exploiting pricing inefficiencies between Kalshi prediction markets and sharp sportsbook consensus.

**Core Premise:** Sportsbooks (DraftKings, FanDuel, BetMGM) are the "sharp" market with professional line-setters and massive liquidity. Kalshi is a retail-dominated prediction market with slower price discovery. When these prices diverge significantly, we bet on Kalshi in the direction indicated by sportsbook consensus.

---

## Market Analysis Findings

### 1. Sportsbook Consensus Quality

From live data analysis:
```
Team                    | Consensus | Book Spread | Confidence
Los Angeles Rams        | 86.3c     | 1.3c        | HIGH
Green Bay Packers       | 54.3c     | 0.8c        | VERY HIGH
Chicago Bears           | 50.2c     | 0.5c        | VERY HIGH
Pittsburgh Steelers     | 42.0c     | 2.7c        | MEDIUM
```

**Key Finding:** When major books agree within 2c, the "true" probability is well-established. This is our signal.

### 2. Kalshi Market Efficiency

```
Spread Category    | Markets | Avg Volume | Implication
Tight (≤3c)        | 1,668   | 78,873     | Efficient - NO EDGE
Normal (4-6c)      | 886     | 1,434      | Target Zone
Wide (7-10c)       | 318     | 196        | Edge exists, execution risk
Very Wide (>10c)   | 824     | 77         | Illiquid - SKIP
```

**Key Finding:** Target markets with 4-10c Kalshi spread and volume > 500.

### 3. Price Level Efficiency

```
Price Bucket       | Kalshi Spread | Strategy
0-20c (longshot)   | 3.1c          | Efficient - avoid
21-40c (underdog)  | 8.5c          | Moderate opportunity
41-60c (tossup)    | 10.9c         | BEST opportunity
61-80c (favorite)  | 16.1c         | Good opportunity
81-100c (heavy fav)| 52.4c         | Very inefficient but risky
```

**Key Finding:** Best opportunities are in the 41-80c range where Kalshi is least efficient.

---

## The Algorithm

### Phase 1: Signal Detection

**Trigger Conditions (ALL must be TRUE):**

```python
ENTRY_CONDITIONS = {
    # Edge Requirements
    "min_edge_cents": 5,              # Minimum 5c edge (sportsbook - kalshi)
    "min_edge_high_conf": 3,          # 3c edge OK if 4/4 books agree

    # Book Consensus Requirements
    "min_books": 3,                   # At least 3 of 4 target books
    "max_book_spread": 3.0,           # Books must agree within 3c
    "target_books": ["draftkings", "fanduel", "betmgm", "caesars"],

    # Kalshi Market Requirements
    "max_kalshi_spread": 10,          # Kalshi bid-ask ≤ 10c
    "min_kalshi_volume": 500,         # Minimum volume for liquidity
    "min_kalshi_liquidity": 1000,     # Minimum $ at target price

    # Timing Requirements
    "min_hours_to_event": 2,          # Not too close to game
    "max_hours_to_event": 48,         # Not too far (lines change)

    # Price Level Filter
    "target_price_range": (25, 85),   # Avoid extremes (longshots/locks)
}
```

### Phase 2: Edge Calculation

```python
def calculate_edge(kalshi_yes_ask, kalshi_no_ask, sportsbook_consensus):
    """
    Calculate edge in cents.

    Positive edge on YES = sportsbook thinks YES is more likely than Kalshi price
    Positive edge on NO = sportsbook thinks NO is more likely than Kalshi price
    """
    # Sportsbook says this team has X% chance
    book_yes_prob = sportsbook_consensus  # e.g., 55c = 55%
    book_no_prob = 100 - sportsbook_consensus  # e.g., 45c = 45%

    # Edge = what books think - what Kalshi is charging
    yes_edge = book_yes_prob - kalshi_yes_ask  # Buy YES if positive
    no_edge = book_no_prob - kalshi_no_ask     # Buy NO if positive

    return {
        "yes_edge": yes_edge,
        "no_edge": no_edge,
        "best_side": "yes" if yes_edge > no_edge else "no",
        "best_edge": max(yes_edge, no_edge),
    }
```

**Example:**
```
Sportsbook consensus: 55c (55% for Team A)
Kalshi YES ask: 48c
Kalshi NO ask: 54c

YES edge = 55 - 48 = +7c  ← BUY YES
NO edge = 45 - 54 = -9c   ← Don't buy NO

Action: BUY YES at 48c, expected value = 7c per contract
```

### Phase 3: Confidence Scoring

```python
def calculate_confidence(edge, book_count, book_spread, kalshi_volume, hours_to_event):
    """
    Score opportunity from 0-100.

    Higher score = more confident in the edge.
    """
    score = 0

    # Edge size (0-30 points)
    if edge >= 10: score += 30
    elif edge >= 7: score += 25
    elif edge >= 5: score += 20
    elif edge >= 3: score += 10

    # Book consensus (0-30 points)
    if book_count == 4 and book_spread <= 1: score += 30
    elif book_count >= 3 and book_spread <= 2: score += 25
    elif book_count >= 3 and book_spread <= 3: score += 15

    # Kalshi liquidity (0-20 points)
    if kalshi_volume >= 5000: score += 20
    elif kalshi_volume >= 2000: score += 15
    elif kalshi_volume >= 500: score += 10

    # Timing (0-20 points)
    if 6 <= hours_to_event <= 24: score += 20  # Sweet spot
    elif 2 <= hours_to_event <= 48: score += 10

    return score

# Confidence tiers
CONFIDENCE_TIERS = {
    "HIGH": 70,      # Score >= 70: Full position
    "MEDIUM": 50,    # Score >= 50: Half position
    "LOW": 30,       # Score >= 30: Quarter position
    "SKIP": 0,       # Score < 30: No trade
}
```

### Phase 4: Position Sizing

```python
def calculate_position_size(edge, confidence_score, bankroll, max_position):
    """
    Modified Kelly Criterion for position sizing.

    Kelly optimal: f* = edge / odds
    We use fractional Kelly (25%) for safety.
    """
    # Convert edge to win probability estimate
    # If we're buying at 48c with 7c edge, fair value is 55c
    fair_value = kalshi_price + edge

    # Kelly fraction
    kelly_fraction = 0.25  # Conservative

    # Expected return per dollar risked
    if buying_yes:
        # Pay kalshi_price, win (100 - kalshi_price) if right
        win_payout = 100 - kalshi_price
        p_win = fair_value / 100
        p_lose = 1 - p_win
        kelly = (p_win * win_payout - p_lose * kalshi_price) / win_payout

    # Position size
    raw_size = bankroll * kelly * kelly_fraction

    # Apply confidence multiplier
    if confidence_score >= 70:
        size = raw_size * 1.0
    elif confidence_score >= 50:
        size = raw_size * 0.5
    else:
        size = raw_size * 0.25

    # Apply max position constraint
    return min(size, max_position)
```

### Phase 5: Execution Rules

```python
EXECUTION_RULES = {
    # Order type
    "order_type": "limit",           # Never market orders
    "limit_offset": 1,               # Bid 1c better than current ask

    # Execution timing
    "max_wait_seconds": 60,          # Cancel if not filled in 60s
    "retry_with_worse_price": True,  # Retry at +1c if not filled
    "max_retries": 3,

    # Slippage protection
    "max_slippage_cents": 2,         # Abort if price moves > 2c

    # Partial fills
    "accept_partial": True,
    "min_partial_pct": 50,           # Need at least 50% filled
}
```

### Phase 6: Exit & Risk Management

```python
EXIT_CONDITIONS = {
    # Take profit (edge collapses)
    "exit_if_edge_below": 1,         # Exit if edge shrinks to < 1c

    # Stop loss
    "max_adverse_move": 8,           # Exit if Kalshi moves 8c against us

    # Time-based
    "exit_before_event_hours": 0.5,  # Exit 30min before game if no hedge

    # Sportsbook movement
    "exit_if_books_move_toward_kalshi": True,  # Books moving = we're wrong
}

RISK_LIMITS = {
    # Position limits
    "max_position_per_event": 500,   # $500 max per game
    "max_daily_exposure": 5000,      # $5000 max daily

    # Loss limits
    "max_daily_loss": 1000,          # Stop trading if down $1000
    "max_weekly_loss": 3000,         # Stop if down $3000/week

    # Concentration limits
    "max_pct_bankroll_per_trade": 5, # No more than 5% per trade
    "max_correlated_exposure": 20,   # No more than 20% in same sport
}
```

---

## Complete Signal Output Format

When the algorithm detects an opportunity, output:

```json
{
    "signal": {
        "id": "sig_20260108_nfl_buf_jax_yes",
        "timestamp": "2026-01-08T16:30:00Z",
        "action": "BUY_YES",

        "kalshi": {
            "ticker": "KXNFL-26JAN11-BUF",
            "title": "Buffalo Bills to beat Jacksonville Jaguars",
            "yes_ask": 48,
            "no_ask": 54,
            "spread": 6,
            "volume": 12500
        },

        "sportsbook": {
            "consensus": 55,
            "books": {
                "draftkings": 53,
                "fanduel": 55,
                "betmgm": 57,
                "caesars": 55
            },
            "book_count": 4,
            "book_spread": 4
        },

        "edge": {
            "cents": 7,
            "percent": 14.6,
            "side": "yes"
        },

        "confidence": {
            "score": 75,
            "tier": "HIGH",
            "reasons": [
                "4/4 books reporting",
                "Book spread only 4c",
                "Volume > 10000",
                "12 hours to event"
            ]
        },

        "position": {
            "recommended_size": 25,
            "max_price": 50,
            "risk_amount": 1200,
            "potential_profit": 1300
        },

        "timing": {
            "event_start": "2026-01-11T18:00:00Z",
            "hours_to_event": 12.5,
            "execute_by": "2026-01-08T17:00:00Z"
        }
    },

    "validation": {
        "all_conditions_met": true,
        "conditions": {
            "min_edge": {"required": 5, "actual": 7, "pass": true},
            "min_books": {"required": 3, "actual": 4, "pass": true},
            "max_book_spread": {"required": 3, "actual": 4, "pass": false},
            "max_kalshi_spread": {"required": 10, "actual": 6, "pass": true},
            "min_volume": {"required": 500, "actual": 12500, "pass": true},
            "price_range": {"required": "25-85", "actual": 48, "pass": true},
            "timing": {"required": "2-48h", "actual": 12.5, "pass": true}
        }
    }
}
```

---

## Implementation Priority

### P0: Core Detection (Day 1)
1. Real-time sportsbook polling (every 30s)
2. Kalshi price fetch (every 10s)
3. Edge calculation
4. Basic filtering (min edge, min books)

### P1: Confidence Scoring (Day 2)
1. Book consensus quality scoring
2. Kalshi liquidity scoring
3. Timing scoring
4. Combined confidence tier

### P2: Position Sizing (Day 3)
1. Kelly criterion implementation
2. Fractional Kelly with confidence adjustment
3. Risk limit enforcement

### P3: Execution (Day 4-5)
1. Kalshi API integration
2. Limit order placement
3. Fill monitoring
4. Retry logic

### P4: Risk Management (Day 6-7)
1. Real-time P&L tracking
2. Exit condition monitoring
3. Daily/weekly limit enforcement
4. Alert system

---

## Expected Performance

Based on market analysis:

| Metric | Conservative | Expected | Optimistic |
|--------|-------------|----------|------------|
| Signals per day | 2-3 | 5-8 | 10-15 |
| Win rate | 55% | 58% | 62% |
| Avg edge | 5c | 6c | 8c |
| Avg position | $100 | $200 | $300 |
| Daily P&L | +$10 | +$50 | +$150 |
| Monthly P&L | +$200 | +$1,000 | +$3,000 |

**Key Assumption:** Sportsbook consensus is a reliable predictor of true probability. Historical data suggests this is true for major US sports with 4+ books reporting.

---

## Risk Factors

1. **Model Risk:** Sportsbooks could be wrong (upsets happen)
2. **Execution Risk:** Can't get filled at target price
3. **Liquidity Risk:** Kalshi liquidity dries up
4. **API Risk:** Rate limits, downtime
5. **Regulatory Risk:** Kalshi rule changes

**Mitigation:** Fractional Kelly sizing, strict position limits, diversification across multiple games.
