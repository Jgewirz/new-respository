# Kalshi Arbitrage System - Complete Architecture Documentation

**Generated:** 2026-01-12
**Status:** PRODUCTION RUNNING
**Version:** 1.0

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Currently Running Containers](#2-currently-running-containers)
3. [Data Flow Architecture](#3-data-flow-architecture)
4. [File-by-File Breakdown](#4-file-by-file-breakdown)
5. [Authentication & Connectivity](#5-authentication--connectivity)
6. [WebSocket Real-Time Pipeline](#6-websocket-real-time-pipeline)
7. [Redis Caching Layer](#7-redis-caching-layer)
8. [Edge Detection & Signal Generation](#8-edge-detection--signal-generation)
9. [Position Entry Logic](#9-position-entry-logic)
10. [Position Exit Logic](#10-position-exit-logic)
11. [Risk Controls & Circuit Breaker](#11-risk-controls--circuit-breaker)
12. [Latency Analysis](#12-latency-analysis)
13. [QuestDB Time-Series Storage](#13-questdb-time-series-storage)
14. [Configuration Reference](#14-configuration-reference)

---

## 1. System Overview

This is a **Kalshi-first arbitrage detection system** that identifies pricing discrepancies between Kalshi prediction markets and traditional sportsbooks (DraftKings, FanDuel, BetMGM, Caesars).

### Core Concept

```
If Kalshi says: "Bills win" = 48¢ (48% implied)
And Sportsbooks say: "Bills win" = 55% consensus

Edge = 55% - 48% = 7¢ profit opportunity
Action: BUY YES @ 48¢, expect settlement at 55¢+
```

### Architecture Philosophy

- **Kalshi-First:** All flows originate from Kalshi markets
- **Sub-Millisecond Latency:** Redis hot cache for instant lookups
- **Real-Time Detection:** WebSocket-triggered per-tick analysis
- **Risk-First:** Circuit breaker prevents catastrophic losses
- **Paper-First:** Paper trading by default for validation

---

## 2. Currently Running Containers

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                         DOCKER CONTAINER STATUS                           ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║   CONTAINER         PORT(S)              FUNCTION                         ║
║   ─────────         ───────              ────────                         ║
║   arb-ws-consumer   (internal)           WebSocket data streaming         ║
║   arb-odds-ingest   (internal)           Sportsbook odds polling          ║
║   arb-postgres      5432                 Order/position persistence       ║
║   arb-redis         6379                 Hot cache (sub-ms lookups)       ║
║   arb-questdb       9000, 9009, 8812     Time-series storage              ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

### Container Responsibilities

| Container | Image | Purpose |
|-----------|-------|---------|
| `arb-ws-consumer` | Custom Python | Connects to Kalshi WebSocket, streams ticker/trade data |
| `arb-odds-ingest` | Custom Python | Polls The Odds API, calculates consensus, writes to Redis |
| `arb-postgres` | postgres:16-alpine | Stores orders, positions, configuration |
| `arb-redis` | redis:7-alpine | Hot cache for sub-ms consensus lookups |
| `arb-questdb` | questdb/questdb:8.2.1 | Time-series DB for ticks, trades, signals |

---

## 3. Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EXTERNAL DATA SOURCES                             │
└─────────────────────────────────────────────────────────────────────────────┘
          │                                              │
          │                                              │
          ▼                                              ▼
┌─────────────────────┐                      ┌─────────────────────┐
│    KALSHI API       │                      │   THE ODDS API      │
│                     │                      │                     │
│  • REST: Markets    │                      │  • DraftKings       │
│  • WebSocket: Live  │                      │  • FanDuel          │
│    - Tickers        │                      │  • BetMGM           │
│    - Trades         │                      │  • Caesars          │
│    - Orderbook      │                      │                     │
└──────────┬──────────┘                      └──────────┬──────────┘
           │                                            │
═══════════╪════════════════════════════════════════════╪═══════════════════
           │              INGESTION LAYER               │
═══════════╪════════════════════════════════════════════╪═══════════════════
           │                                            │
           ▼                                            ▼
┌─────────────────────────────────┐      ┌─────────────────────────────────┐
│     arb-ws-consumer             │      │     arb-odds-ingest             │
│     ws_consumer.py              │      │     odds_ingest.py              │
│                                 │      │                                 │
│  1. RSA-PSS Authentication      │      │  1. Poll every 30 seconds       │
│  2. Discover sports markets     │      │  2. Calculate consensus prob    │
│  3. Subscribe to channels       │      │  3. Write to Redis cache        │
│  4. Stream ticker/trade msgs    │      │  4. Write to QuestDB history    │
│                                 │      │                                 │
│  Output: Raw market data        │      │  Output: Consensus probabilities│
└──────────────┬──────────────────┘      └──────────────┬──────────────────┘
               │                                        │
               │                                        │
               ▼                                        │
┌─────────────────────────────────┐                    │
│     ws_processor.py             │                    │
│                                 │                    │
│  Routes messages to:            │                    │
│  • QuestDB (ILP writes)         │                    │
│  • Redis (hot cache)            │                    │
│  • Callbacks (detector)         │                    │
│  • TakeProfitMonitor            │                    │
└──────────────┬──────────────────┘                    │
               │                                        │
═══════════════╪════════════════════════════════════════╪═══════════════════
               │              STORAGE LAYER             │
═══════════════╪════════════════════════════════════════╪═══════════════════
               │                                        │
               ▼                                        ▼
┌─────────────────────────────────┐      ┌─────────────────────────────────┐
│         arb-redis               │      │         arb-questdb             │
│         (Hot Cache)             │      │         (Cold Storage)          │
│                                 │      │                                 │
│  kalshi:m:{ticker} → prices     │◄─────│  kalshi_ticks table             │
│  odds:consensus:{event}:{team}  │      │  kalshi_trades table            │
│  positions:active:{ticker}      │      │  sportsbook_odds table          │
│                                 │      │  arb_opportunities table        │
│  Access: < 1ms                  │      │  Access: < 100ms                │
└──────────────┬──────────────────┘      └─────────────────────────────────┘
               │
═══════════════╪═══════════════════════════════════════════════════════════
               │              DETECTION LAYER
═══════════════╪═══════════════════════════════════════════════════════════
               │
               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      realtime_detector.py                               │
│                                                                         │
│  On each ticker update:                                                 │
│                                                                         │
│  1. Rate limit check (100ms per ticker) ─────────────────► Skip if     │
│                                                              too recent │
│  2. Update cached market with WebSocket prices                          │
│                                                                         │
│  3. Resolve Kalshi → Sportsbook event ──────────────────► resolver.py  │
│     (RapidFuzz fuzzy team matching)                                     │
│                                                                         │
│  4. Lookup consensus from Redis ────────────────────────► odds:consensus│
│                                                                         │
│  5. Run EdgeDetector.detect() ──────────────────────────► detector.py  │
│     • Calculate yes_edge, no_edge                                       │
│     • Check entry conditions (profile-dependent)                        │
│     • Score confidence (HIGH/MEDIUM/LOW/SKIP)                          │
│     • Calculate Kelly position sizing                                   │
│                                                                         │
│  6. Check CircuitBreaker ───────────────────────────────► If blocked,  │
│                                                              skip trade │
│  7. Fire callback with Signal                                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
               │
               │ Signal (BUY_YES, BUY_NO, or NO_TRADE)
               │
═══════════════╪═══════════════════════════════════════════════════════════
               │              EXECUTION LAYER
═══════════════╪═══════════════════════════════════════════════════════════
               │
               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      circuit_breaker.py                                 │
│                                                                         │
│  Pre-trade validation:                                                  │
│  • Position size ≤ 100 contracts                                        │
│  • Risk per trade ≤ $5.00                                               │
│  • Daily loss ≤ $100                                                    │
│  • Daily trades ≤ 500                                                   │
│  • Consecutive losses < 100                                             │
│  • Event concentration ≤ 5,000 contracts                                │
│  • Sport exposure ≤ 30% bankroll                                        │
│  • Rate limit: 20 orders/minute                                         │
│                                                                         │
│  If ANY check fails: Trade BLOCKED                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
               │
               │ If allowed
               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      Kalshi Order Submission                            │
│                      client.py                                          │
│                                                                         │
│  1. Create ExecutionOrder from Signal                                   │
│  2. Generate UUID for idempotency                                       │
│  3. POST /portfolio/orders with:                                        │
│     {                                                                   │
│       "ticker": "KXNFL-26JAN11-BUF",                                   │
│       "side": "yes",                                                    │
│       "action": "buy",                                                  │
│       "count": 25,                                                      │
│       "type": "limit",                                                  │
│       "yes_price": 48                                                   │
│     }                                                                   │
│  4. Monitor fill status                                                 │
│  5. On fill: Register with TakeProfitMonitor                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
               │
               │ Position registered
               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      take_profit.py                                     │
│                                                                         │
│  Real-time exit monitoring:                                             │
│                                                                         │
│  On each WebSocket tick:                                                │
│  • Update position's current_bid                                        │
│  • Check: current_bid ≥ take_profit_price? → EXIT (take profit)        │
│  • Check: current_bid ≤ stop_loss_price? → EXIT (stop loss)            │
│                                                                         │
│  On exit signal:                                                        │
│  • Submit SELL order via KalshiClient                                   │
│  • Record realized P&L                                                  │
│  • Update CircuitBreaker stats                                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. File-by-File Breakdown

### Connectors (`app/connectors/kalshi/`)

| File | Purpose | Key Classes |
|------|---------|-------------|
| `auth.py` | RSA-PSS authentication | `KalshiAuth`, `AuthHeaders` |
| `client.py` | REST API client | `KalshiClient`, `RateLimiter` |
| `ws_consumer.py` | WebSocket connection & streaming | `KalshiWebSocketConsumer` |
| `ws_processor.py` | Route messages to storage | `WebSocketProcessor` |

### Services (`app/services/`)

| File | Purpose | Key Classes |
|------|---------|-------------|
| `realtime_detector.py` | Per-tick edge detection | `RealtimeDetector` |
| `odds_ingest.py` | Sportsbook consensus pipeline | `OddsIngestionPipeline` |
| `arb_pipeline.py` | Batch detection orchestration | `ArbPipeline` |

### Arbitrage (`app/arb/`)

| File | Purpose | Key Classes |
|------|---------|-------------|
| `detector.py` | Edge calculation & signal generation | `EdgeDetector`, `Signal` |
| `optimal_exit.py` | Dynamic TP/SL calibration | Exit strategies by edge |

### Execution (`app/execution/`)

| File | Purpose | Key Classes |
|------|---------|-------------|
| `circuit_breaker.py` | Risk controls & limits | `CircuitBreaker`, `RiskLimits` |
| `position_store.py` | Track open positions | `PositionStore`, `TrackedPosition` |
| `take_profit.py` | Real-time exit monitoring | `TakeProfitMonitor`, `ExitSignal` |
| `models.py` | Order lifecycle models | `ExecutionOrder`, `Fill` |

### Mapping (`app/mapping/`)

| File | Purpose | Key Classes |
|------|---------|-------------|
| `resolver.py` | Kalshi ↔ Sportsbook event mapping | `EventResolver`, `MappedEvent` |

### Data (`app/data/`)

| File | Purpose | Key Classes |
|------|---------|-------------|
| `questdb.py` | Time-series storage | `QuestDBILPClient`, `QuestDBClient` |

---

## 5. Authentication & Connectivity

### RSA-PSS Signature Flow

```python
# Message format for signing
message = f"{timestamp_ms}{METHOD}{path}"

# Example for WebSocket:
message = "1768235838794GET/trade-api/ws/v2"

# Signing parameters:
# - Algorithm: RSA-PSS
# - Hash: SHA256
# - MGF: MGF1 with SHA256
# - Salt length: PSS.MAX_LENGTH
```

### Environment Variables

```bash
KALSHI_KEY_ID=5d239a9f-f225-4732-a0ed-9728f2c6c4eb
KALSHI_PRIVATE_KEY_PATH=/app/keys/kalshi_demo_private_key.pem
KALSHI_ENV=demo  # or "prod"
```

### Connection Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Timeout | 5s | Request timeout |
| Max retries | 3 | Exponential backoff |
| Backoff base | 100ms | First retry delay |
| Backoff max | 2s | Maximum retry delay |
| Pool keepalive | 5 | Connection pool size |
| Pool max | 10 | Maximum connections |

---

## 6. WebSocket Real-Time Pipeline

### Message Types Processed

```python
# Ticker update (most frequent)
{
    "type": "ticker",
    "msg": {
        "market_ticker": "KXNBA-26-NYK",
        "yes_bid": 6,
        "yes_ask": 7,
        "no_bid": 93,
        "no_ask": 94,
        "volume": 5359720,
        "ts": 1768235835959
    }
}

# Trade execution
{
    "type": "trade",
    "msg": {
        "market_ticker": "KXNHL-26-MIN",
        "count": 17,
        "yes_price": 6,
        "taker_side": "yes",
        "ts": 1768235835963
    }
}
```

### Processing Pipeline

```
WebSocket Message
       │
       ▼
┌──────────────────┐
│ ws_processor.py  │
│                  │
│ 1. Parse JSON    │
│ 2. Route by type │
└──────────────────┘
       │
       ├─────────────────────────────────┐
       │                                 │
       ▼                                 ▼
┌──────────────────┐           ┌──────────────────┐
│ QuestDB ILP      │           │ Redis Cache      │
│                  │           │                  │
│ kalshi_ticks     │           │ kalshi:m:{ticker}│
│ kalshi_trades    │           │                  │
└──────────────────┘           └──────────────────┘
       │
       │
       ▼
┌──────────────────┐
│ Callbacks        │
│                  │
│ • on_ticker()    │──────► RealtimeDetector
│ • on_trade()     │
│ • on_orderbook() │
└──────────────────┘
```

---

## 7. Redis Caching Layer

### Key Schema

```
SPORTSBOOK CONSENSUS (from odds_ingest.py):
─────────────────────────────────────────────
odds:consensus:{event_id}:{team}  →  String (cents 0-100)
  Example: odds:consensus:abc123:buffalo_bills → "55"

odds:{sport}:{event_id}:{team}    →  Hash (full odds)
  Example: odds:nfl:abc123:buffalo_bills → {
    "draftkings": "53",
    "fanduel": "56",
    "betmgm": "54",
    "caesars": "57",
    "consensus": "55",
    "book_count": "4",
    "updated_at": "1768235800000"
  }


KALSHI MARKETS (from ws_processor.py):
──────────────────────────────────────
kalshi:m:{ticker}  →  Hash
  Example: kalshi:m:KXNFL-26JAN11-BUF → {
    "yes_bid": "48",
    "yes_ask": "49",
    "no_bid": "51",
    "no_ask": "52",
    "volume": "1234567",
    "last_price": "48",
    "updated_at": "1768235838794"
  }

kalshi:markets  →  Set (all active tickers)


POSITION TRACKING (from position_store.py):
───────────────────────────────────────────
positions:active:{ticker}  →  Hash
  Example: positions:active:KXNFL-26JAN11-BUF → {
    "entry_price": "48",
    "contracts": "25",
    "side": "yes",
    "take_profit": "55",
    "stop_loss": "42",
    "entry_time": "1768235800000"
  }

positions:tickers  →  Set (tracked tickers)


INDEXES:
────────
odds:events:{sport}  →  Set (event IDs)
odds:events:all      →  Set (all events)
odds:last_update     →  String (timestamp)
```

### TTL Strategy

| Key Pattern | TTL | Reason |
|-------------|-----|--------|
| `odds:consensus:*` | 2 hours | Games can be delayed |
| `odds:*:*:*` | 2 hours | Full odds data |
| `kalshi:m:*` | 1 hour | Market snapshots |
| `positions:*` | 24 hours | Active positions |
| Indexes | Persistent | Until invalidated |

### Access Patterns

```python
# Consensus lookup (during detection) - <1ms
consensus = redis.get(f"odds:consensus:{event_id}:{team}")

# Market lookup - <1ms
market = redis.hgetall(f"kalshi:m:{ticker}")

# Batch pipeline for writes
pipe = redis.pipeline()
pipe.hset(key, mapping=data)
pipe.expire(key, TTL)
pipe.sadd(index_set, ticker)
pipe.execute()  # Atomic batch
```

---

## 8. Edge Detection & Signal Generation

### Trading Profiles

| Profile | Min Edge | Books | Max Spread | Kelly | Use Case |
|---------|----------|-------|------------|-------|----------|
| **CONSERVATIVE** | 6¢ | 3+ | 2.5¢ | 0.15 | Low risk, proven edges |
| **STANDARD** | 5¢ | 3+ | 3.0¢ | 0.25 | Balanced approach |
| **AGGRESSIVE** | 3¢ | 3+ | 4.0¢ | 0.35 | Higher volume |
| **VERY_AGGRESSIVE** | 2¢ | 2+ | 5.0¢ | 0.50 | Maximum opportunity |
| **ANY_EDGE** | 1¢ | 2+ | 10.0¢ | 0.50 | Development/testing |
| **SCALP** | 1¢ | 3+ | 2.0¢ | 0.75 | Quick in/out |

### Edge Calculation

```python
# Given:
kalshi_yes_ask = 48    # Cost to buy YES
kalshi_no_ask = 52     # Cost to buy NO
sportsbook_consensus = 55  # Implied probability (cents)

# Calculate edges:
yes_edge = sportsbook_consensus - kalshi_yes_ask
         = 55 - 48 = 7¢  ← PROFITABLE

no_edge = (100 - sportsbook_consensus) - kalshi_no_ask
        = (100 - 55) - 52 = -7¢  ← NOT PROFITABLE

# Best trade:
best_side = "yes"
best_edge = 7¢
edge_percent = (7 / 48) * 100 = 14.6%
```

### Entry Condition Checks

```python
conditions = [
    ("min_edge", best_edge >= profile.min_edge_cents),
    ("book_count", book_count >= profile.min_books),
    ("book_spread", book_spread <= profile.max_book_spread),
    ("kalshi_spread", kalshi_spread <= profile.max_kalshi_spread),
    ("volume", volume >= profile.min_volume),
    ("hours_to_event", min_hours <= hours <= max_hours),
    ("price_bounds", min_price <= price <= max_price),
]

# All conditions must pass for should_trade = True
```

### Confidence Scoring

```python
# Factors that increase confidence:
+20  # 4+ books agree
+15  # Book spread < 2¢
+10  # High volume (>5000)
+10  # Event within 24 hours
-10  # Extreme price (<10¢ or >90¢)
-15  # Low book count (2 books)

# Thresholds:
HIGH   = score >= 70
MEDIUM = score >= 50
LOW    = score >= 30
SKIP   = score < 30
```

### Signal Output

```python
Signal(
    action = "BUY_YES",           # or "BUY_NO", "NO_TRADE"
    should_trade = True,          # All conditions met

    # Market data
    kalshi = KalshiMarket(...),
    sportsbook = SportsbookConsensus(...),
    edge = EdgeCalculation(
        yes_edge = 7,
        no_edge = -7,
        best_side = "yes",
        best_edge = 7,
    ),

    # Confidence
    confidence_score = 75,
    confidence_tier = "HIGH",

    # Sizing
    recommended_contracts = 25,
    max_price = 49,              # Don't pay more than this
    risk_amount = 12.00,         # $ at risk
    potential_profit = 1.75,     # Expected $ profit
)
```

---

## 9. Position Entry Logic

### Entry Flow

```
Signal Generated
       │
       ▼
┌──────────────────────────────────────┐
│ 1. CIRCUIT BREAKER PRE-CHECK         │
│                                      │
│    check_trade(                      │
│        ticker="KXNFL-26JAN11-BUF",  │
│        contracts=25,                 │
│        risk_cents=1200,              │
│        sport="nfl"                   │
│    )                                 │
│                                      │
│    Validates:                        │
│    • Position size ≤ limit           │
│    • Risk amount ≤ limit             │
│    • Daily loss ≤ limit              │
│    • Event concentration ≤ limit     │
│    • Rate limit not exceeded         │
│                                      │
│    If BLOCKED → Skip trade           │
└──────────────────────────────────────┘
       │
       │ Allowed
       ▼
┌──────────────────────────────────────┐
│ 2. CREATE EXECUTION ORDER            │
│                                      │
│    ExecutionOrder(                   │
│        execution_id = uuid4(),       │
│        client_order_id = uuid4(),    │ ← Kalshi idempotency
│        ticker = "KXNFL-26JAN11-BUF", │
│        side = "yes",                 │
│        action = "buy",               │
│        order_type = "limit",         │
│        limit_price = 48,             │
│        contracts = 25,               │
│        state = PENDING,              │
│        mode = PAPER or LIVE,         │
│    )                                 │
└──────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│ 3. SUBMIT TO KALSHI                  │
│                                      │
│    POST /portfolio/orders            │
│    {                                 │
│        "ticker": "KXNFL-26JAN11-BUF",│
│        "client_order_id": "uuid...", │
│        "side": "yes",                │
│        "action": "buy",              │
│        "count": 25,                  │
│        "type": "limit",              │
│        "yes_price": 48               │
│    }                                 │
│                                      │
│    State: PENDING → SUBMITTED        │
└──────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│ 4. MONITOR FILLS                     │
│                                      │
│    Poll GET /portfolio/orders/{id}   │
│    or WebSocket order updates        │
│                                      │
│    States:                           │
│    SUBMITTED → RESTING → FILLED      │
│                       → PARTIAL_FILL │
│                       → CANCELED     │
│                                      │
│    Record each fill:                 │
│    Fill(price=48, count=10, ts=...)  │
│    Fill(price=48, count=15, ts=...)  │
└──────────────────────────────────────┘
       │
       │ Fully filled
       ▼
┌──────────────────────────────────────┐
│ 5. REGISTER WITH TAKE-PROFIT         │
│                                      │
│    position_store.add_position(      │
│        TrackedPosition(              │
│            ticker = ticker,          │
│            entry_price = 48,         │
│            contracts = 25,           │
│            side = "yes",             │
│            take_profit = 55,         │ ← From optimal_exit
│            stop_loss = 42,           │
│            entry_time = now(),       │
│        )                             │
│    )                                 │
│                                      │
│    take_profit_monitor.register(     │
│        ticker, entry_price,          │
│        take_profit, stop_loss        │
│    )                                 │
└──────────────────────────────────────┘
```

---

## 10. Position Exit Logic

### Exit Triggers

| Trigger | Condition | Action |
|---------|-----------|--------|
| **TAKE_PROFIT** | `current_bid >= take_profit_price` | Sell at profit |
| **STOP_LOSS** | `current_bid <= stop_loss_price` | Sell to limit loss |
| **TIME_DECAY** | Position held > max_hold_time | Close before settlement |
| **MANUAL** | User-initiated | Immediate close |

### Exit Flow

```
WebSocket Ticker Update
       │
       ▼
┌──────────────────────────────────────┐
│ TakeProfitMonitor.on_price_update()  │
│                                      │
│ For each tracked position:           │
│                                      │
│ 1. Update current_bid from ticker    │
│    position.current_bid = yes_bid    │
│                                      │
│ 2. Check take-profit                 │
│    if current_bid >= take_profit:    │
│        trigger = TAKE_PROFIT         │
│                                      │
│ 3. Check stop-loss                   │
│    if current_bid <= stop_loss:      │
│        trigger = STOP_LOSS           │
│                                      │
│ 4. Calculate expected P&L            │
│    pnl = (current_bid - entry) * qty │
│                                      │
└──────────────────────────────────────┘
       │
       │ Exit triggered
       ▼
┌──────────────────────────────────────┐
│ Generate ExitSignal                  │
│                                      │
│ ExitSignal(                          │
│     ticker = "KXNFL-26JAN11-BUF",   │
│     trigger = TAKE_PROFIT,           │
│     entry_price = 48,                │
│     exit_price = 55,                 │
│     contracts = 25,                  │
│     realized_pnl = 175,  # cents     │
│     hold_time_ms = 300000,           │
│ )                                    │
└──────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│ Execute Exit Order                   │
│                                      │
│ POST /portfolio/orders               │
│ {                                    │
│     "ticker": "KXNFL-26JAN11-BUF",  │
│     "side": "yes",                   │
│     "action": "sell",                │
│     "count": 25,                     │
│     "type": "limit",                 │
│     "yes_price": 54  # Slightly below│
│ }                                    │
│                                      │
│ Or market order for urgency          │
└──────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│ Post-Exit Accounting                 │
│                                      │
│ 1. Remove from position_store        │
│ 2. Update circuit_breaker stats      │
│    • Record profit/loss              │
│    • Update daily P&L                │
│    • Reset consecutive loss counter  │
│ 3. Log to QuestDB for analytics      │
└──────────────────────────────────────┘
```

### Optimal Exit Calibration

Exit thresholds are dynamically calculated based on:

```python
# Edge-based take-profit (from optimal_exit.py)

edge_tp_map = {
    1:  (3, 2),   # 1¢ edge → TP at +3¢, SL at -2¢
    2:  (4, 3),   # 2¢ edge → TP at +4¢, SL at -3¢
    3:  (5, 3),   # 3¢ edge → TP at +5¢, SL at -3¢
    5:  (6, 4),   # 5¢ edge → TP at +6¢, SL at -4¢
    6:  (7, 4),   # 6¢+ edge → TP at +7¢, SL at -4¢
}

# Liquidity adjustment
if liquidity == HIGH:    # >10k volume
    tp_multiplier = 0.8  # Take profit sooner
    sl_multiplier = 1.2  # Wider stop loss
elif liquidity == LOW:   # <2k volume
    tp_multiplier = 1.5  # Wait for larger move
    sl_multiplier = 0.8  # Tighter stop loss

# Example:
# Entry at 48¢, edge = 5¢, HIGH liquidity
# TP = 48 + (6 * 0.8) = 52.8¢ → 53¢
# SL = 48 - (4 * 1.2) = 43.2¢ → 43¢
```

---

## 11. Risk Controls & Circuit Breaker

### Multi-Layered Protection

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CIRCUIT BREAKER STATES                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   CLOSED (Normal)                                                       │
│   │                                                                     │
│   │ Daily loss > $100                                                   │
│   │ OR consecutive losses > 100                                         │
│   │ OR manual halt()                                                    │
│   ▼                                                                     │
│   OPEN (Halted) ─────────────────────────────────────────────────────► │
│   │              30 min cooldown         resume() called                │
│   │              (or manual resume)                                     │
│   ▼                                                                     │
│   HALF_OPEN (Testing) ──────────► CLOSED (if successful)               │
│              │                                                          │
│              └──────────────────► OPEN (if failure)                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Limit Defaults

```python
RiskLimits(
    # Per-trade limits
    max_position_size=100,        # contracts
    max_risk_per_trade_cents=500, # $5.00

    # Daily limits
    max_daily_loss_cents=10000,   # $100.00
    max_trades_per_day=500,
    max_daily_volume=50000,       # contracts

    # Concentration limits
    max_contracts_per_event=5000,
    max_sport_exposure_pct=30.0,  # of bankroll

    # Rate limits
    max_orders_per_minute=20,
    min_order_interval_ms=100,

    # Loss circuit
    consecutive_loss_halt=100,
    cooldown_minutes=30,
)
```

### Block Reasons

```python
class BlockReason(Enum):
    MANUAL_HALT = "manual_halt"
    BREAKER_OPEN = "breaker_open"
    POSITION_SIZE = "position_size_exceeded"
    RISK_PER_TRADE = "risk_per_trade_exceeded"
    DAILY_LOSS = "daily_loss_exceeded"
    DAILY_TRADES = "daily_trades_exceeded"
    DAILY_VOLUME = "daily_volume_exceeded"
    CONSECUTIVE_LOSSES = "consecutive_losses"
    RATE_LIMIT = "rate_limit_exceeded"
    EVENT_CONCENTRATION = "event_concentration"
    SPORT_CONCENTRATION = "sport_concentration"
    INSUFFICIENT_BALANCE = "insufficient_balance"
```

---

## 12. Latency Analysis

### Current Performance (Measured)

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                         LATENCY BREAKDOWN                                 ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║   Component                          Measured        Target      Status   ║
║   ─────────                          ────────        ──────      ──────   ║
║   WebSocket message receive          ~0.1ms          <1ms        OK       ║
║   JSON parsing                       ~0.01ms         <1ms        OK       ║
║   Redis consensus lookup             ~0.3ms          <1ms        OK       ║
║   Edge calculation                   ~0.01ms         <1ms        OK       ║
║   Confidence scoring                 ~0.01ms         <1ms        OK       ║
║   Circuit breaker check              ~0.01ms         <1ms        OK       ║
║   ─────────────────────────────────────────────────────────────────────   ║
║   TOTAL DETECTION                    ~0.5ms          <5ms        OK       ║
║   ─────────────────────────────────────────────────────────────────────   ║
║                                                                           ║
║   QuestDB ILP write                  ~1ms            <2ms        OK       ║
║   Kalshi REST API call               ~50-200ms       <500ms      OK       ║
║   Order fill (network)               ~100-500ms      <1s         OK       ║
║   ─────────────────────────────────────────────────────────────────────   ║
║   TOTAL END-TO-END                   ~200-700ms      <1s         OK       ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

### Integration Test Results

```
Message Statistics:
  Total messages received: 71
  Ticker updates processed: 26
  Trade messages processed: 43
  Messages/second: 4.7

Latency Statistics (message processing):
  Average: 0.035ms
  P50:     0.014ms
  P95:     0.043ms
  Min:     0.007ms
  Max:     0.353ms
```

### Latency Optimization Strategies

1. **Redis Hot Cache** - Sub-ms consensus lookups vs 50ms+ API calls
2. **Rate Limiting** - 100ms min between detections per ticker prevents thrashing
3. **Batch Writes** - QuestDB ILP batches multiple records per flush
4. **Connection Pooling** - Reuse HTTP connections to Kalshi
5. **Pre-calculated Thresholds** - Profile thresholds computed once at startup

---

## 13. QuestDB Time-Series Storage

### Tables

```sql
-- Price snapshots (every ticker update)
CREATE TABLE kalshi_ticks (
    ticker SYMBOL CAPACITY 10000,
    yes_bid INT,
    yes_ask INT,
    no_bid INT,
    no_ask INT,
    volume LONG,
    last_price INT,
    timestamp TIMESTAMP
) TIMESTAMP(timestamp) PARTITION BY DAY WAL;

-- Trade executions
CREATE TABLE kalshi_trades (
    ticker SYMBOL CAPACITY 10000,
    count INT,
    price INT,
    side SYMBOL CAPACITY 4,
    taker_side SYMBOL CAPACITY 4,
    trade_id STRING,
    timestamp TIMESTAMP
) TIMESTAMP(timestamp) PARTITION BY DAY WAL;

-- Detected arbitrage opportunities
CREATE TABLE arb_opportunities (
    ticker SYMBOL CAPACITY 10000,
    event_id STRING,
    sport SYMBOL CAPACITY 20,
    kalshi_yes_ask INT,
    consensus_prob INT,
    edge_cents INT,
    confidence_score INT,
    action SYMBOL CAPACITY 20,
    timestamp TIMESTAMP
) TIMESTAMP(timestamp) PARTITION BY DAY WAL;
```

### Access Patterns

```sql
-- Latest prices per market
SELECT * FROM kalshi_ticks
WHERE timestamp > dateadd('m', -5, now())
LATEST ON timestamp PARTITION BY ticker;

-- Trade volume by sport (last 24h)
SELECT
    substring(ticker, 1, 5) as sport,
    sum(count) as total_contracts,
    count() as trade_count
FROM kalshi_trades
WHERE timestamp > dateadd('d', -1, now())
GROUP BY sport;

-- Profitable signals
SELECT * FROM arb_opportunities
WHERE edge_cents >= 5
AND action != 'NO_TRADE'
ORDER BY timestamp DESC
LIMIT 100;
```

---

## 14. Configuration Reference

### Environment Variables

```bash
# ═══════════════════════════════════════════════════════════════════════════
# KALSHI API
# ═══════════════════════════════════════════════════════════════════════════
KALSHI_BASE_URL=https://demo-api.kalshi.co/trade-api/v2
KALSHI_KEY_ID=5d239a9f-f225-4732-a0ed-9728f2c6c4eb
KALSHI_PRIVATE_KEY_PATH=/app/keys/kalshi_demo_private_key.pem
KALSHI_ENV=demo  # or "prod"

# ═══════════════════════════════════════════════════════════════════════════
# SPORTSBOOKS
# ═══════════════════════════════════════════════════════════════════════════
ODDS_API_KEY=dac80126dedbfbe3ff7d1edb216a6c88
SPORTSBOOK_PROVIDER=odds_api

# ═══════════════════════════════════════════════════════════════════════════
# DATABASES
# ═══════════════════════════════════════════════════════════════════════════
DATABASE_URL=postgresql+psycopg://postgres:postgres@postgres:5432/arb_kalshi
REDIS_URL=redis://redis:6379/0

QUESTDB_ILP_HOST=questdb
QUESTDB_ILP_PORT=9009
QUESTDB_PG_HOST=questdb
QUESTDB_PG_PORT=8812
QUESTDB_PG_USER=admin
QUESTDB_PG_PASSWORD=quest

# ═══════════════════════════════════════════════════════════════════════════
# TRADING
# ═══════════════════════════════════════════════════════════════════════════
LIVE_TRADING=false
PAPER_TRADING=true
TRADING_PROFILE=ANY_EDGE
TRADING_BANKROLL=10000

# ═══════════════════════════════════════════════════════════════════════════
# RISK CONTROLS
# ═══════════════════════════════════════════════════════════════════════════
MAX_POSITION_SIZE=100
MAX_DAILY_LOSS=100
MAX_OPEN_ORDERS=10
CIRCUIT_BREAKER_THRESHOLD=3

# ═══════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════
LOG_LEVEL=INFO
```

### Sports Series Supported

| Series | Sport | Example Ticker |
|--------|-------|----------------|
| KXNFL | NFL Football | KXNFL-26JAN11-BUF |
| KXNCAAF | College Football | KXNCAAF-26-MIA |
| KXNBA | NBA Basketball | KXNBA-26-NYK |
| KXNCAAB | College Basketball | KXNCAAB-26-DUKE |
| KXMLB | MLB Baseball | KXMLB-26-NYY |
| KXNHL | NHL Hockey | KXNHL-26-BOS |

---

## Quick Reference Commands

```bash
# View system status
docker ps --filter "name=arb-"

# View live WebSocket data
docker logs -f arb-ws-consumer

# Check QuestDB console
open http://localhost:9000

# Query recent ticks
curl "http://localhost:9000/exec?query=SELECT%20*%20FROM%20kalshi_ticks%20LIMIT%2010"

# Check Redis cache
docker exec arb-redis redis-cli KEYS "kalshi:*"
docker exec arb-redis redis-cli HGETALL "kalshi:m:KXNBA-26-NYK"

# Restart system
docker compose -f docker-compose.full.yml restart

# Stop system
docker compose -f docker-compose.full.yml down

# View all logs
docker compose -f docker-compose.full.yml logs -f
```

---

**Document Version:** 1.0
**Last Updated:** 2026-01-12
**System Status:** PRODUCTION RUNNING
