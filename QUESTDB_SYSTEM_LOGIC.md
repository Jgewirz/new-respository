# QuestDB System Logic for Kalshi Arbitrage Platform

## System Context

You are implementing the data layer for a **real-time arbitrage detection system** that identifies pricing discrepancies between Kalshi prediction markets and traditional sportsbooks. QuestDB serves as the **high-performance time-series backbone** for storing and querying market data at sub-millisecond latency.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA INGESTION LAYER                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐    │
│  │ Kalshi WebSocket │     │ Sportsbook APIs  │     │ Redis (Hot Cache)│    │
│  │ (ticker/trade/   │     │ (DraftKings,     │     │ (Latest prices,  │    │
│  │  orderbook)      │     │  FanDuel, etc.)  │     │  market state)   │    │
│  └────────┬─────────┘     └────────┬─────────┘     └────────┬─────────┘    │
│           │                        │                        │               │
│           ▼                        ▼                        ▼               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    ILP INGESTION (Port 9009)                         │   │
│  │            High-speed writes via InfluxDB Line Protocol              │   │
│  │                    Target: < 1ms per write                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
└────────────────────────────────────┼────────────────────────────────────────┘
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              QUESTDB STORAGE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │  kalshi_ticks   │  │  kalshi_trades  │  │ kalshi_orderbook│             │
│  │  (PARTITION BY  │  │  (PARTITION BY  │  │  (PARTITION BY  │             │
│  │   DAY, WAL)     │  │   DAY, WAL)     │  │   DAY, WAL)     │             │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘             │
│                                                                              │
│  ┌─────────────────┐  ┌─────────────────┐                                   │
│  │ sportsbook_odds │  │arb_opportunities│                                   │
│  │  (PARTITION BY  │  │  (PARTITION BY  │                                   │
│  │   DAY, WAL)     │  │   DAY, WAL)     │                                   │
│  └─────────────────┘  └─────────────────┘                                   │
│                                                                              │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         QUERY & ANALYTICS LAYER                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐    │
│  │  Arb Detector    │     │  Metrics Service │     │  REST API        │    │
│  │  (Real-time      │     │  (P&L, hit rate, │     │  (Dashboard,     │    │
│  │   edge calc)     │     │   latency)       │     │   historical)    │    │
│  └──────────────────┘     └──────────────────┘     └──────────────────┘    │
│           │                        │                        │               │
│           └────────────────────────┴────────────────────────┘               │
│                                    │                                        │
│                                    ▼                                        │
│              PostgreSQL Wire Protocol (Port 8812) - SQL Queries             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow & Logic

### 1. INGESTION PIPELINE

#### Kalshi WebSocket → QuestDB

```python
# When a ticker update arrives from Kalshi WebSocket:
async def on_kalshi_ticker(msg: dict):
    """
    msg format:
    {
        "type": "ticker",
        "msg": {
            "market_ticker": "KXNFL-25JAN12-BUF",
            "yes_bid": 45,
            "yes_ask": 47,
            "no_bid": 53,
            "no_ask": 55,
            "volume": 12500,
            "ts": 1704672000000  # milliseconds
        }
    }
    """
    ticker_data = msg["msg"]

    # 1. Write to QuestDB via ILP (sub-millisecond)
    ilp_client.write_tick(
        ticker=ticker_data["market_ticker"],
        yes_bid=ticker_data["yes_bid"],
        yes_ask=ticker_data["yes_ask"],
        no_bid=ticker_data["no_bid"],
        no_ask=ticker_data["no_ask"],
        volume=ticker_data["volume"],
        timestamp_ns=ticker_data["ts"] * 1_000_000  # Convert ms → ns
    )

    # 2. Update Redis hot cache (for instant lookups)
    await redis.hset(f"kalshi:m:{ticker_data['market_ticker']}", mapping={
        "yes_bid": ticker_data["yes_bid"],
        "yes_ask": ticker_data["yes_ask"],
        "updated_at": ticker_data["ts"]
    })

    # 3. Trigger arbitrage check (async)
    await arb_detector.check_opportunity(ticker_data["market_ticker"])
```

#### Sportsbook Polling → QuestDB

```python
# Sportsbook odds are polled every N seconds
async def ingest_sportsbook_odds(book: str, event_id: str, odds: dict):
    """
    Store sportsbook odds for arbitrage comparison.

    odds format:
    {
        "moneyline_home": -150,  # American odds
        "moneyline_away": +130,
        "spread_home": -3.5,
        "spread_home_odds": -110,
        "total": 48.5,
        "over_odds": -110,
        "under_odds": -110
    }
    """
    timestamp_ns = int(time.time() * 1e9)

    # Convert American odds to implied probability
    for market_type, american_odds in [
        ("moneyline_home", odds.get("moneyline_home")),
        ("moneyline_away", odds.get("moneyline_away")),
    ]:
        if american_odds is None:
            continue

        implied_prob = american_to_implied_prob(american_odds)
        decimal_odds = american_to_decimal(american_odds)

        ilp_client.write_line(
            f"sportsbook_odds,event_id={event_id},book={book},market_type={market_type} "
            f"odds_decimal={decimal_odds},implied_prob={implied_prob} {timestamp_ns}"
        )
```

---

### 2. ARBITRAGE DETECTION QUERIES

#### Real-Time Edge Calculation

```sql
-- Find current arbitrage opportunities
-- Compare Kalshi YES price vs Sportsbook implied probability

WITH latest_kalshi AS (
    SELECT
        ticker,
        yes_ask / 100.0 AS kalshi_yes_prob,  -- Kalshi prices are in cents (0-100)
        no_ask / 100.0 AS kalshi_no_prob,
        timestamp
    FROM kalshi_ticks
    WHERE timestamp > dateadd('m', -1, now())  -- Last 1 minute
    LATEST ON timestamp PARTITION BY ticker
),
latest_sportsbook AS (
    SELECT
        event_id,
        book,
        market_type,
        implied_prob AS book_prob,
        timestamp
    FROM sportsbook_odds
    WHERE timestamp > dateadd('m', -5, now())  -- Last 5 minutes
    LATEST ON timestamp PARTITION BY event_id, book, market_type
),
mapped_events AS (
    -- Join Kalshi markets to sportsbook events via mapping table
    SELECT
        k.ticker AS kalshi_ticker,
        s.event_id AS sportsbook_event,
        s.book,
        k.kalshi_yes_prob,
        s.book_prob,
        -- Calculate edge: if Kalshi YES is cheaper than sportsbook implies
        (s.book_prob - k.kalshi_yes_prob) * 100 AS edge_pct,
        k.timestamp AS kalshi_ts,
        s.timestamp AS book_ts
    FROM latest_kalshi k
    INNER JOIN event_mapping m ON k.ticker = m.kalshi_ticker
    INNER JOIN latest_sportsbook s ON m.sportsbook_event_id = s.event_id
    WHERE s.market_type = 'moneyline_home'  -- Example: home team win
)
SELECT *
FROM mapped_events
WHERE edge_pct > 2.0  -- Minimum 2% edge threshold
ORDER BY edge_pct DESC
LIMIT 20;
```

#### Time-Windowed Opportunity Detection

```sql
-- Detect opportunities that persist across multiple ticks (reduces noise)

SELECT
    ticker,
    avg(yes_bid) AS avg_yes_bid,
    avg(yes_ask) AS avg_yes_ask,
    max(yes_bid) AS max_yes_bid,
    min(yes_ask) AS min_yes_ask,
    count() AS tick_count,
    max(timestamp) - min(timestamp) AS duration
FROM kalshi_ticks
WHERE
    timestamp > dateadd('s', -30, now())  -- Last 30 seconds
    AND ticker = 'KXNFL-25JAN12-BUF'
SAMPLE BY 5s  -- 5-second buckets
ORDER BY timestamp DESC;
```

---

### 3. EXECUTION & RECONCILIATION

#### Log Executed Trades

```python
async def log_execution(
    kalshi_ticker: str,
    side: str,  # "yes" or "no"
    price: int,
    quantity: int,
    order_id: str,
    hedge_book: str,
    hedge_event: str
):
    """Log trade execution for P&L tracking."""

    # Write to QuestDB
    ilp_client.write_line(
        f"executions,ticker={kalshi_ticker},side={side},hedge_book={hedge_book} "
        f"price={price}i,quantity={quantity}i,order_id=\"{order_id}\" "
        f"{int(time.time() * 1e9)}"
    )
```

#### P&L Calculation Query

```sql
-- Calculate daily P&L from executed trades

WITH trades AS (
    SELECT
        ticker,
        side,
        price,
        quantity,
        timestamp
    FROM executions
    WHERE timestamp > dateadd('d', -1, now())
),
settlements AS (
    -- Join with settlement data when markets close
    SELECT
        t.ticker,
        t.side,
        t.price,
        t.quantity,
        s.winning_side,
        CASE
            WHEN t.side = s.winning_side THEN (100 - t.price) * t.quantity
            ELSE -t.price * t.quantity
        END AS pnl_cents
    FROM trades t
    LEFT JOIN market_settlements s ON t.ticker = s.ticker
)
SELECT
    sum(pnl_cents) / 100.0 AS total_pnl_dollars,
    count() AS trade_count,
    sum(CASE WHEN pnl_cents > 0 THEN 1 ELSE 0 END) AS winning_trades,
    sum(CASE WHEN pnl_cents < 0 THEN 1 ELSE 0 END) AS losing_trades
FROM settlements;
```

---

### 4. METRICS & MONITORING

#### Latency Tracking

```sql
-- Measure ingestion latency (time from Kalshi tick to QuestDB write)

SELECT
    ticker,
    avg(write_latency_ms) AS avg_latency,
    max(write_latency_ms) AS p99_latency,
    count() AS tick_count
FROM (
    SELECT
        ticker,
        (systimestamp() - timestamp) / 1000000 AS write_latency_ms
    FROM kalshi_ticks
    WHERE timestamp > dateadd('h', -1, now())
)
GROUP BY ticker
ORDER BY avg_latency DESC
LIMIT 20;
```

#### Opportunity Hit Rate

```sql
-- Track how many detected opportunities were actually executable

SELECT
    date_trunc('hour', timestamp) AS hour,
    count() AS opportunities_detected,
    sum(CASE WHEN executed = true THEN 1 ELSE 0 END) AS executed,
    sum(CASE WHEN executed = true THEN 1 ELSE 0 END) * 100.0 / count() AS hit_rate_pct
FROM arb_opportunities
WHERE timestamp > dateadd('d', -7, now())
SAMPLE BY 1h
ORDER BY hour DESC;
```

---

## Schema Design Rationale

### Why These Tables?

| Table | Purpose | Write Pattern | Query Pattern |
|-------|---------|---------------|---------------|
| `kalshi_ticks` | Store every price update | High-frequency ILP writes (100s/sec) | Real-time LATEST BY queries |
| `kalshi_trades` | Trade execution log | Medium frequency (on trades) | Time-range aggregations |
| `kalshi_orderbook` | L2 orderbook deltas | High-frequency (orderbook updates) | Reconstruct book state |
| `sportsbook_odds` | Comparison pricing | Polling-based (every few seconds) | Join with Kalshi for arb detection |
| `arb_opportunities` | Detected edges | On detection (event-driven) | Analytics, hit rate tracking |

### Partitioning Strategy

```sql
-- All tables use: PARTITION BY DAY WAL
--
-- Why DAY partitioning?
-- - Sports events are date-bound (game day)
-- - Efficient pruning for "last N hours" queries
-- - Automatic data lifecycle (drop old partitions)
--
-- Why WAL (Write-Ahead Log)?
-- - Enables concurrent writes from multiple ingestors
-- - Durability without sacrificing write speed
-- - Required for DEDUP UPSERT functionality
```

### Symbol Columns

```sql
-- Use SYMBOL type for low-cardinality string columns:
-- - ticker: ~1000-5000 unique values (markets)
-- - series: ~100-500 unique values
-- - sport: ~10 unique values (nfl, nba, mlb, etc.)
-- - book: ~20 unique values (sportsbooks)
--
-- Benefits:
-- - Automatic dictionary encoding
-- - Faster GROUP BY and WHERE filtering
-- - Lower memory footprint
```

---

## Performance Guidelines

### Write Path (ILP)

```python
# DO: Batch writes when possible
with QuestDBILPClient() as ilp:
    for tick in batch:
        ilp.write_tick(**tick)
    # Implicit flush on context exit

# DON'T: Open/close connection per write
for tick in batch:
    with QuestDBILPClient() as ilp:  # BAD: connection overhead
        ilp.write_tick(**tick)
```

### Query Path (SQL)

```sql
-- DO: Use LATEST ON for real-time price lookups
SELECT * FROM kalshi_ticks
WHERE ticker = 'KXNFL-25JAN12-BUF'
LATEST ON timestamp PARTITION BY ticker;

-- DON'T: Use ORDER BY + LIMIT 1 (full scan)
SELECT * FROM kalshi_ticks
WHERE ticker = 'KXNFL-25JAN12-BUF'
ORDER BY timestamp DESC
LIMIT 1;  -- BAD: scans entire partition
```

### Time Filters

```sql
-- DO: Always include time bounds for large tables
SELECT * FROM kalshi_ticks
WHERE timestamp > dateadd('h', -1, now())  -- Last hour only
  AND ticker = 'KXNFL-25JAN12-BUF';

-- DON'T: Query without time filter
SELECT * FROM kalshi_ticks
WHERE ticker = 'KXNFL-25JAN12-BUF';  -- BAD: scans all partitions
```

---

## Integration Points

### 1. WebSocket Consumer → QuestDB

```
scripts/kalshi_ws_sports_only_consumer.py
    │
    ├── on_ticker() ──────────► ilp.write_tick()
    ├── on_trade()  ──────────► ilp.write_trade()
    └── on_orderbook_delta() ─► ilp.write_orderbook_delta()
```

### 2. Arb Detector → QuestDB

```
app/services/arb_service.py
    │
    ├── detect_opportunities()
    │       │
    │       ├── Query: SELECT ... FROM kalshi_ticks LATEST ON ...
    │       ├── Query: SELECT ... FROM sportsbook_odds LATEST ON ...
    │       └── Write: ilp.write_opportunity()
    │
    └── get_historical_performance()
            │
            └── Query: SELECT ... FROM arb_opportunities SAMPLE BY ...
```

### 3. Redis ↔ QuestDB Sync

```
Redis (Hot Cache)                    QuestDB (Cold Storage)
─────────────────                    ─────────────────────
kalshi:m:{ticker}  ◄───── write ────► kalshi_ticks
  - yes_bid                            - Full history
  - yes_ask                            - Partitioned by day
  - updated_at                         - Query analytics

Use Redis for:                       Use QuestDB for:
  - Instant price lookups              - Time-series queries
  - Current market state               - Historical analysis
  - Sub-ms read latency                - Aggregations (SAMPLE BY)
```

---

## Example: Complete Arbitrage Detection Flow

```python
async def arbitrage_detection_loop():
    """
    Main loop that detects and logs arbitrage opportunities.
    """
    questdb_sql = QuestDBClient()
    questdb_ilp = QuestDBILPClient()
    questdb_sql.connect()
    questdb_ilp.connect()

    while True:
        # 1. Query latest Kalshi prices
        kalshi_prices = questdb_sql.execute("""
            SELECT ticker, yes_ask, no_ask, timestamp
            FROM kalshi_ticks
            WHERE timestamp > dateadd('s', -30, now())
            LATEST ON timestamp PARTITION BY ticker
        """)

        # 2. Query latest sportsbook odds
        sportsbook_prices = questdb_sql.execute("""
            SELECT event_id, book, implied_prob, timestamp
            FROM sportsbook_odds
            WHERE timestamp > dateadd('m', -5, now())
            LATEST ON timestamp PARTITION BY event_id, book
        """)

        # 3. Match events and calculate edges
        for kalshi in kalshi_prices:
            mapped_event = get_mapped_event(kalshi["ticker"])
            if not mapped_event:
                continue

            for book_price in sportsbook_prices:
                if book_price["event_id"] != mapped_event:
                    continue

                kalshi_prob = kalshi["yes_ask"] / 100.0
                book_prob = book_price["implied_prob"]
                edge = (book_prob - kalshi_prob) * 100

                if edge > MIN_EDGE_THRESHOLD:
                    # 4. Log opportunity to QuestDB
                    questdb_ilp.write_line(
                        f"arb_opportunities,"
                        f"kalshi_ticker={kalshi['ticker']},"
                        f"sportsbook_event={mapped_event},"
                        f"book={book_price['book']} "
                        f"kalshi_prob={kalshi_prob},"
                        f"book_prob={book_prob},"
                        f"edge_pct={edge} "
                        f"{int(time.time() * 1e9)}"
                    )

                    # 5. Notify execution layer
                    await notify_opportunity({
                        "kalshi_ticker": kalshi["ticker"],
                        "edge_pct": edge,
                        "kalshi_price": kalshi["yes_ask"],
                        "book": book_price["book"]
                    })

        await asyncio.sleep(0.1)  # 100ms detection cycle
```

---

## Summary

| Component | Protocol | Port | Use Case |
|-----------|----------|------|----------|
| ILP Ingestion | TCP | 9009 | All writes (ticks, trades, opportunities) |
| SQL Queries | PostgreSQL | 8812 | Analytics, arb detection, P&L |
| Web Console | HTTP | 9000 | Debugging, ad-hoc queries |

**Key Principles:**
1. **Write via ILP** - Never use SQL INSERT for time-series data
2. **Always time-bound queries** - Include timestamp filters
3. **Use LATEST ON** - For real-time price lookups
4. **SYMBOL for strings** - Low-cardinality columns only
5. **Redis for hot path** - QuestDB for analytics and history
