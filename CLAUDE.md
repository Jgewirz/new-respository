# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A production-grade arbitrage detection and execution system that identifies pricing discrepancies between Kalshi prediction markets and traditional sportsbook odds (DraftKings, FanDuel, BetMGM, Caesars). The system follows a **Kalshi-first architecture**: all flows originate from Kalshi markets, which are then mapped to sportsbook events for edge detection.

## Build & Run Commands

```bash
# Install dependencies
uv sync                          # Production deps
uv sync --all-extras             # With dev tools (pytest, ruff, mypy)

# Start infrastructure (PostgreSQL, Redis, QuestDB)
docker-compose up -d

# Create QuestDB tables
python -m app.cli.run_ingest schema

# Ingest JSONL data
python -m app.cli.run_ingest jsonl path/to/data.jsonl --batch-size 5000
python -m app.cli.run_ingest jsonl data.jsonl --dry-run   # Validate only

# Query ingested data
python -m app.cli.run_ingest query --limit 10
python -m app.cli.run_ingest query --stats

# Run main arbitrage pipeline
python -m app.services.arb_pipeline

# Run tests
pytest tests/
pytest tests/test_placeholder.py -v    # Single test file
pytest tests/ --cov=app                # With coverage

# Linting and type checking
ruff check app/
ruff format app/
mypy app/ --strict
```

## Architecture

The main application lives in `arb-kalshi-sportsbook/app/`:

```
Kalshi-First Flow:

1. Kalshi Markets ──► 2. Event Mapping ──► 3. Sportsbook Query ──► 4. Edge Detection ──► 5. Execution
   (KalshiClient)       (EventResolver)      (OddsAPIClient)         (EdgeDetector)       (TBD)
```

```
Data Layer:

Data Sources                        Storage Layer                     Analysis
─────────────                       ─────────────                     ────────
Kalshi WebSocket ─┐                 ┌─────────────┐
Sportsbook APIs ──┼── ILP (9009) ──►│  QuestDB    │── SQL (8812) ──► Arb Detector
Redis hot cache ──┘                 │  (6 tables) │                   Metrics/P&L
                                    └─────────────┘                   REST API
```

### Key Modules

| Module | Purpose | Status |
|--------|---------|--------|
| `app/services/arb_pipeline.py` | **Main orchestrator** - unified Kalshi-first pipeline | Implemented |
| `app/arb/detector.py` | Edge detection, confidence scoring, signal generation | Implemented |
| `app/connectors/kalshi/client.py` | Full Kalshi API client (auth, orders, markets) | Implemented |
| `app/connectors/kalshi/auth.py` | RSA-PSS authentication | Implemented |
| `app/mapping/resolver.py` | Event mapping (Kalshi ↔ sportsbook) with fuzzy matching | Implemented |
| `app/connectors/odds_api/client.py` | The Odds API integration | Implemented |
| `app/services/odds_ingest.py` | Redis caching for sportsbook consensus | Implemented |
| `app/data/questdb.py` | QuestDB ILP + SQL clients | Implemented |
| `app/data/jsonl_reader.py` | JSONL streaming with `KalshiMarket` dataclass | Implemented |
| `app/services/jsonl_ingest.py` | Batch JSONL pipeline | Implemented |
| `app/cli/run_ingest.py` | Data ingestion CLI | Implemented |
| `app/execution/models.py` | ExecutionOrder, Fill, ExecutionResult dataclasses | Implemented |
| `app/execution/circuit_breaker.py` | Risk controls, daily limits, kill switch | Implemented |
| `app/execution/paper_executor.py` | Simulated execution for testing | Pending |
| `app/execution/kalshi_executor.py` | Live execution against Kalshi API | Pending |

### Docker Services

| Service | Container | Ports |
|---------|-----------|-------|
| PostgreSQL 16 | arb-postgres | 5432 |
| Redis 7 | arb-redis | 6379 |
| QuestDB 8.2.1 | arb-questdb | 9000 (Web), 9009 (ILP), 8812 (SQL), 9003 (Health) |

### QuestDB Tables

- `kalshi_ticks` - Real-time price updates (DEDUP UPSERT)
- `kalshi_trades` - Trade execution records
- `kalshi_orderbook` - L2 orderbook deltas
- `kalshi_markets` - Market snapshots from JSONL
- `sportsbook_odds` - Normalized sportsbook prices
- `arb_opportunities` - Detected arbitrage edges

All tables: `PARTITION BY DAY WAL` for performance and durability.

## Trading Profiles

The `EdgeDetector` supports 6 trading profiles defined in `app/arb/detector.py`:

| Profile | Min Edge | Book Consensus | Kelly Fraction | Use Case |
|---------|----------|----------------|----------------|----------|
| CONSERVATIVE | 6c | 4/4 books, 2.5c spread | 0.15 | Low risk, few trades |
| STANDARD | 5c | 3/4 books, 3c spread | 0.25 | Balanced approach |
| AGGRESSIVE | 3c | 3/4 books, 4c spread | 0.35 | More trades |
| VERY_AGGRESSIVE | 2c | 2/4 books, 5c spread | 0.50 | Maximum capture |
| ANY_EDGE | 1c | 2/4 books, any spread | 0.50 | Research/backtesting |
| SCALP | 2c | N/A (Kalshi spread only) | 0.10 | Market making |

Set via environment: `TRADING_PROFILE=STANDARD`

## Code Patterns

### Running the Main Pipeline

```python
from app.services.arb_pipeline import ArbPipeline

pipeline = ArbPipeline.from_env()
signals = pipeline.run_cycle()

for signal in signals:
    if signal.should_trade:
        print(f"Trade: {signal.action} on {signal.kalshi.ticker}")
```

### Using the Edge Detector Directly

```python
from app.arb.detector import EdgeDetector, KalshiMarket, SportsbookConsensus

detector = EdgeDetector(profile="STANDARD", bankroll=10000)

kalshi = KalshiMarket(ticker="KXNFL-26JAN11-BUF", yes_ask=48, no_ask=54, ...)
consensus = SportsbookConsensus(implied_prob=55, book_count=4, book_spread=2.0, ...)

signal = detector.evaluate(kalshi, consensus)
if signal.should_trade:
    print(f"Edge: {signal.edge.cents}c, Confidence: {signal.confidence.score}")
```

### QuestDB ILP Writes

```python
# Use context manager for batched writes
with QuestDBILPClient() as ilp:
    for market in batch:
        ilp.write_market_snapshot(**market.to_ilp_kwargs())
    # Implicit flush on exit
```

### QuestDB SQL Queries

```python
# Always include time bounds for performance
with QuestDBClient() as client:
    results = client.execute("""
        SELECT * FROM kalshi_ticks
        WHERE timestamp > dateadd('h', -1, now())
        LATEST ON timestamp PARTITION BY ticker
    """)
```

### Kalshi API Client

```python
from app.connectors.kalshi.client import KalshiClient

with KalshiClient.from_env() as client:
    balance = client.get_balance()
    market = client.get_market("KXNFL-26JAN11-BUF")
    order = client.create_order(
        ticker="KXNFL-26JAN11-BUF",
        side="yes",
        action="buy",
        count=10,
        type="limit",
        yes_price=52,
    )
```

### Execution Models (Signal to Order)

```python
from app.execution import ExecutionOrder, ExecutionResult, ExecutionMode

# Create order from detector Signal
order = ExecutionOrder.from_signal(signal, mode=ExecutionMode.PAPER)

# Simulate submission and fills
order.mark_submitted(kalshi_order_id="kalshi_123")
fill = order.record_fill(contracts=25, price=48)

# Build result
result = ExecutionResult.from_order(order, fills=[fill])
print(result.summary())  # "FILLED: 25/25 contracts @ 48.0c..."
```

### Circuit Breaker (Risk Controls)

```python
from app.execution import CircuitBreaker, RiskLimits

# Create with environment config
breaker = CircuitBreaker.from_env()

# Or with custom limits
limits = RiskLimits(max_position_size=50, max_daily_loss_cents=1000_00)
breaker = CircuitBreaker(limits=limits)

# Check before every trade
result = breaker.check_trade(
    ticker="KXNFL-26JAN11-BUF",
    contracts=25,
    risk_cents=1200,
    sport="nfl",
)
if not result.allowed:
    print(f"Blocked: {result.reason}")  # e.g., "Position size exceeded"

# Record after execution
breaker.record_trade(execution_result, sport="nfl")

# Record settlement P&L
breaker.record_settlement(pnl_cents=500)  # Win
breaker.record_settlement(pnl_cents=-300)  # Loss (tracks consecutive)

# Emergency controls
breaker.halt("Market anomaly")
breaker.resume()
```

## Configuration

Environment variables (see `.env.example`):

```bash
# Kalshi API
KALSHI_BASE_URL=https://trading-api.kalshi.com/trade-api/v2
KALSHI_KEY_ID=<key>
KALSHI_PRIVATE_KEY=<pem>

# Sportsbooks
ODDS_API_KEY=<key>

# Databases
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/arb_kalshi
REDIS_URL=redis://localhost:6379/0
QUESTDB_ILP_HOST=localhost
QUESTDB_ILP_PORT=9009
QUESTDB_PG_PORT=8812

# Trading
TRADING_PROFILE=STANDARD
TRADING_BANKROLL=10000
PAPER_TRADING=true
MAX_POSITION_SIZE=100
MAX_DAILY_LOSS=500
CIRCUIT_BREAKER_THRESHOLD=3
```

## Code Quality Settings

From `pyproject.toml`:
- Python 3.11+
- Line length: 100
- Ruff rules: E, F, I, N, W, UP
- MyPy: strict mode
- Pytest: asyncio_mode = "auto"

## Performance Targets

- ILP writes: <1ms per record
- LATEST ON queries: <10ms
- Arbitrage detection cycle: 100ms
- End-to-end opportunity → execution: <500ms

## Key Documentation

- `ALGORITHM_STRATEGY.md` - Complete trading algorithm, entry conditions, Kelly sizing
- `QUESTDB_SYSTEM_LOGIC.md` - Data layer architecture and query patterns
- `arb-kalshi-sportsbook/EXECUTION_ENGINE_PLAN.md` - Kalshi-first implementation phases
- `EXECUTION_LAYER_PLAN.md` - Detailed execution layer implementation plan with file specs
