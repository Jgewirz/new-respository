# Kalshi Arbitrage Detection System

A production-grade arbitrage detection and execution system that identifies pricing discrepancies between Kalshi prediction markets and traditional sportsbook odds (DraftKings, FanDuel, BetMGM, Caesars).

## Architecture

The system follows a **Kalshi-first architecture**: all flows originate from Kalshi markets, which are then mapped to sportsbook events for edge detection.

```
Kalshi Markets ──► Event Mapping ──► Sportsbook Query ──► Edge Detection ──► Execution
   (KalshiClient)    (EventResolver)   (OddsAPIClient)      (EdgeDetector)    (CircuitBreaker)
```

## Features

- **Real-time Arbitrage Detection** - Identifies pricing discrepancies between Kalshi and sportsbooks
- **Configurable Trading Profiles** - Conservative, Standard, Aggressive, and custom profiles
- **Circuit Breaker Risk Controls** - Multi-layered protection against runaway losses
- **Docker Infrastructure** - PostgreSQL, Redis, and QuestDB for data persistence
- **RSA-PSS Authentication** - Secure Kalshi API integration
- **Fuzzy Event Matching** - Intelligent mapping between Kalshi and sportsbook events

## Quick Start

### Prerequisites

- Python 3.11+
- Docker Desktop
- Kalshi API credentials
- The Odds API key

### Installation

```bash
# Clone the repository
git clone https://github.com/Jgewirz/new-respository.git
cd new-respository

# Install dependencies
cd arb-kalshi-sportsbook
pip install -e .

# Copy environment template
cp .env.example .env
# Edit .env with your API credentials
```

### Start Infrastructure

```bash
# Start Docker containers (PostgreSQL, Redis, QuestDB)
docker-compose up -d

# Verify containers are running
docker ps

# Create QuestDB tables
python -m app.cli.run_ingest schema
```

### Run the Pipeline

```bash
# Run arbitrage detection cycle
python -m app.services.arb_pipeline
```

## Trading Profiles

| Profile | Min Edge | Book Consensus | Kelly Fraction | Use Case |
|---------|----------|----------------|----------------|----------|
| CONSERVATIVE | 6c | 4/4 books | 0.15 | Low risk |
| STANDARD | 5c | 3/4 books | 0.25 | Balanced |
| AGGRESSIVE | 3c | 3/4 books | 0.35 | More trades |
| VERY_AGGRESSIVE | 2c | 2/4 books | 0.50 | Maximum capture |

Set profile via environment:
```bash
export TRADING_PROFILE=STANDARD
```

## Docker Services

| Service | Port | Purpose |
|---------|------|---------|
| PostgreSQL | 5432 | Order history, configuration |
| Redis | 6379 | Real-time odds cache |
| QuestDB | 9000, 9009, 8812 | Time-series data |

Access QuestDB dashboard at: http://localhost:9000

## Project Structure

```
arb-kalshi-sportsbook/
├── app/
│   ├── arb/
│   │   └── detector.py          # Edge detection engine
│   ├── connectors/
│   │   ├── kalshi/              # Kalshi API client
│   │   └── odds_api/            # Sportsbook odds client
│   ├── execution/
│   │   ├── circuit_breaker.py   # Risk controls
│   │   ├── models.py            # Order/Fill dataclasses
│   │   └── paper_executor.py    # Simulated execution
│   ├── mapping/
│   │   └── resolver.py          # Event matching
│   └── services/
│       ├── arb_pipeline.py      # Main orchestrator
│       └── odds_ingest.py       # Redis caching
├── docker-compose.yml
└── pyproject.toml
```

## Configuration

Create a `.env` file with:

```bash
# Kalshi API
KALSHI_BASE_URL=https://trading-api.kalshi.com/trade-api/v2
KALSHI_KEY_ID=your_key_id
KALSHI_PRIVATE_KEY_PATH=./kalshi_private_key.pem

# Sportsbooks
ODDS_API_KEY=your_odds_api_key

# Trading
TRADING_PROFILE=STANDARD
TRADING_BANKROLL=10000
PAPER_TRADING=true

# Risk Limits
MAX_POSITION_SIZE=100
MAX_DAILY_LOSS=100
MAX_CONSECUTIVE_LOSSES=100
```

## Risk Controls

The circuit breaker enforces:

- **Per-trade limits** - Max position size, max risk per trade
- **Daily limits** - Max loss, max trades, max volume
- **Concentration limits** - Max exposure per event/sport
- **Rate limits** - Orders per minute
- **Consecutive loss protection** - Auto-halt after N losses

## Commands Reference

```bash
# Container management
docker-compose up -d      # Start containers
docker-compose down       # Stop containers
docker-compose logs -f    # View logs

# Database access
docker exec -it arb-postgres psql -U postgres -d arb_kalshi
docker exec -it arb-redis redis-cli

# Run tests
pytest tests/

# Lint and type check
ruff check app/
mypy app/ --strict
```

## Documentation

- [Algorithm Strategy](ALGORITHM_STRATEGY.md) - Trading algorithm details
- [Docker Plan](DOCKER_CONTAINERIZATION_PLAN.md) - Infrastructure setup
- [Execution Engine](EXECUTION_ENGINE_PLAN.md) - Order execution flow
- [QuestDB Logic](QUESTDB_SYSTEM_LOGIC.md) - Time-series data patterns

## Performance Targets

- ILP writes: <1ms per record
- LATEST ON queries: <10ms
- Detection cycle: <100ms
- End-to-end execution: <500ms

## License

Educational/Research Use Only. Users are responsible for compliance with applicable laws and regulations.

## Disclaimer

This software is for educational purposes. Past performance does not guarantee future results. Trading prediction markets involves risk of loss.
