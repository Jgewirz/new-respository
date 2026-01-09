# Kalshi ↔ Sportsbook Arbitrage System

A production-grade arbitrage detection and execution system that identifies pricing discrepancies between Kalshi prediction markets and traditional sportsbook odds. The system ingests real-time odds from multiple sources, normalizes event mappings, calculates implied probabilities, detects arbitrage opportunities, and executes trades with proper risk management and circuit breakers.

## Setup

1. Clone the repository
2. Copy `.env.example` to `.env` and configure credentials
3. Install dependencies: `uv sync` or `poetry install`
4. Start infrastructure: `docker-compose up -d`
5. Run database migrations
6. Start the application

## Repo Structure

```
/arb-kalshi-sportsbook/
├── README.md
├── .gitignore
├── .env.example
├── docker-compose.yml
├── pyproject.toml
├── /app/
│   ├── main.py                 # Application entry point
│   ├── settings.py             # Configuration management
│   ├── logging_config.py       # Logging setup
│   ├── /core/                  # Core utilities
│   │   ├── constants.py
│   │   ├── time.py
│   │   ├── math.py
│   │   └── errors.py
│   ├── /connectors/            # External API integrations
│   │   ├── /kalshi/            # Kalshi API client
│   │   └── /sportsbooks/       # Sportsbook providers
│   ├── /mapping/               # Event matching logic
│   │   ├── resolver.py
│   │   ├── rules.py
│   │   └── manual_overrides.py
│   ├── /arb/                   # Arbitrage detection
│   │   ├── implied_prob.py
│   │   ├── opportunity.py
│   │   ├── sizing.py
│   │   ├── planner.py
│   │   └── constraints.py
│   ├── /execution/             # Trade execution
│   │   ├── paper.py
│   │   ├── kalshi_executor.py
│   │   ├── hedge_instructions.py
│   │   ├── reconciliation.py
│   │   └── circuit_breaker.py
│   ├── /data/                  # Database layer
│   │   ├── db.py
│   │   ├── models.py
│   │   ├── repo.py
│   │   └── /migrations/
│   ├── /services/              # Business logic services
│   │   ├── ingest_service.py
│   │   ├── arb_service.py
│   │   ├── execution_service.py
│   │   └── metrics_service.py
│   ├── /api/                   # REST API
│   │   ├── routes.py
│   │   └── schemas.py
│   └── /cli/                   # CLI commands
│       ├── run_ingest.py
│       ├── run_detector.py
│       └── run_executor.py
└── /tests/                     # Test suite
    └── test_placeholder.py
```

## Next Steps

- [ ] Implement Kalshi API authentication and client
- [ ] Implement sportsbook provider connectors
- [ ] Build event mapping/resolution logic
- [ ] Implement implied probability calculations
- [ ] Build arbitrage opportunity detection
- [ ] Implement position sizing with Kelly criterion
- [ ] Build execution layer with paper trading mode
- [ ] Add circuit breaker and risk controls
- [ ] Implement reconciliation and P&L tracking
- [ ] Add metrics and monitoring
- [ ] Build REST API for dashboard
- [ ] Add comprehensive test coverage
