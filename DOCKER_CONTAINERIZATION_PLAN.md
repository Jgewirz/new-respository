# Docker Containerization Plan
## Kalshi Arbitrage Detection System

**Purpose:** This document explains how the Docker environment works and how to set it up for someone unfamiliar with the system.

---

## Table of Contents
1. [What is This System?](#1-what-is-this-system)
2. [The Big Picture](#2-the-big-picture)
3. [Why Docker?](#3-why-docker)
4. [The Container Architecture](#4-the-container-architecture)
5. [How Data Flows Through the System](#5-how-data-flows-through-the-system)
6. [Container Details](#6-container-details)
7. [Network Configuration](#7-network-configuration)
8. [Getting Started](#8-getting-started)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. What is This System?

Imagine you're comparing prices at two different stores to find a bargain. This system does exactly that, but with prediction markets:

- **Kalshi** = A prediction market where you can bet on outcomes (like "Will the Buffalo Bills win?")
- **Sportsbooks** (DraftKings, FanDuel, etc.) = Traditional betting sites with their own odds

When Kalshi thinks Team A has a 60% chance of winning, but sportsbooks think it's 70%, there's a potential profit opportunity. This system automatically finds these discrepancies.

---

## 2. The Big Picture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         YOUR COMPUTER (HOST)                                │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    DOCKER ENVIRONMENT                               │   │
│  │                                                                     │   │
│  │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │   │
│  │   │   POSTGRES   │  │    REDIS     │  │   QUESTDB    │             │   │
│  │   │  (Database)  │  │   (Cache)    │  │ (Time-Series)│             │   │
│  │   │              │  │              │  │              │             │   │
│  │   │  Port 5432   │  │  Port 6379   │  │ Ports 9000,  │             │   │
│  │   │              │  │              │  │ 9009, 8812   │             │   │
│  │   └──────────────┘  └──────────────┘  └──────────────┘             │   │
│  │          │                 │                 │                      │   │
│  │          └─────────────────┼─────────────────┘                      │   │
│  │                            │                                        │   │
│  │                    ┌───────▼───────┐                               │   │
│  │                    │  ARB-NETWORK  │                               │   │
│  │                    │ (Docker Net)  │                               │   │
│  │                    └───────────────┘                               │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    YOUR APPLICATION                                 │   │
│  │                                                                     │   │
│  │   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐            │   │
│  │   │   Kalshi    │    │    Edge     │    │  Execution  │            │   │
│  │   │   Client    │───►│  Detector   │───►│   Engine    │            │   │
│  │   └─────────────┘    └─────────────┘    └─────────────┘            │   │
│  │          │                 │                   │                    │   │
│  │          │                 │                   │                    │   │
│  │          ▼                 ▼                   ▼                    │   │
│  │   ┌─────────────────────────────────────────────────────────┐      │   │
│  │   │         Connects to Docker containers above             │      │   │
│  │   └─────────────────────────────────────────────────────────┘      │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Why Docker?

Think of Docker like a **shipping container** for software:

| Without Docker | With Docker |
|----------------|-------------|
| "Works on my machine" problems | Same environment everywhere |
| Install PostgreSQL, Redis, etc. manually | One command starts everything |
| Version conflicts between tools | Each tool isolated in its container |
| Complex setup instructions | Just run `docker-compose up` |

**Real-world analogy:** Instead of building a house from scratch (installing everything manually), you're placing pre-built modular rooms (containers) that automatically connect together.

---

## 4. The Container Architecture

We use **3 containers** that each serve a specific purpose:

### Container 1: PostgreSQL (arb-postgres)
```
┌────────────────────────────────────────┐
│            POSTGRESQL                  │
│                                        │
│  Purpose: Long-term data storage       │
│                                        │
│  Stores:                               │
│  • Trade history                       │
│  • Order records                       │
│  • Account information                 │
│  • Configuration settings              │
│                                        │
│  Port: 5432                            │
│  Image: postgres:16-alpine             │
│                                        │
│  Think of it as: A filing cabinet      │
│  that keeps permanent records          │
└────────────────────────────────────────┘
```

### Container 2: Redis (arb-redis)
```
┌────────────────────────────────────────┐
│               REDIS                    │
│                                        │
│  Purpose: Lightning-fast cache         │
│                                        │
│  Stores:                               │
│  • Latest sportsbook odds              │
│  • Session data                        │
│  • Rate limiting counters              │
│  • Temporary calculations              │
│                                        │
│  Port: 6379                            │
│  Image: redis:7-alpine                 │
│                                        │
│  Think of it as: A sticky note board   │
│  for quick reference                   │
└────────────────────────────────────────┘
```

### Container 3: QuestDB (arb-questdb)
```
┌────────────────────────────────────────┐
│              QUESTDB                   │
│                                        │
│  Purpose: Time-series data             │
│           (super fast for trading)     │
│                                        │
│  Stores:                               │
│  • Price ticks (every second)          │
│  • Trade executions                    │
│  • Orderbook changes                   │
│  • Arbitrage opportunities             │
│                                        │
│  Ports:                                │
│  • 9000: Web Console (dashboard)       │
│  • 9009: ILP (fast data ingestion)     │
│  • 8812: SQL queries                   │
│                                        │
│  Think of it as: A high-speed          │
│  recording device for market data      │
└────────────────────────────────────────┘
```

---

## 5. How Data Flows Through the System

Here's the complete journey of data through the system:

```
Step 1: EXTERNAL DATA SOURCES
══════════════════════════════

    ┌──────────────┐         ┌──────────────────┐
    │    KALSHI    │         │   SPORTSBOOKS    │
    │     API      │         │   (The Odds API) │
    │              │         │                  │
    │  Real-time   │         │  DraftKings      │
    │  prediction  │         │  FanDuel         │
    │  markets     │         │  BetMGM          │
    └──────┬───────┘         │  Caesars         │
           │                 └────────┬─────────┘
           │                          │
           ▼                          ▼

Step 2: DATA INGESTION
══════════════════════

    ┌─────────────────────────────────────────────┐
    │            ARBITRAGE PIPELINE               │
    │                                             │
    │   1. Fetch Kalshi markets                   │
    │   2. Fetch sportsbook odds                  │
    │   3. Map Kalshi ↔ Sportsbook events        │
    │   4. Calculate price differences            │
    │                                             │
    └──────────────────┬──────────────────────────┘
                       │
                       ▼

Step 3: CACHE LAYER (Speed)
═══════════════════════════

    ┌─────────────────────────────────────────────┐
    │              REDIS CONTAINER                │
    │                                             │
    │   Stores in memory for instant access:      │
    │                                             │
    │   odds:nfl:event123 = {                     │
    │     "draftkings": 55,                       │
    │     "fanduel": 54,                          │
    │     "consensus": 54.5                       │
    │   }                                         │
    │                                             │
    │   Access time: < 1 millisecond              │
    │                                             │
    └──────────────────┬──────────────────────────┘
                       │
                       ▼

Step 4: DETECTION ENGINE
════════════════════════

    ┌─────────────────────────────────────────────┐
    │            EDGE DETECTOR                    │
    │                                             │
    │   Kalshi YES price: 48¢                     │
    │   Sportsbook consensus: 55%                 │
    │                                             │
    │   Edge = 55 - 48 = 7¢  ✓ PROFITABLE        │
    │                                             │
    │   Generate Signal:                          │
    │   → BUY_YES @ 48¢                          │
    │   → Confidence: HIGH                        │
    │   → Recommended: 25 contracts               │
    │                                             │
    └──────────────────┬──────────────────────────┘
                       │
                       ▼

Step 5: EXECUTION & STORAGE
═══════════════════════════

    ┌─────────────────┐    ┌─────────────────┐
    │   CIRCUIT       │    │    QUESTDB      │
    │   BREAKER       │    │   CONTAINER     │
    │                 │    │                 │
    │ Risk checks:    │    │ Records:        │
    │ • Position OK   │    │ • Every tick    │
    │ • Daily limit   │    │ • Every trade   │
    │ • Rate limit    │    │ • Every signal  │
    │                 │    │                 │
    │ If PASS ──────►│───►│ For analysis    │
    │                 │    │                 │
    └────────┬────────┘    └─────────────────┘
             │
             ▼

    ┌─────────────────────────────────────────────┐
    │            KALSHI EXECUTION                 │
    │                                             │
    │   Submit order:                             │
    │   POST /trade-api/v2/orders                 │
    │   {                                         │
    │     "ticker": "KXNFL-26JAN11-BUF",         │
    │     "side": "yes",                          │
    │     "action": "buy",                        │
    │     "count": 25,                            │
    │     "type": "limit",                        │
    │     "yes_price": 48                         │
    │   }                                         │
    │                                             │
    └──────────────────┬──────────────────────────┘
                       │
                       ▼

Step 6: RECORD KEEPING
══════════════════════

    ┌─────────────────────────────────────────────┐
    │           POSTGRESQL CONTAINER              │
    │                                             │
    │   Permanent storage of:                     │
    │                                             │
    │   orders_table:                             │
    │   | id | ticker | side | price | status |   │
    │   | 1  | KXNFL..| yes  | 48    | filled |   │
    │                                             │
    │   positions_table:                          │
    │   | ticker | contracts | avg_price | pnl |  │
    │   | KXNFL..| 25        | 48        | 0   |  │
    │                                             │
    └─────────────────────────────────────────────┘
```

---

## 6. Container Details

### docker-compose.yml Explained

```yaml
version: "3.8"                    # Docker Compose format version

services:                          # List of containers to run

  # ═══════════════════════════════════════════════════════════════════
  # CONTAINER 1: PostgreSQL - Relational Database
  # ═══════════════════════════════════════════════════════════════════
  postgres:
    image: postgres:16-alpine      # Use official PostgreSQL 16 (slim)
    container_name: arb-postgres   # Name shown in `docker ps`

    environment:                   # Database credentials
      POSTGRES_USER: postgres      # Username to connect
      POSTGRES_PASSWORD: postgres  # Password (change in production!)
      POSTGRES_DB: arb_kalshi      # Database name to create

    ports:
      - "5432:5432"               # host:container - access from localhost

    volumes:
      - postgres_data:/var/lib/postgresql/data  # Persist data on restart

    healthcheck:                   # Auto-check if database is ready
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s                # Check every 10 seconds
      timeout: 5s                  # Wait 5 seconds for response
      retries: 5                   # Give up after 5 failures

  # ═══════════════════════════════════════════════════════════════════
  # CONTAINER 2: Redis - In-Memory Cache
  # ═══════════════════════════════════════════════════════════════════
  redis:
    image: redis:7-alpine          # Use official Redis 7 (slim)
    container_name: arb-redis

    ports:
      - "6379:6379"               # Standard Redis port

    volumes:
      - redis_data:/data           # Persist cache on restart

    healthcheck:
      test: ["CMD", "redis-cli", "ping"]  # Simple ping test
      interval: 10s
      timeout: 5s
      retries: 5

  # ═══════════════════════════════════════════════════════════════════
  # CONTAINER 3: QuestDB - Time-Series Database
  # ═══════════════════════════════════════════════════════════════════
  questdb:
    image: questdb/questdb:8.2.1   # Specific QuestDB version
    container_name: arb-questdb

    ports:
      - "9000:9000"   # Web Console - view at http://localhost:9000
      - "9009:9009"   # ILP ingestion - high-speed data writes
      - "8812:8812"   # PostgreSQL wire - SQL queries
      - "9003:9003"   # Health endpoint

    volumes:
      - questdb_data:/var/lib/questdb  # Persist time-series data

    environment:
      - QDB_PG_USER=admin          # Username for SQL queries
      - QDB_PG_PASSWORD=quest      # Password for SQL queries
      - QDB_TELEMETRY_ENABLED=false
      - QDB_HTTP_MIN_ENABLED=true

    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9003"]
      interval: 10s
      timeout: 5s
      retries: 5

# ═════════════════════════════════════════════════════════════════════
# PERSISTENT STORAGE
# ═════════════════════════════════════════════════════════════════════
volumes:
  postgres_data:    # Named volume for PostgreSQL
  redis_data:       # Named volume for Redis
  questdb_data:     # Named volume for QuestDB
```

---

## 7. Network Configuration

### Port Mapping Explained

```
YOUR COMPUTER (HOST)                    DOCKER CONTAINERS
════════════════════                    ═══════════════════

localhost:5432  ─────────────────────►  postgres:5432
                                        (PostgreSQL)

localhost:6379  ─────────────────────►  redis:6379
                                        (Redis)

localhost:9000  ─────────────────────►  questdb:9000
localhost:9009  ─────────────────────►  (QuestDB Web Console)
localhost:8812  ─────────────────────►  (QuestDB ILP)
localhost:9003  ─────────────────────►  (QuestDB SQL)
                                        (QuestDB Health)
```

### Why These Ports?

| Port | Service | Purpose |
|------|---------|---------|
| 5432 | PostgreSQL | Standard PostgreSQL database port |
| 6379 | Redis | Standard Redis cache port |
| 9000 | QuestDB Web | Browser dashboard for queries |
| 9009 | QuestDB ILP | Ultra-fast data ingestion |
| 8812 | QuestDB SQL | SQL queries via PostgreSQL protocol |
| 9003 | QuestDB Health | Health monitoring endpoint |

---

## 8. Getting Started

### Step-by-Step Setup

```
STEP 1: Install Docker Desktop
══════════════════════════════

Download from: https://www.docker.com/products/docker-desktop/

After installation, verify:
$ docker --version
Docker version 24.x.x

$ docker-compose --version
Docker Compose version v2.x.x


STEP 2: Navigate to Project Directory
══════════════════════════════════════

$ cd "C:\Users\jgewi\OneDrive\Attachments\Desktop\Kalshi Version 1\arb-kalshi-sportsbook"


STEP 3: Start All Containers
════════════════════════════

$ docker-compose up -d

This command:
  • Pulls the container images (first time only)
  • Creates the containers
  • Starts them in the background (-d = detached)
  • Sets up the network connections

Expected output:
  [+] Running 4/4
   ✔ Network arb-kalshi-sportsbook_default  Created
   ✔ Container arb-postgres                 Started
   ✔ Container arb-redis                    Started
   ✔ Container arb-questdb                  Started


STEP 4: Verify Containers Are Running
═════════════════════════════════════

$ docker ps

Expected output:
  CONTAINER ID   IMAGE                    STATUS          PORTS
  abc123...      postgres:16-alpine       Up 10 seconds   0.0.0.0:5432->5432/tcp
  def456...      redis:7-alpine           Up 10 seconds   0.0.0.0:6379->6379/tcp
  ghi789...      questdb/questdb:8.2.1    Up 10 seconds   0.0.0.0:9000->9000/tcp...


STEP 5: Create QuestDB Tables
═════════════════════════════

$ python -m app.cli.run_ingest schema

This creates the time-series tables:
  • kalshi_ticks (price updates)
  • kalshi_trades (trade executions)
  • kalshi_orderbook (order book changes)
  • sportsbook_odds (sportsbook prices)
  • arb_opportunities (detected arbitrage)
  • kalshi_markets (market snapshots)


STEP 6: Test the Pipeline
═════════════════════════

$ python -m app.services.arb_pipeline

This runs a test cycle with mock data to verify everything works.


STEP 7: Access QuestDB Dashboard
════════════════════════════════

Open browser: http://localhost:9000

You'll see a SQL interface where you can:
  • Query market data
  • View price history
  • Analyze arbitrage opportunities
```

### Useful Commands Reference

```bash
# ═══════════════════════════════════════════════════════════════
# CONTAINER MANAGEMENT
# ═══════════════════════════════════════════════════════════════

# Start all containers
docker-compose up -d

# Stop all containers
docker-compose down

# Restart all containers
docker-compose restart

# View running containers
docker ps

# View all containers (including stopped)
docker ps -a


# ═══════════════════════════════════════════════════════════════
# VIEWING LOGS
# ═══════════════════════════════════════════════════════════════

# View all logs
docker-compose logs

# View specific container logs
docker logs arb-postgres
docker logs arb-redis
docker logs arb-questdb

# Follow logs in real-time
docker logs -f arb-questdb


# ═══════════════════════════════════════════════════════════════
# DATABASE ACCESS
# ═══════════════════════════════════════════════════════════════

# Connect to PostgreSQL
docker exec -it arb-postgres psql -U postgres -d arb_kalshi

# Connect to Redis
docker exec -it arb-redis redis-cli

# Access QuestDB
# Just open http://localhost:9000 in browser


# ═══════════════════════════════════════════════════════════════
# TROUBLESHOOTING
# ═══════════════════════════════════════════════════════════════

# Check container health
docker inspect --format='{{.State.Health.Status}}' arb-postgres

# View container resource usage
docker stats

# Remove all containers and start fresh
docker-compose down -v  # -v removes volumes too (WARNING: deletes data!)
docker-compose up -d
```

---

## 9. Troubleshooting

### Common Issues and Solutions

#### Issue 1: "Port already in use"
```
Error: Bind for 0.0.0.0:5432 failed: port is already allocated
```

**Solution:** Another service is using that port
```bash
# Find what's using the port (Windows)
netstat -ano | findstr :5432

# Stop the other service, or change the port in docker-compose.yml:
ports:
  - "5433:5432"  # Use 5433 on host instead
```

#### Issue 2: "Container keeps restarting"
```
$ docker ps
CONTAINER ID   STATUS
abc123         Restarting (1) 5 seconds ago
```

**Solution:** Check the logs for errors
```bash
docker logs arb-postgres --tail 50
```

#### Issue 3: "Cannot connect to database"
```
Error: Connection refused to localhost:5432
```

**Solution:**
1. Verify container is running: `docker ps`
2. Check container health: `docker inspect arb-postgres`
3. Wait for container to fully start (health checks)

#### Issue 4: "Data disappeared after restart"

**Solution:** Ensure volumes are properly configured
```bash
# Check if volumes exist
docker volume ls

# Volumes should show:
# arb-kalshi-sportsbook_postgres_data
# arb-kalshi-sportsbook_redis_data
# arb-kalshi-sportsbook_questdb_data
```

---

## Appendix: Visual Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                    KALSHI ARBITRAGE SYSTEM                                  │
│                    Docker Architecture                                      │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  EXTERNAL APIs                  DOCKER CONTAINERS                           │
│  ════════════                   ════════════════                            │
│                                                                             │
│  ┌─────────────┐               ┌─────────────────────────────────────────┐  │
│  │   KALSHI    │               │                                         │  │
│  │  (Markets)  │◄─────────────►│  POSTGRES    REDIS      QUESTDB        │  │
│  └─────────────┘               │  :5432       :6379      :9000/9009     │  │
│                                │                                         │  │
│  ┌─────────────┐               │  ┌───────┐   ┌──────┐   ┌────────┐     │  │
│  │  ODDS API   │               │  │Orders │   │Odds  │   │Ticks   │     │  │
│  │(Sportsbooks)│◄─────────────►│  │Config │   │Cache │   │Trades  │     │  │
│  └─────────────┘               │  │History│   │State │   │Signals │     │  │
│                                │  └───────┘   └──────┘   └────────┘     │  │
│                                │                                         │  │
│                                └─────────────────────────────────────────┘  │
│                                           ▲                                  │
│                                           │                                  │
│                                           ▼                                  │
│                                ┌─────────────────────────────────────────┐  │
│                                │                                         │  │
│                                │          YOUR APPLICATION               │  │
│                                │                                         │  │
│                                │    Detector ──► Breaker ──► Executor   │  │
│                                │                                         │  │
│                                └─────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

QUICK START:
  1. docker-compose up -d          # Start containers
  2. python -m app.cli.run_ingest schema  # Create tables
  3. python -m app.services.arb_pipeline  # Run system
  4. http://localhost:9000         # View QuestDB dashboard
```

---

**Document Version:** 1.0
**Last Updated:** 2026-01-08
**Author:** System Documentation
