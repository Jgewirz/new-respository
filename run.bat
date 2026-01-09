@echo off
setlocal enabledelayedexpansion

:: ============================================================================
:: Kalshi Arbitrage System - Command Runner
:: ============================================================================
:: Usage: run.bat [command] [args...]
::
:: Core Philosophy: Kalshi-First Architecture
:: All detection flows originate from Kalshi markets, which are then
:: mapped to sportsbook events for edge detection.
:: ============================================================================

set COMMAND=%1
set ARG2=%2
set ARG3=%3
set ARG4=%4

if "%COMMAND%"=="" goto :help

:: Navigate to project directory
cd /d "%~dp0arb-kalshi-sportsbook"

:: Route commands
if "%COMMAND%"=="help" goto :help
if "%COMMAND%"=="-h" goto :help
if "%COMMAND%"=="--help" goto :help

:: Setup commands
if "%COMMAND%"=="install" goto :install
if "%COMMAND%"=="install-dev" goto :install-dev
if "%COMMAND%"=="env" goto :env

:: Docker commands
if "%COMMAND%"=="docker-up" goto :docker-up
if "%COMMAND%"=="docker-down" goto :docker-down
if "%COMMAND%"=="docker-restart" goto :docker-restart
if "%COMMAND%"=="docker-logs" goto :docker-logs
if "%COMMAND%"=="docker-ps" goto :docker-ps
if "%COMMAND%"=="docker-clean" goto :docker-clean

:: Database commands
if "%COMMAND%"=="schema" goto :schema
if "%COMMAND%"=="ingest" goto :ingest
if "%COMMAND%"=="ingest-dry" goto :ingest-dry
if "%COMMAND%"=="query" goto :query
if "%COMMAND%"=="stats" goto :stats

:: Pipeline commands (NEW - Core functionality)
if "%COMMAND%"=="pipeline" goto :pipeline
if "%COMMAND%"=="pipeline-test" goto :pipeline-test
if "%COMMAND%"=="detect" goto :detect
if "%COMMAND%"=="detector" goto :detector

:: Odds API commands
if "%COMMAND%"=="odds" goto :odds
if "%COMMAND%"=="odds-nfl" goto :odds-nfl
if "%COMMAND%"=="odds-nba" goto :odds-nba
if "%COMMAND%"=="odds-continuous" goto :odds-continuous
if "%COMMAND%"=="odds-stats" goto :odds-stats

:: Resolver commands (NEW)
if "%COMMAND%"=="resolver" goto :resolver
if "%COMMAND%"=="map" goto :resolver

:: Testing commands
if "%COMMAND%"=="test" goto :test
if "%COMMAND%"=="test-cov" goto :test-cov
if "%COMMAND%"=="test-detector" goto :test-detector
if "%COMMAND%"=="test-breaker" goto :test-breaker

:: Code quality commands
if "%COMMAND%"=="lint" goto :lint
if "%COMMAND%"=="format" goto :format
if "%COMMAND%"=="typecheck" goto :typecheck
if "%COMMAND%"=="check" goto :check

:: Database client commands
if "%COMMAND%"=="questdb" goto :questdb
if "%COMMAND%"=="redis-cli" goto :redis-cli
if "%COMMAND%"=="psql" goto :psql

:: Status commands (NEW)
if "%COMMAND%"=="status" goto :status
if "%COMMAND%"=="health" goto :health

echo Unknown command: %COMMAND%
echo Run "run help" for available commands.
exit /b 1

:: =============================================================================
:: HELP
:: =============================================================================
:help
echo.
echo ============================================================================
echo   KALSHI ARBITRAGE SYSTEM - Command Runner
echo ============================================================================
echo.
echo   CORE WORKFLOW:
echo   --------------
echo   1. START.bat          - Initialize entire system (Docker + DB + validation)
echo   2. run pipeline-test  - Test detection with mock data
echo   3. run odds           - Fetch sportsbook odds into Redis
echo   4. run pipeline       - Run live detection cycle
echo.
echo   SETUP:
echo   ------
echo   run install           Install production dependencies (uv sync)
echo   run install-dev       Install with dev tools (pytest, ruff, mypy)
echo   run env               Show current environment configuration
echo.
echo   DOCKER:
echo   -------
echo   run docker-up         Start all services (Postgres, Redis, QuestDB)
echo   run docker-down       Stop all services
echo   run docker-restart    Restart all services
echo   run docker-logs       View container logs (live)
echo   run docker-ps         Show running containers
echo   run docker-clean      Remove all containers and volumes (DESTRUCTIVE)
echo.
echo   DATABASE:
echo   ---------
echo   run schema            Create QuestDB tables
echo   run ingest FILE       Ingest JSONL file into QuestDB
echo   run ingest-dry FILE   Validate JSONL without writing
echo   run query             Query latest market data
echo   run stats             Show ingestion statistics
echo.
echo   PIPELINE (Core Detection):
echo   --------------------------
echo   run pipeline          Run detection cycle (requires Kalshi client)
echo   run pipeline-test     Test pipeline with mock Kalshi data
echo   run detector          Validate detector with test cases
echo   run resolver          Test event resolver/mapping
echo.
echo   ODDS API:
echo   ---------
echo   run odds              Fetch all sports odds (single run)
echo   run odds-nfl          Fetch NFL odds only
echo   run odds-nba          Fetch NBA odds only
echo   run odds-continuous   Poll odds every 30s (Ctrl+C to stop)
echo   run odds-stats        Show Redis odds cache statistics
echo.
echo   TESTING:
echo   --------
echo   run test              Run all tests
echo   run test-cov          Run tests with coverage report
echo   run test-detector     Run detector-specific tests
echo   run test-breaker      Run circuit breaker tests
echo.
echo   CODE QUALITY:
echo   -------------
echo   run lint              Run ruff linter
echo   run format            Format code with ruff
echo   run typecheck         Run mypy type checker (strict)
echo   run check             Run all checks (lint + typecheck + test)
echo.
echo   DATABASE CLIENTS:
echo   -----------------
echo   run questdb           Open QuestDB web console
echo   run redis-cli         Open Redis CLI
echo   run psql              Open PostgreSQL CLI
echo.
echo   STATUS:
echo   -------
echo   run status            Show system status
echo   run health            Check all service health
echo.
echo ============================================================================
echo   TRADING PROFILES (set via TRADING_PROFILE env var):
echo   - CONSERVATIVE: 6c min edge, 0.15 Kelly, safest
echo   - STANDARD:     5c min edge, 0.25 Kelly, balanced (default)
echo   - AGGRESSIVE:   3c min edge, 0.35 Kelly, more trades
echo   - VERY_AGGRESSIVE: 2c min edge, 0.50 Kelly, high volume
echo   - ANY_EDGE:     1c min edge, 0.50 Kelly, maximum capture
echo   - SCALP:        1c min edge, 0.75 Kelly, tight spreads only
echo ============================================================================
echo.
goto :eof

:: =============================================================================
:: SETUP COMMANDS
:: =============================================================================
:install
echo Installing production dependencies...
uv sync
goto :eof

:install-dev
echo Installing with dev tools...
uv sync --all-extras
goto :eof

:env
echo.
echo Current Environment Configuration:
echo -----------------------------------
if exist ".env" (
    echo .env file exists
    echo.
    type .env | findstr /v "^#" | findstr /v "^$"
) else (
    echo WARNING: .env file not found!
)
goto :eof

:: =============================================================================
:: DOCKER COMMANDS
:: =============================================================================
:docker-up
echo Starting Docker services...
docker-compose up -d
echo.
echo Services:
echo   PostgreSQL: localhost:5432
echo   Redis:      localhost:6379
echo   QuestDB:    localhost:9000 (Web), 9009 (ILP), 8812 (SQL)
goto :eof

:docker-down
echo Stopping Docker services...
docker-compose down
goto :eof

:docker-restart
echo Restarting Docker services...
docker-compose down
docker-compose up -d
goto :eof

:docker-logs
echo Showing container logs (Ctrl+C to exit)...
docker-compose logs -f
goto :eof

:docker-ps
echo.
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
goto :eof

:docker-clean
echo WARNING: This will remove ALL containers and volumes!
echo Press Ctrl+C to cancel, or any key to continue...
pause >nul
docker-compose down -v
echo Cleaned.
goto :eof

:: =============================================================================
:: DATABASE COMMANDS
:: =============================================================================
:schema
echo Creating QuestDB tables...
python -m app.cli.run_ingest schema
goto :eof

:ingest
if "%ARG2%"=="" (
    echo Error: Please specify a JSONL file
    echo Usage: run ingest path/to/file.jsonl
    goto :eof
)
echo Ingesting %ARG2%...
python -m app.cli.run_ingest jsonl "%ARG2%" --batch-size 5000 --stats
goto :eof

:ingest-dry
if "%ARG2%"=="" (
    echo Error: Please specify a JSONL file
    echo Usage: run ingest-dry path/to/file.jsonl
    goto :eof
)
echo Validating %ARG2% (dry run)...
python -m app.cli.run_ingest jsonl "%ARG2%" --dry-run
goto :eof

:query
echo Querying latest data...
python -m app.cli.run_ingest query --limit 20
goto :eof

:stats
echo Showing statistics...
python -m app.cli.run_ingest query --stats
goto :eof

:: =============================================================================
:: PIPELINE COMMANDS (CORE DETECTION)
:: =============================================================================
:pipeline
echo.
echo ============================================
echo   Running Arbitrage Detection Pipeline
echo ============================================
echo.
echo Profile: %TRADING_PROFILE%
if "%TRADING_PROFILE%"=="" echo Profile: STANDARD (default)
echo.
python -m app.services.arb_pipeline
goto :eof

:pipeline-test
echo.
echo ============================================
echo   Running Pipeline Test (Mock Kalshi Data)
echo ============================================
echo.
python -m app.services.arb_pipeline
goto :eof

:detect
echo.
echo Running detection cycle...
python -m app.services.arb_pipeline
goto :eof

:detector
echo.
echo ============================================
echo   Edge Detector Validation
echo ============================================
echo.
python -m app.arb.detector
goto :eof

:resolver
echo.
echo ============================================
echo   Event Resolver Test
echo ============================================
echo.
python -m app.mapping.resolver
goto :eof

:: =============================================================================
:: ODDS API COMMANDS
:: =============================================================================
:odds
echo Fetching all sports odds...
python -m app.services.odds_ingest
goto :eof

:odds-nfl
echo Fetching NFL odds...
python -m app.services.odds_ingest --sport americanfootball_nfl
goto :eof

:odds-nba
echo Fetching NBA odds...
python -m app.services.odds_ingest --sport basketball_nba
goto :eof

:odds-continuous
echo Running continuous odds polling (Ctrl+C to stop)...
python -m app.services.odds_ingest --continuous --interval 30
goto :eof

:odds-stats
echo Redis odds statistics...
python -m app.services.odds_ingest --stats
goto :eof

:: =============================================================================
:: TESTING COMMANDS
:: =============================================================================
:test
echo Running all tests...
pytest tests/ -v
goto :eof

:test-cov
echo Running tests with coverage...
pytest tests/ --cov=app --cov-report=html --cov-report=term
echo.
echo Coverage report: htmlcov/index.html
goto :eof

:test-detector
echo Running detector tests...
pytest tests/test_detector.py -v 2>nul
if errorlevel 1 (
    echo Note: test_detector.py may not exist yet
    python -m app.arb.detector
)
goto :eof

:test-breaker
echo Running circuit breaker tests...
pytest tests/test_circuit_breaker.py -v
goto :eof

:: =============================================================================
:: CODE QUALITY COMMANDS
:: =============================================================================
:lint
echo Running ruff linter...
ruff check app/
goto :eof

:format
echo Formatting code...
ruff format app/
goto :eof

:typecheck
echo Running mypy type checker...
mypy app/ --strict
goto :eof

:check
echo Running all checks...
echo.
echo === LINT ===
ruff check app/
echo.
echo === FORMAT CHECK ===
ruff format --check app/
echo.
echo === TYPECHECK ===
mypy app/ --strict
echo.
echo === TESTS ===
pytest tests/ -v
goto :eof

:: =============================================================================
:: DATABASE CLIENT COMMANDS
:: =============================================================================
:questdb
echo Opening QuestDB web console...
start http://localhost:9000
goto :eof

:redis-cli
echo Connecting to Redis...
docker exec -it arb-redis redis-cli
goto :eof

:psql
echo Connecting to PostgreSQL...
docker exec -it arb-postgres psql -U postgres -d arb_kalshi
goto :eof

:: =============================================================================
:: STATUS COMMANDS
:: =============================================================================
:status
echo.
echo ============================================
echo   KALSHI ARBITRAGE SYSTEM STATUS
echo ============================================
echo.
echo Docker Services:
docker ps --format "  {{.Names}}: {{.Status}}" 2>nul
if errorlevel 1 (
    echo   Docker not running or no containers
)
echo.
echo Configuration:
if exist ".env" (
    echo   .env: Found
) else (
    echo   .env: MISSING
)
echo   Profile: %TRADING_PROFILE%
if "%TRADING_PROFILE%"=="" echo   Profile: STANDARD (default)
echo.
echo Quick Health Check:
curl -s http://localhost:9003 >nul 2>&1
if errorlevel 1 (
    echo   QuestDB: DOWN
) else (
    echo   QuestDB: UP
)
docker exec arb-redis redis-cli ping >nul 2>&1
if errorlevel 1 (
    echo   Redis: DOWN
) else (
    echo   Redis: UP
)
docker exec arb-postgres pg_isready -U postgres >nul 2>&1
if errorlevel 1 (
    echo   PostgreSQL: DOWN
) else (
    echo   PostgreSQL: UP
)
echo.
goto :eof

:health
echo.
echo Checking service health...
echo.

echo [QuestDB]
curl -s http://localhost:9003 >nul 2>&1
if errorlevel 1 (
    echo   Status: DOWN
    echo   Action: Run "run docker-up" or check Docker Desktop
) else (
    echo   Status: UP
    echo   Console: http://localhost:9000
    echo   ILP: localhost:9009
    echo   SQL: localhost:8812
)
echo.

echo [Redis]
docker exec arb-redis redis-cli ping >nul 2>&1
if errorlevel 1 (
    echo   Status: DOWN
) else (
    echo   Status: UP
    for /f %%i in ('docker exec arb-redis redis-cli dbsize 2^>nul') do echo   Keys: %%i
)
echo.

echo [PostgreSQL]
docker exec arb-postgres pg_isready -U postgres >nul 2>&1
if errorlevel 1 (
    echo   Status: DOWN
) else (
    echo   Status: UP
)
echo.
goto :eof
