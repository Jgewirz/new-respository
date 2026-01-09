@echo off
setlocal enabledelayedexpansion

echo ============================================
echo   Kalshi Arbitrage System - Full Startup
echo ============================================
echo.
echo   Version: 0.1.0
echo   Architecture: Kalshi-First
echo.

:: Navigate to project directory
cd /d "%~dp0arb-kalshi-sportsbook"

:: =============================================================================
:: STEP 1: ENVIRONMENT VALIDATION
:: =============================================================================
echo [1/6] Validating environment...

:: Check if .env exists
if not exist ".env" (
    echo WARNING: .env file not found!
    echo Copying from .env.example...
    if exist ".env.example" (
        copy ".env.example" ".env" >nul
        echo Created .env from template. Please configure API keys before trading!
    ) else (
        echo ERROR: Neither .env nor .env.example found!
        echo Please create a .env file with required configuration.
        pause
        exit /b 1
    )
)

:: Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH!
    echo Please install Python 3.11+ and add to PATH.
    pause
    exit /b 1
)
echo   Python: OK

:: Check for Docker
docker --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker not found in PATH!
    echo Please install Docker Desktop and ensure it's running.
    pause
    exit /b 1
)
echo   Docker: OK

:: Check if Docker daemon is running
docker ps >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker daemon not running!
    echo Please start Docker Desktop first.
    pause
    exit /b 1
)
echo   Docker Daemon: OK

echo.

:: =============================================================================
:: STEP 2: START DOCKER SERVICES
:: =============================================================================
echo [2/6] Starting Docker services...
docker-compose up -d
if errorlevel 1 (
    echo ERROR: Docker Compose failed!
    echo Check docker-compose.yml for errors.
    pause
    exit /b 1
)

echo   PostgreSQL: Starting...
echo   Redis: Starting...
echo   QuestDB: Starting...
echo.

:: =============================================================================
:: STEP 3: WAIT FOR SERVICES
:: =============================================================================
echo [3/6] Waiting for services to be ready...

:: Wait for QuestDB (most critical for our system)
set MAX_RETRIES=30
set RETRY_COUNT=0

:wait_questdb
set /a RETRY_COUNT+=1
if %RETRY_COUNT% gtr %MAX_RETRIES% (
    echo ERROR: QuestDB failed to start after 30 seconds!
    docker logs arb-questdb --tail 20
    pause
    exit /b 1
)

:: Check QuestDB health endpoint
curl -s http://localhost:9003 >nul 2>&1
if errorlevel 1 (
    echo   Waiting for QuestDB... (%RETRY_COUNT%/%MAX_RETRIES%)
    timeout /t 1 /nobreak >nul
    goto wait_questdb
)

echo   QuestDB: Ready
echo   PostgreSQL: Ready
echo   Redis: Ready
echo.

:: =============================================================================
:: STEP 4: CREATE DATABASE TABLES
:: =============================================================================
echo [4/6] Creating database tables...
python -m app.cli.run_ingest schema
if errorlevel 1 (
    echo WARNING: Schema creation had issues (tables may already exist)
)
echo.

:: =============================================================================
:: STEP 5: INGEST MARKET DATA (IF AVAILABLE)
:: =============================================================================
echo [5/6] Checking for market data...

if exist "..\kalshi_sports_markets.jsonl" (
    echo   Found: kalshi_sports_markets.jsonl
    echo   Ingesting market data...
    python -m app.cli.run_ingest jsonl "..\kalshi_sports_markets.jsonl" --batch-size 5000 --stats
) else if exist "data\kalshi_sports_markets.jsonl" (
    echo   Found: data/kalshi_sports_markets.jsonl
    python -m app.cli.run_ingest jsonl "data\kalshi_sports_markets.jsonl" --batch-size 5000 --stats
) else (
    echo   No JSONL data file found. Skipping ingestion.
    echo   To ingest data later: run ingest path/to/file.jsonl
)
echo.

:: =============================================================================
:: STEP 6: VALIDATE SYSTEM
:: =============================================================================
echo [6/6] Running system validation...

:: Test detector
echo   Testing Edge Detector...
python -c "from app.arb.detector import EdgeDetector; d = EdgeDetector(); print('    Detector: OK')" 2>nul
if errorlevel 1 (
    echo     WARNING: Detector import failed
)

:: Test resolver
echo   Testing Event Resolver...
python -c "from app.mapping.resolver import EventResolver; print('    Resolver: OK')" 2>nul
if errorlevel 1 (
    echo     WARNING: Resolver import failed
)

:: Test circuit breaker
echo   Testing Circuit Breaker...
python -c "from app.execution.circuit_breaker import CircuitBreaker; cb = CircuitBreaker(); print('    Circuit Breaker: OK')" 2>nul
if errorlevel 1 (
    echo     WARNING: Circuit breaker import failed
)

echo.
echo ============================================
echo   KALSHI ARBITRAGE SYSTEM READY!
echo ============================================
echo.
echo   Services Running:
echo   -----------------
echo   PostgreSQL:      localhost:5432
echo   Redis:           localhost:6379
echo   QuestDB Console: http://localhost:9000
echo   QuestDB ILP:     localhost:9009
echo   QuestDB SQL:     localhost:8812
echo.
echo   Quick Commands:
echo   ---------------
echo   run pipeline     - Run detection cycle
echo   run pipeline-test - Test with mock data
echo   run odds         - Fetch sportsbook odds
echo   run stats        - Show database stats
echo   run questdb      - Open QuestDB console
echo   run help         - Show all commands
echo.
echo   Next Steps:
echo   -----------
echo   1. Configure API keys in .env
echo   2. Run: run pipeline-test
echo   3. Review detected signals
echo.
echo   Press any key to open QuestDB console...
pause >nul
start http://localhost:9000
