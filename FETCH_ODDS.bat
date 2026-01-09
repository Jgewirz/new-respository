@echo off
echo ============================================
echo   Fetching Sportsbook Odds from Odds API
echo ============================================
echo.

cd /d "%~dp0arb-kalshi-sportsbook"

:: Check if Docker is running
docker ps >nul 2>&1
if errorlevel 1 (
    echo Docker not running. Starting services...
    docker-compose up -d
    timeout /t 5 /nobreak >nul
)

:: Fetch odds
echo Fetching odds from DraftKings, FanDuel, BetMGM, Caesars...
echo.
python -m app.services.odds_ingest

echo.
echo ============================================
echo   Done! Odds stored in Redis + QuestDB
echo ============================================
echo.
pause
