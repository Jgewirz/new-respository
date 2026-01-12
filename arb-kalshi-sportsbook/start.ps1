# =============================================================================
# Kalshi Arbitrage System - Windows Start Script
# =============================================================================
# Usage:
#   .\start.ps1              # Start full system
#   .\start.ps1 -Mode infra  # Start infrastructure only
#   .\start.ps1 -Mode test   # Run integration test
#   .\start.ps1 -Mode stop   # Stop all containers
#   .\start.ps1 -Mode logs   # View logs
#   .\start.ps1 -Mode clean  # Stop and remove volumes
# =============================================================================

param(
    [Parameter(Position=0)]
    [ValidateSet("full", "infra", "test", "stop", "logs", "clean", "status", "build")]
    [string]$Mode = "full"
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Colors for output
function Write-Header { param($msg) Write-Host "`n$("=" * 70)" -ForegroundColor Cyan; Write-Host $msg -ForegroundColor Cyan; Write-Host "$("=" * 70)" -ForegroundColor Cyan }
function Write-Step { param($msg) Write-Host "`n[+] $msg" -ForegroundColor Green }
function Write-Info { param($msg) Write-Host "    $msg" -ForegroundColor Gray }
function Write-Warn { param($msg) Write-Host "[!] $msg" -ForegroundColor Yellow }
function Write-Err { param($msg) Write-Host "[X] $msg" -ForegroundColor Red }

# =============================================================================
# PREREQUISITES CHECK
# =============================================================================
function Test-Prerequisites {
    Write-Header "CHECKING PREREQUISITES"

    # Check Docker
    Write-Step "Checking Docker..."
    try {
        $dockerVersion = docker --version
        Write-Info $dockerVersion
    } catch {
        Write-Err "Docker is not installed or not in PATH"
        Write-Info "Download from: https://www.docker.com/products/docker-desktop/"
        exit 1
    }

    # Check Docker Compose
    Write-Step "Checking Docker Compose..."
    try {
        $composeVersion = docker compose version
        Write-Info $composeVersion
    } catch {
        Write-Err "Docker Compose is not available"
        exit 1
    }

    # Check if Docker daemon is running
    Write-Step "Checking Docker daemon..."
    try {
        docker info | Out-Null
        Write-Info "Docker daemon is running"
    } catch {
        Write-Err "Docker daemon is not running. Start Docker Desktop first."
        exit 1
    }

    # Check .env file
    Write-Step "Checking environment configuration..."
    $envFile = Join-Path $ProjectDir ".env"
    if (Test-Path $envFile) {
        Write-Info ".env file found"

        # Check for required variables
        $envContent = Get-Content $envFile -Raw
        if ($envContent -notmatch "KALSHI_KEY_ID=.+") {
            Write-Warn "KALSHI_KEY_ID is not set in .env"
        }
        if ($envContent -notmatch "KALSHI_PRIVATE_KEY_PATH=.+") {
            Write-Warn "KALSHI_PRIVATE_KEY_PATH is not set in .env (WebSocket will fail)"
        }
    } else {
        Write-Warn ".env file not found. Copying from .env.example..."
        $exampleEnv = Join-Path $ProjectDir ".env.example"
        if (Test-Path $exampleEnv) {
            Copy-Item $exampleEnv $envFile
            Write-Info "Created .env from .env.example - please configure it"
        } else {
            Write-Err "No .env.example found"
            exit 1
        }
    }

    # Check keys directory
    Write-Step "Checking keys directory..."
    $keysDir = Join-Path $ProjectDir "keys"
    if (-not (Test-Path $keysDir)) {
        New-Item -ItemType Directory -Path $keysDir | Out-Null
        Write-Info "Created keys/ directory"
    }

    Write-Step "Prerequisites check complete"
}

# =============================================================================
# INFRASTRUCTURE ONLY
# =============================================================================
function Start-Infrastructure {
    Write-Header "STARTING INFRASTRUCTURE"

    Set-Location $ProjectDir

    Write-Step "Starting PostgreSQL, Redis, QuestDB..."
    docker compose -f docker-compose.yml up -d

    Write-Step "Waiting for services to be healthy..."
    $timeout = 60
    $elapsed = 0
    while ($elapsed -lt $timeout) {
        $postgres = docker inspect --format='{{.State.Health.Status}}' arb-postgres 2>$null
        $redis = docker inspect --format='{{.State.Health.Status}}' arb-redis 2>$null
        $questdb = docker inspect --format='{{.State.Health.Status}}' arb-questdb 2>$null

        if ($postgres -eq "healthy" -and $redis -eq "healthy" -and $questdb -eq "healthy") {
            Write-Info "All infrastructure services healthy"
            break
        }

        Write-Info "Waiting... (postgres=$postgres, redis=$redis, questdb=$questdb)"
        Start-Sleep -Seconds 5
        $elapsed += 5
    }

    if ($elapsed -ge $timeout) {
        Write-Warn "Timeout waiting for services. Check docker logs."
    }

    Write-Step "Infrastructure started successfully"
    Write-Info "QuestDB Console: http://localhost:9000"
    Write-Info "PostgreSQL: localhost:5432"
    Write-Info "Redis: localhost:6379"
}

# =============================================================================
# FULL SYSTEM (Infrastructure + Application)
# =============================================================================
function Start-FullSystem {
    Write-Header "STARTING KALSHI ARBITRAGE SYSTEM"

    Set-Location $ProjectDir

    Write-Step "Building application image..."
    docker compose -f docker-compose.full.yml build

    Write-Step "Starting all services..."
    docker compose -f docker-compose.full.yml up -d

    Write-Step "Waiting for services to initialize..."
    Start-Sleep -Seconds 10

    # Check schema init
    Write-Step "Checking schema initialization..."
    $schemaLogs = docker logs arb-schema-init 2>&1
    if ($schemaLogs -match "error|Error|ERROR") {
        Write-Warn "Schema init may have issues. Check: docker logs arb-schema-init"
    } else {
        Write-Info "Schema initialized"
    }

    Write-Step "System started successfully"
    Show-Status
}

# =============================================================================
# RUN INTEGRATION TEST
# =============================================================================
function Start-IntegrationTest {
    Write-Header "RUNNING INTEGRATION TEST"

    Set-Location $ProjectDir

    Write-Step "Starting infrastructure..."
    docker compose -f docker-compose.full.yml up -d postgres redis questdb

    Write-Step "Waiting for infrastructure..."
    Start-Sleep -Seconds 15

    Write-Step "Running integration test (30 seconds)..."
    docker compose -f docker-compose.full.yml --profile test up integration-test

    Write-Step "Test complete. View results above."
}

# =============================================================================
# SHOW STATUS
# =============================================================================
function Show-Status {
    Write-Header "SYSTEM STATUS"

    Write-Step "Running containers:"
    docker ps --filter "name=arb-" --format "table {{.Names}}`t{{.Status}}`t{{.Ports}}"

    Write-Step "Service endpoints:"
    Write-Info "QuestDB Console:  http://localhost:9000"
    Write-Info "PostgreSQL:       localhost:5432 (user: postgres, pass: postgres)"
    Write-Info "Redis:            localhost:6379"

    Write-Step "Quick commands:"
    Write-Info "View logs:        docker logs -f arb-ws-consumer"
    Write-Info "Stop system:      .\start.ps1 -Mode stop"
    Write-Info "Run test:         .\start.ps1 -Mode test"
}

# =============================================================================
# VIEW LOGS
# =============================================================================
function Show-Logs {
    Write-Header "CONTAINER LOGS"

    Set-Location $ProjectDir
    docker compose -f docker-compose.full.yml logs -f --tail 100
}

# =============================================================================
# STOP SYSTEM
# =============================================================================
function Stop-System {
    Write-Header "STOPPING KALSHI ARBITRAGE SYSTEM"

    Set-Location $ProjectDir

    Write-Step "Stopping all containers..."
    docker compose -f docker-compose.full.yml down
    docker compose -f docker-compose.yml down

    Write-Step "System stopped"
}

# =============================================================================
# CLEAN RESET
# =============================================================================
function Reset-System {
    Write-Header "CLEANING SYSTEM (Removing volumes)"

    $confirm = Read-Host "This will DELETE all data. Continue? (y/N)"
    if ($confirm -ne "y") {
        Write-Info "Aborted"
        return
    }

    Set-Location $ProjectDir

    Write-Step "Stopping and removing containers and volumes..."
    docker compose -f docker-compose.full.yml down -v
    docker compose -f docker-compose.yml down -v

    Write-Step "System cleaned. All data removed."
}

# =============================================================================
# BUILD IMAGES
# =============================================================================
function Build-Images {
    Write-Header "BUILDING DOCKER IMAGES"

    Set-Location $ProjectDir

    Write-Step "Building application image..."
    docker compose -f docker-compose.full.yml build --no-cache

    Write-Step "Build complete"
}

# =============================================================================
# MAIN
# =============================================================================
Write-Host @"

  _  __     _     _     _      _         _     _ _
 | |/ /__ _| |___| |__ (_)    / \   _ __| |__ (_) |_ _ __ __ _  __ _  ___
 | ' // _` | / __| '_ \| |   / _ \ | '__| '_ \| | __| '__/ _` |/ _` |/ _ \
 | . \ (_| | \__ \ | | | |  / ___ \| |  | |_) | | |_| | | (_| | (_| |  __/
 |_|\_\__,_|_|___/_| |_|_| /_/   \_\_|  |_.__/|_|\__|_|  \__,_|\__, |\___|
                                                                |___/
                         SYSTEM LAUNCHER v1.0

"@ -ForegroundColor Cyan

switch ($Mode) {
    "full" {
        Test-Prerequisites
        Start-FullSystem
    }
    "infra" {
        Test-Prerequisites
        Start-Infrastructure
    }
    "test" {
        Test-Prerequisites
        Start-IntegrationTest
    }
    "stop" {
        Stop-System
    }
    "logs" {
        Show-Logs
    }
    "clean" {
        Reset-System
    }
    "status" {
        Show-Status
    }
    "build" {
        Build-Images
    }
}
