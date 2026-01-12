#!/bin/bash
# =============================================================================
# Kalshi Arbitrage System - Unix/Linux/Mac Start Script
# =============================================================================
# Usage:
#   ./start.sh              # Start full system
#   ./start.sh infra        # Start infrastructure only
#   ./start.sh test         # Run integration test
#   ./start.sh stop         # Stop all containers
#   ./start.sh logs         # View logs
#   ./start.sh clean        # Stop and remove volumes
#   ./start.sh status       # Show system status
#   ./start.sh build        # Build Docker images
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-full}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

header() { echo -e "\n${CYAN}======================================================================${NC}"; echo -e "${CYAN}$1${NC}"; echo -e "${CYAN}======================================================================${NC}"; }
step() { echo -e "\n${GREEN}[+] $1${NC}"; }
info() { echo -e "    ${NC}$1"; }
warn() { echo -e "${YELLOW}[!] $1${NC}"; }
err() { echo -e "${RED}[X] $1${NC}"; }

# =============================================================================
# PREREQUISITES CHECK
# =============================================================================
check_prerequisites() {
    header "CHECKING PREREQUISITES"

    # Check Docker
    step "Checking Docker..."
    if ! command -v docker &> /dev/null; then
        err "Docker is not installed"
        info "Install from: https://www.docker.com/products/docker-desktop/"
        exit 1
    fi
    info "$(docker --version)"

    # Check Docker Compose
    step "Checking Docker Compose..."
    if ! docker compose version &> /dev/null; then
        err "Docker Compose is not available"
        exit 1
    fi
    info "$(docker compose version)"

    # Check if Docker daemon is running
    step "Checking Docker daemon..."
    if ! docker info &> /dev/null; then
        err "Docker daemon is not running. Start Docker first."
        exit 1
    fi
    info "Docker daemon is running"

    # Check .env file
    step "Checking environment configuration..."
    if [ -f "$SCRIPT_DIR/.env" ]; then
        info ".env file found"

        if ! grep -q "KALSHI_KEY_ID=." "$SCRIPT_DIR/.env"; then
            warn "KALSHI_KEY_ID is not set in .env"
        fi
        if ! grep -q "KALSHI_PRIVATE_KEY_PATH=." "$SCRIPT_DIR/.env"; then
            warn "KALSHI_PRIVATE_KEY_PATH is not set in .env (WebSocket will fail)"
        fi
    else
        warn ".env file not found. Copying from .env.example..."
        if [ -f "$SCRIPT_DIR/.env.example" ]; then
            cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
            info "Created .env from .env.example - please configure it"
        else
            err "No .env.example found"
            exit 1
        fi
    fi

    # Check keys directory
    step "Checking keys directory..."
    if [ ! -d "$SCRIPT_DIR/keys" ]; then
        mkdir -p "$SCRIPT_DIR/keys"
        info "Created keys/ directory"
    fi

    step "Prerequisites check complete"
}

# =============================================================================
# INFRASTRUCTURE ONLY
# =============================================================================
start_infrastructure() {
    header "STARTING INFRASTRUCTURE"

    cd "$SCRIPT_DIR"

    step "Starting PostgreSQL, Redis, QuestDB..."
    docker compose -f docker-compose.yml up -d

    step "Waiting for services to be healthy..."
    timeout=60
    elapsed=0
    while [ $elapsed -lt $timeout ]; do
        postgres=$(docker inspect --format='{{.State.Health.Status}}' arb-postgres 2>/dev/null || echo "unknown")
        redis=$(docker inspect --format='{{.State.Health.Status}}' arb-redis 2>/dev/null || echo "unknown")
        questdb=$(docker inspect --format='{{.State.Health.Status}}' arb-questdb 2>/dev/null || echo "unknown")

        if [ "$postgres" = "healthy" ] && [ "$redis" = "healthy" ] && [ "$questdb" = "healthy" ]; then
            info "All infrastructure services healthy"
            break
        fi

        info "Waiting... (postgres=$postgres, redis=$redis, questdb=$questdb)"
        sleep 5
        elapsed=$((elapsed + 5))
    done

    if [ $elapsed -ge $timeout ]; then
        warn "Timeout waiting for services. Check docker logs."
    fi

    step "Infrastructure started successfully"
    info "QuestDB Console: http://localhost:9000"
    info "PostgreSQL: localhost:5432"
    info "Redis: localhost:6379"
}

# =============================================================================
# FULL SYSTEM (Infrastructure + Application)
# =============================================================================
start_full_system() {
    header "STARTING KALSHI ARBITRAGE SYSTEM"

    cd "$SCRIPT_DIR"

    step "Building application image..."
    docker compose -f docker-compose.full.yml build

    step "Starting all services..."
    docker compose -f docker-compose.full.yml up -d

    step "Waiting for services to initialize..."
    sleep 10

    # Check schema init
    step "Checking schema initialization..."
    if docker logs arb-schema-init 2>&1 | grep -iq "error"; then
        warn "Schema init may have issues. Check: docker logs arb-schema-init"
    else
        info "Schema initialized"
    fi

    step "System started successfully"
    show_status
}

# =============================================================================
# RUN INTEGRATION TEST
# =============================================================================
run_integration_test() {
    header "RUNNING INTEGRATION TEST"

    cd "$SCRIPT_DIR"

    step "Starting infrastructure..."
    docker compose -f docker-compose.full.yml up -d postgres redis questdb

    step "Waiting for infrastructure..."
    sleep 15

    step "Running integration test (30 seconds)..."
    docker compose -f docker-compose.full.yml --profile test up integration-test

    step "Test complete. View results above."
}

# =============================================================================
# SHOW STATUS
# =============================================================================
show_status() {
    header "SYSTEM STATUS"

    step "Running containers:"
    docker ps --filter "name=arb-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

    step "Service endpoints:"
    info "QuestDB Console:  http://localhost:9000"
    info "PostgreSQL:       localhost:5432 (user: postgres, pass: postgres)"
    info "Redis:            localhost:6379"

    step "Quick commands:"
    info "View logs:        docker logs -f arb-ws-consumer"
    info "Stop system:      ./start.sh stop"
    info "Run test:         ./start.sh test"
}

# =============================================================================
# VIEW LOGS
# =============================================================================
show_logs() {
    header "CONTAINER LOGS"

    cd "$SCRIPT_DIR"
    docker compose -f docker-compose.full.yml logs -f --tail 100
}

# =============================================================================
# STOP SYSTEM
# =============================================================================
stop_system() {
    header "STOPPING KALSHI ARBITRAGE SYSTEM"

    cd "$SCRIPT_DIR"

    step "Stopping all containers..."
    docker compose -f docker-compose.full.yml down 2>/dev/null || true
    docker compose -f docker-compose.yml down 2>/dev/null || true

    step "System stopped"
}

# =============================================================================
# CLEAN RESET
# =============================================================================
reset_system() {
    header "CLEANING SYSTEM (Removing volumes)"

    read -p "This will DELETE all data. Continue? (y/N): " confirm
    if [ "$confirm" != "y" ]; then
        info "Aborted"
        return
    fi

    cd "$SCRIPT_DIR"

    step "Stopping and removing containers and volumes..."
    docker compose -f docker-compose.full.yml down -v 2>/dev/null || true
    docker compose -f docker-compose.yml down -v 2>/dev/null || true

    step "System cleaned. All data removed."
}

# =============================================================================
# BUILD IMAGES
# =============================================================================
build_images() {
    header "BUILDING DOCKER IMAGES"

    cd "$SCRIPT_DIR"

    step "Building application image..."
    docker compose -f docker-compose.full.yml build --no-cache

    step "Build complete"
}

# =============================================================================
# MAIN
# =============================================================================
echo -e "${CYAN}"
cat << "EOF"

  _  __     _     _     _      _         _     _ _
 | |/ /__ _| |___| |__ (_)    / \   _ __| |__ (_) |_ _ __ __ _  __ _  ___
 | ' // _` | / __| '_ \| |   / _ \ | '__| '_ \| | __| '__/ _` |/ _` |/ _ \
 | . \ (_| | \__ \ | | | |  / ___ \| |  | |_) | | |_| | | (_| | (_| |  __/
 |_|\_\__,_|_|___/_| |_|_| /_/   \_\_|  |_.__/|_|\__|_|  \__,_|\__, |\___|
                                                                |___/
                         SYSTEM LAUNCHER v1.0

EOF
echo -e "${NC}"

case "$MODE" in
    "full")
        check_prerequisites
        start_full_system
        ;;
    "infra")
        check_prerequisites
        start_infrastructure
        ;;
    "test")
        check_prerequisites
        run_integration_test
        ;;
    "stop")
        stop_system
        ;;
    "logs")
        show_logs
        ;;
    "clean")
        reset_system
        ;;
    "status")
        show_status
        ;;
    "build")
        build_images
        ;;
    *)
        echo "Usage: $0 {full|infra|test|stop|logs|clean|status|build}"
        echo ""
        echo "Commands:"
        echo "  full    - Start full system (infrastructure + application)"
        echo "  infra   - Start infrastructure only (postgres, redis, questdb)"
        echo "  test    - Run integration test"
        echo "  stop    - Stop all containers"
        echo "  logs    - View container logs"
        echo "  clean   - Stop and remove all data (volumes)"
        echo "  status  - Show system status"
        echo "  build   - Build Docker images"
        exit 1
        ;;
esac
