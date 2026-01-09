# Kalshi Execution Layer - Complete Implementation Plan

**Document Version:** 1.0
**Created:** 2026-01-08
**Scope:** Paper Executor + Live Kalshi Executor + WebSocket Client
**Estimated Effort:** 40-60 hours

---

## Executive Summary

This plan details the implementation of the execution layer for the Kalshi Arbitrage System. The execution layer is the critical missing component that bridges edge detection (implemented) to actual trade execution.

### Implementation Order

```
Phase 1: Base Executor Interface (4 hours)
    ↓
Phase 2: Paper Executor (8 hours) ← ENABLES TESTING
    ↓
Phase 3: Kalshi WebSocket Client (12 hours)
    ↓
Phase 4: Kalshi Live Executor (16 hours) ← ENABLES LIVE TRADING
    ↓
Phase 5: Execution Service Orchestrator (8 hours)
    ↓
Phase 6: CLI Runner (4 hours)
    ↓
Phase 7: Integration Tests (8 hours)
```

---

## Phase 1: Base Executor Interface

**File:** `app/execution/base.py`
**Effort:** 4 hours
**Dependencies:** `app/execution/models.py` (exists)

### Purpose
Define the abstract interface that all executors must implement. This ensures Paper and Live executors are interchangeable.

### Design

```python
# app/execution/base.py

from abc import ABC, abstractmethod
from typing import Optional
from app.execution.models import ExecutionOrder, ExecutionResult, Fill
from app.arb.detector import Signal


class ExecutorBase(ABC):
    """
    Abstract base class for order executors.

    All executors (Paper, Live, Demo) must implement this interface.
    This allows the ExecutionService to swap executors without code changes.
    """

    @property
    @abstractmethod
    def mode(self) -> str:
        """Return executor mode: 'paper', 'live', or 'demo'."""
        pass

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if executor can accept orders."""
        pass

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection (WebSocket, API session, etc.)."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Clean shutdown."""
        pass

    @abstractmethod
    async def execute(self, order: ExecutionOrder) -> ExecutionResult:
        """
        Execute an order and return result.

        This is the main entry point. Implementations should:
        1. Submit order to exchange/simulator
        2. Wait for fills or timeout
        3. Return ExecutionResult with final state
        """
        pass

    @abstractmethod
    async def cancel(self, order: ExecutionOrder) -> ExecutionResult:
        """Cancel an open order."""
        pass

    @abstractmethod
    async def get_position(self, ticker: str) -> dict:
        """Get current position for a ticker."""
        pass

    @abstractmethod
    async def get_balance(self) -> int:
        """Get available balance in cents."""
        pass

    # Optional hooks for subclasses
    async def on_fill(self, order: ExecutionOrder, fill: Fill) -> None:
        """Called when a fill is received. Override for custom handling."""
        pass

    async def on_order_update(self, order: ExecutionOrder) -> None:
        """Called when order state changes. Override for logging/tracking."""
        pass


class ExecutorConfig:
    """Configuration for executors."""

    def __init__(
        self,
        mode: str = "paper",
        fill_probability: float = 0.95,  # Paper only
        fill_delay_ms: float = 50.0,     # Paper only
        slippage_cents: int = 1,         # Paper only
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
    ):
        self.mode = mode
        self.fill_probability = fill_probability
        self.fill_delay_ms = fill_delay_ms
        self.slippage_cents = slippage_cents
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    @classmethod
    def from_env(cls) -> "ExecutorConfig":
        """Load from environment variables."""
        import os
        return cls(
            mode=os.environ.get("EXECUTION_MODE", "paper"),
            fill_probability=float(os.environ.get("PAPER_FILL_PROB", "0.95")),
            fill_delay_ms=float(os.environ.get("PAPER_FILL_DELAY_MS", "50")),
            slippage_cents=int(os.environ.get("PAPER_SLIPPAGE_CENTS", "1")),
            timeout_seconds=float(os.environ.get("EXECUTION_TIMEOUT", "30")),
            max_retries=int(os.environ.get("EXECUTION_MAX_RETRIES", "3")),
        )
```

### Tests Required
- [ ] `test_executor_config_from_env()`
- [ ] `test_executor_config_defaults()`

---

## Phase 2: Paper Executor

**File:** `app/execution/paper_executor.py`
**Effort:** 8 hours
**Dependencies:** Phase 1, `circuit_breaker.py`, `models.py`

### Purpose
Simulated executor for backtesting and strategy validation WITHOUT real money.

### Key Features
1. **Realistic Fill Simulation** - Configurable probability, delay, slippage
2. **Position Tracking** - In-memory position state
3. **Balance Management** - Virtual balance with P&L tracking
4. **Market Data Integration** - Uses real Kalshi prices for realistic simulation
5. **Full ExecutionResult Output** - Same format as live executor

### Design

```python
# app/execution/paper_executor.py

"""
Paper Trading Executor - Simulated Execution for Testing

Provides realistic trade simulation with:
- Configurable fill probability and latency
- Price slippage modeling
- Position and balance tracking
- Full ExecutionResult compatibility

Usage:
    from app.execution.paper_executor import PaperExecutor

    executor = PaperExecutor(initial_balance=10000_00)  # $10,000 in cents

    async with executor:
        result = await executor.execute(order)
        print(result.summary())
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Dict

from app.execution.base import ExecutorBase, ExecutorConfig
from app.execution.models import (
    ExecutionOrder, ExecutionResult, Fill, OrderState, ExecutionMode,
    now_ns, generate_id
)
from app.execution.circuit_breaker import CircuitBreaker


@dataclass
class PaperPosition:
    """Tracked position in paper trading."""
    ticker: str
    yes_contracts: int = 0
    no_contracts: int = 0
    avg_yes_price: float = 0.0
    avg_no_price: float = 0.0
    total_cost_cents: int = 0
    realized_pnl_cents: int = 0


@dataclass
class PaperExecutorState:
    """Internal state for paper executor."""
    balance_cents: int = 10000_00  # $10,000 default
    positions: Dict[str, PaperPosition] = field(default_factory=dict)
    orders: Dict[str, ExecutionOrder] = field(default_factory=dict)
    fills: list[Fill] = field(default_factory=list)
    total_trades: int = 0
    total_filled: int = 0
    total_rejected: int = 0
    realized_pnl_cents: int = 0


class PaperExecutor(ExecutorBase):
    """
    Paper trading executor for strategy testing.

    Simulates order execution with configurable:
    - Fill probability (default 95%)
    - Fill latency (default 50ms)
    - Price slippage (default 1 cent)

    All positions and balances are tracked in memory.
    Results can be exported for analysis.
    """

    def __init__(
        self,
        config: Optional[ExecutorConfig] = None,
        initial_balance: int = 10000_00,
        circuit_breaker: Optional[CircuitBreaker] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize paper executor.

        Args:
            config: Executor configuration
            initial_balance: Starting balance in cents
            circuit_breaker: Optional risk controls
            seed: Random seed for reproducibility
        """
        self.config = config or ExecutorConfig()
        self._state = PaperExecutorState(balance_cents=initial_balance)
        self._circuit_breaker = circuit_breaker or CircuitBreaker()
        self._connected = False
        self._lock = asyncio.Lock()

        # For reproducible testing
        if seed is not None:
            random.seed(seed)

    # =========================================================================
    # ExecutorBase Implementation
    # =========================================================================

    @property
    def mode(self) -> str:
        return "paper"

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        """Initialize paper executor."""
        self._connected = True

    async def disconnect(self) -> None:
        """Cleanup paper executor."""
        self._connected = False

    async def __aenter__(self) -> "PaperExecutor":
        await self.connect()
        return self

    async def __aexit__(self, *args) -> None:
        await self.disconnect()

    async def execute(self, order: ExecutionOrder) -> ExecutionResult:
        """
        Execute order in paper trading mode.

        Simulation logic:
        1. Validate order against circuit breaker
        2. Check balance for buy orders
        3. Simulate fill based on probability
        4. Apply slippage to fill price
        5. Update positions and balance
        6. Return ExecutionResult
        """
        async with self._lock:
            # Mark order as paper mode
            order.mode = ExecutionMode.PAPER

            # Step 1: Circuit breaker check
            risk_cents = order.contracts * order.limit_price
            check = self._circuit_breaker.check_trade(
                ticker=order.ticker,
                contracts=order.contracts,
                risk_cents=risk_cents,
            )

            if not check.allowed:
                order.mark_rejected(code="CIRCUIT_BREAKER", message=check.reason)
                return ExecutionResult.from_order(order, fills=[])

            # Step 2: Balance check for buys
            if order.action == "buy":
                required = order.contracts * order.limit_price
                if required > self._state.balance_cents:
                    order.mark_rejected(
                        code="INSUFFICIENT_FUNDS",
                        message=f"Need {required}c, have {self._state.balance_cents}c"
                    )
                    self._state.total_rejected += 1
                    return ExecutionResult.from_order(order, fills=[])

            # Step 3: Simulate submission
            order.mark_submitted(kalshi_order_id=f"PAPER-{generate_id()[:8]}")

            # Simulate network latency
            await asyncio.sleep(self.config.fill_delay_ms / 1000)

            # Step 4: Determine if order fills
            fills = []
            if random.random() < self.config.fill_probability:
                # Order fills - apply slippage
                fill_price = self._calculate_fill_price(order)

                fill = order.record_fill(
                    contracts=order.contracts,
                    price=fill_price,
                    is_maker=False,
                )
                fills.append(fill)

                # Update state
                self._update_position(order, fill)
                self._update_balance(order, fill)
                self._state.fills.append(fill)
                self._state.total_filled += 1

                # Notify circuit breaker
                result = ExecutionResult.from_order(order, fills=fills)
                self._circuit_breaker.record_trade(result)

            else:
                # Order doesn't fill - simulate timeout/cancel
                order.mark_expired()

            self._state.orders[order.execution_id] = order
            self._state.total_trades += 1

            return ExecutionResult.from_order(order, fills=fills)

    async def cancel(self, order: ExecutionOrder) -> ExecutionResult:
        """Cancel a paper order."""
        async with self._lock:
            if order.is_active:
                order.mark_canceled(reason="User requested cancellation")
            return ExecutionResult.from_order(order, fills=[])

    async def get_position(self, ticker: str) -> dict:
        """Get position for ticker."""
        pos = self._state.positions.get(ticker, PaperPosition(ticker=ticker))
        return {
            "ticker": ticker,
            "yes_contracts": pos.yes_contracts,
            "no_contracts": pos.no_contracts,
            "avg_yes_price": pos.avg_yes_price,
            "avg_no_price": pos.avg_no_price,
        }

    async def get_balance(self) -> int:
        """Get current balance in cents."""
        return self._state.balance_cents

    # =========================================================================
    # Paper-Specific Methods
    # =========================================================================

    def _calculate_fill_price(self, order: ExecutionOrder) -> int:
        """
        Calculate fill price with slippage.

        For buys: fill at limit + slippage (worse for us)
        For sells: fill at limit - slippage (worse for us)
        """
        base_price = order.limit_price

        if order.action == "buy":
            # Buying: pay more (slippage against us)
            fill_price = min(99, base_price + self.config.slippage_cents)
        else:
            # Selling: receive less
            fill_price = max(1, base_price - self.config.slippage_cents)

        return fill_price

    def _update_position(self, order: ExecutionOrder, fill: Fill) -> None:
        """Update position tracking after fill."""
        ticker = order.ticker

        if ticker not in self._state.positions:
            self._state.positions[ticker] = PaperPosition(ticker=ticker)

        pos = self._state.positions[ticker]

        if order.side == "yes":
            if order.action == "buy":
                # Buying YES
                total_cost = pos.yes_contracts * pos.avg_yes_price + fill.notional_cents
                pos.yes_contracts += fill.contracts
                pos.avg_yes_price = total_cost / pos.yes_contracts if pos.yes_contracts else 0
            else:
                # Selling YES
                pos.yes_contracts -= fill.contracts
                if pos.yes_contracts <= 0:
                    pos.yes_contracts = 0
                    pos.avg_yes_price = 0
        else:
            if order.action == "buy":
                # Buying NO
                total_cost = pos.no_contracts * pos.avg_no_price + fill.notional_cents
                pos.no_contracts += fill.contracts
                pos.avg_no_price = total_cost / pos.no_contracts if pos.no_contracts else 0
            else:
                # Selling NO
                pos.no_contracts -= fill.contracts
                if pos.no_contracts <= 0:
                    pos.no_contracts = 0
                    pos.avg_no_price = 0

    def _update_balance(self, order: ExecutionOrder, fill: Fill) -> None:
        """Update balance after fill."""
        if order.action == "buy":
            # Buying costs money
            self._state.balance_cents -= fill.notional_cents
        else:
            # Selling returns money
            self._state.balance_cents += fill.notional_cents

    def settle_position(self, ticker: str, outcome: str) -> int:
        """
        Settle a position when market resolves.

        Args:
            ticker: Market ticker
            outcome: "yes" or "no"

        Returns:
            P&L in cents
        """
        if ticker not in self._state.positions:
            return 0

        pos = self._state.positions[ticker]
        pnl = 0

        if outcome == "yes":
            # YES wins: YES contracts pay $1, NO contracts worth $0
            pnl += pos.yes_contracts * 100  # Win $1 per contract
            pnl -= pos.yes_contracts * int(pos.avg_yes_price)  # Subtract cost basis
            pnl -= pos.no_contracts * int(pos.avg_no_price)  # NO contracts lost
        else:
            # NO wins: NO contracts pay $1, YES contracts worth $0
            pnl += pos.no_contracts * 100
            pnl -= pos.no_contracts * int(pos.avg_no_price)
            pnl -= pos.yes_contracts * int(pos.avg_yes_price)

        self._state.balance_cents += pnl
        self._state.realized_pnl_cents += pnl

        # Clear position
        del self._state.positions[ticker]

        return pnl

    def get_stats(self) -> dict:
        """Get paper trading statistics."""
        return {
            "balance_cents": self._state.balance_cents,
            "balance_dollars": self._state.balance_cents / 100,
            "total_trades": self._state.total_trades,
            "total_filled": self._state.total_filled,
            "total_rejected": self._state.total_rejected,
            "fill_rate": (
                self._state.total_filled / self._state.total_trades
                if self._state.total_trades > 0 else 0
            ),
            "realized_pnl_cents": self._state.realized_pnl_cents,
            "realized_pnl_dollars": self._state.realized_pnl_cents / 100,
            "open_positions": len(self._state.positions),
            "positions": {
                k: {
                    "yes": v.yes_contracts,
                    "no": v.no_contracts,
                }
                for k, v in self._state.positions.items()
            },
        }

    def reset(self, initial_balance: int = 10000_00) -> None:
        """Reset paper trading state."""
        self._state = PaperExecutorState(balance_cents=initial_balance)


# =============================================================================
# Factory
# =============================================================================

def create_paper_executor(
    initial_balance: int = 10000_00,
    **kwargs
) -> PaperExecutor:
    """Factory function for paper executor."""
    return PaperExecutor(initial_balance=initial_balance, **kwargs)


# =============================================================================
# CLI Test
# =============================================================================

async def test_paper_executor():
    """Test paper executor with sample orders."""
    from app.arb.detector import KalshiMarket, SportsbookConsensus, EdgeDetector

    print("=" * 60)
    print("PAPER EXECUTOR TEST")
    print("=" * 60)
    print()

    # Create executor
    executor = PaperExecutor(initial_balance=10000_00, seed=42)
    await executor.connect()

    print(f"Initial Balance: ${executor._state.balance_cents / 100:.2f}")
    print()

    # Create a mock signal
    kalshi = KalshiMarket(
        ticker="KXNFL-TEST-BUF",
        yes_bid=45,
        yes_ask=47,
        no_bid=53,
        no_ask=55,
        volume=1000,
    )

    # Create mock order
    order = ExecutionOrder(
        ticker="KXNFL-TEST-BUF",
        side="yes",
        action="buy",
        limit_price=48,
        contracts=25,
    )

    print(f"Executing order: BUY 25 YES @ 48c")
    result = await executor.execute(order)

    print(f"Result: {result.summary()}")
    print()
    print(f"Stats: {executor.get_stats()}")

    await executor.disconnect()


if __name__ == "__main__":
    asyncio.run(test_paper_executor())
```

### Tests Required
- [ ] `test_paper_executor_buy_yes()`
- [ ] `test_paper_executor_buy_no()`
- [ ] `test_paper_executor_insufficient_funds()`
- [ ] `test_paper_executor_circuit_breaker_block()`
- [ ] `test_paper_executor_slippage()`
- [ ] `test_paper_executor_settlement()`
- [ ] `test_paper_executor_stats()`

---

## Phase 3: Kalshi WebSocket Client

**File:** `app/connectors/kalshi/websocket.py`
**Effort:** 12 hours
**Dependencies:** `auth.py` (exists)

### Purpose
Real-time WebSocket connection for:
1. Order fill notifications
2. Position updates
3. Market data streaming (optional)

### Kalshi WebSocket Reference

Based on [Kalshi WebSocket Documentation](https://docs.kalshi.com/getting_started/quick_start_websockets):

- **Production URL:** `wss://api.elections.kalshi.com/trade-api/ws/v2`
- **Demo URL:** `wss://demo-api.kalshi.co/trade-api/ws/v2`
- **Authentication:** Same headers as REST (KALSHI-ACCESS-KEY, KALSHI-ACCESS-SIGNATURE, KALSHI-ACCESS-TIMESTAMP)
- **Channels:** ticker, orderbook, trades, fills

### Design

```python
# app/connectors/kalshi/websocket.py

"""
Kalshi WebSocket Client - Real-Time Order and Market Data

Provides:
- Authenticated WebSocket connection to Kalshi
- Fill notifications for live orders
- Position update streaming
- Market data subscriptions (optional)

Usage:
    from app.connectors.kalshi.websocket import KalshiWebSocket

    async def on_fill(fill_data):
        print(f"Fill received: {fill_data}")

    ws = KalshiWebSocket.from_env()
    ws.on_fill = on_fill

    async with ws:
        await ws.subscribe_fills()
        await asyncio.sleep(3600)  # Run for 1 hour
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Any, Awaitable

import websockets
from websockets.client import WebSocketClientProtocol

from app.connectors.kalshi.auth import KalshiAuth


# =============================================================================
# CONFIGURATION
# =============================================================================

KALSHI_WS_PROD = "wss://api.elections.kalshi.com/trade-api/ws/v2"
KALSHI_WS_DEMO = "wss://demo-api.kalshi.co/trade-api/ws/v2"

HEARTBEAT_INTERVAL = 30  # seconds
RECONNECT_DELAY = 5  # seconds
MAX_RECONNECT_ATTEMPTS = 10


# =============================================================================
# MESSAGE TYPES
# =============================================================================

class MessageType(str, Enum):
    """WebSocket message types."""
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    FILL = "fill"
    ORDER_UPDATE = "order_update"
    POSITION_UPDATE = "position_update"
    TICKER = "ticker"
    ORDERBOOK = "orderbook"
    TRADE = "trade"
    ERROR = "error"
    SUBSCRIBED = "subscribed"
    HEARTBEAT = "heartbeat"


@dataclass
class FillMessage:
    """Parsed fill notification."""
    order_id: str
    ticker: str
    side: str
    action: str
    price: int
    count: int
    is_taker: bool
    trade_id: str
    timestamp: int

    @classmethod
    def from_dict(cls, data: dict) -> "FillMessage":
        return cls(
            order_id=data.get("order_id", ""),
            ticker=data.get("ticker", ""),
            side=data.get("side", ""),
            action=data.get("action", ""),
            price=data.get("yes_price") or data.get("no_price") or 0,
            count=data.get("count", 0),
            is_taker=data.get("is_taker", True),
            trade_id=data.get("trade_id", ""),
            timestamp=data.get("created_time", int(time.time() * 1000)),
        )


@dataclass
class OrderUpdateMessage:
    """Parsed order update notification."""
    order_id: str
    ticker: str
    status: str
    remaining_count: int
    filled_count: int
    timestamp: int

    @classmethod
    def from_dict(cls, data: dict) -> "OrderUpdateMessage":
        return cls(
            order_id=data.get("order_id", ""),
            ticker=data.get("ticker", ""),
            status=data.get("status", ""),
            remaining_count=data.get("remaining_count", 0),
            filled_count=data.get("filled_count", 0),
            timestamp=data.get("updated_time", int(time.time() * 1000)),
        )


# =============================================================================
# WEBSOCKET CLIENT
# =============================================================================

class KalshiWebSocket:
    """
    Kalshi WebSocket client for real-time updates.

    Handles:
    - Connection management with auto-reconnect
    - Authentication via RSA-PSS
    - Message routing to callbacks
    - Heartbeat/keepalive
    """

    def __init__(
        self,
        auth: KalshiAuth,
        ws_url: str = KALSHI_WS_PROD,
        auto_reconnect: bool = True,
    ):
        """
        Initialize WebSocket client.

        Args:
            auth: KalshiAuth for signing connection
            ws_url: WebSocket URL
            auto_reconnect: Enable automatic reconnection
        """
        self.auth = auth
        self.ws_url = ws_url
        self.auto_reconnect = auto_reconnect

        self._ws: Optional[WebSocketClientProtocol] = None
        self._connected = False
        self._running = False
        self._reconnect_count = 0

        # Message handlers
        self._handlers: dict[str, list[Callable]] = {
            MessageType.FILL.value: [],
            MessageType.ORDER_UPDATE.value: [],
            MessageType.POSITION_UPDATE.value: [],
            MessageType.TICKER.value: [],
            MessageType.ERROR.value: [],
        }

        # Tasks
        self._recv_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None

        # Subscriptions
        self._subscriptions: set[str] = set()

    @classmethod
    def from_env(cls, demo: bool = False) -> "KalshiWebSocket":
        """Create WebSocket client from environment."""
        auth = KalshiAuth.from_env()

        # Check for demo mode
        if demo or os.environ.get("PAPER_TRADING", "").lower() == "true":
            ws_url = KALSHI_WS_DEMO
        else:
            ws_url = os.environ.get("KALSHI_WS_URL", KALSHI_WS_PROD)

        return cls(auth=auth, ws_url=ws_url)

    # =========================================================================
    # Connection Management
    # =========================================================================

    async def connect(self) -> None:
        """Establish WebSocket connection."""
        # Get authentication headers
        headers = self._get_auth_headers()

        try:
            self._ws = await websockets.connect(
                self.ws_url,
                additional_headers=headers,
                ping_interval=HEARTBEAT_INTERVAL,
                ping_timeout=10,
            )
            self._connected = True
            self._reconnect_count = 0

            # Start background tasks
            self._running = True
            self._recv_task = asyncio.create_task(self._receive_loop())

            print(f"[WebSocket] Connected to {self.ws_url}")

            # Resubscribe after reconnect
            for sub in self._subscriptions:
                await self._send_subscribe(sub)

        except Exception as e:
            print(f"[WebSocket] Connection failed: {e}")
            raise

    async def disconnect(self) -> None:
        """Close WebSocket connection."""
        self._running = False

        if self._recv_task:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass

        if self._ws:
            await self._ws.close()
            self._ws = None

        self._connected = False
        print("[WebSocket] Disconnected")

    async def __aenter__(self) -> "KalshiWebSocket":
        await self.connect()
        return self

    async def __aexit__(self, *args) -> None:
        await self.disconnect()

    def _get_auth_headers(self) -> dict:
        """Generate authentication headers for WebSocket connection."""
        # WebSocket path for signing
        path = "/trade-api/ws/v2"
        headers = self.auth.get_headers("GET", path)
        return headers

    async def _reconnect(self) -> None:
        """Attempt to reconnect after disconnect."""
        if not self.auto_reconnect:
            return

        while self._running and self._reconnect_count < MAX_RECONNECT_ATTEMPTS:
            self._reconnect_count += 1
            print(f"[WebSocket] Reconnecting ({self._reconnect_count}/{MAX_RECONNECT_ATTEMPTS})...")

            await asyncio.sleep(RECONNECT_DELAY * self._reconnect_count)

            try:
                await self.connect()
                return
            except Exception as e:
                print(f"[WebSocket] Reconnect failed: {e}")

        print("[WebSocket] Max reconnect attempts reached")

    # =========================================================================
    # Message Handling
    # =========================================================================

    async def _receive_loop(self) -> None:
        """Background task to receive and route messages."""
        while self._running and self._ws:
            try:
                message = await self._ws.recv()
                await self._handle_message(message)

            except websockets.ConnectionClosed:
                print("[WebSocket] Connection closed")
                self._connected = False
                await self._reconnect()
                break

            except Exception as e:
                print(f"[WebSocket] Receive error: {e}")

    async def _handle_message(self, raw: str) -> None:
        """Parse and route incoming message."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            print(f"[WebSocket] Invalid JSON: {raw[:100]}")
            return

        msg_type = data.get("type", "")

        # Route to handlers
        if msg_type in self._handlers:
            for handler in self._handlers[msg_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(data)
                    else:
                        handler(data)
                except Exception as e:
                    print(f"[WebSocket] Handler error: {e}")

        # Special handling for fills
        if msg_type == MessageType.FILL.value:
            fill = FillMessage.from_dict(data)
            await self._on_fill(fill)

        elif msg_type == MessageType.ORDER_UPDATE.value:
            update = OrderUpdateMessage.from_dict(data)
            await self._on_order_update(update)

    async def _on_fill(self, fill: FillMessage) -> None:
        """Internal fill handler - can be overridden."""
        print(f"[WebSocket] Fill: {fill.ticker} {fill.side} {fill.count}@{fill.price}")

    async def _on_order_update(self, update: OrderUpdateMessage) -> None:
        """Internal order update handler - can be overridden."""
        print(f"[WebSocket] Order Update: {update.order_id} -> {update.status}")

    # =========================================================================
    # Subscriptions
    # =========================================================================

    async def _send(self, message: dict) -> None:
        """Send JSON message."""
        if not self._ws or not self._connected:
            raise RuntimeError("Not connected")
        await self._ws.send(json.dumps(message))

    async def _send_subscribe(self, channel: str, **params) -> None:
        """Send subscription request."""
        msg = {
            "type": MessageType.SUBSCRIBE.value,
            "channel": channel,
            **params,
        }
        await self._send(msg)

    async def subscribe_fills(self) -> None:
        """Subscribe to fill notifications for all orders."""
        await self._send_subscribe("fills")
        self._subscriptions.add("fills")

    async def subscribe_orders(self) -> None:
        """Subscribe to order updates."""
        await self._send_subscribe("orders")
        self._subscriptions.add("orders")

    async def subscribe_positions(self) -> None:
        """Subscribe to position updates."""
        await self._send_subscribe("positions")
        self._subscriptions.add("positions")

    async def subscribe_ticker(self, ticker: str) -> None:
        """Subscribe to market ticker updates."""
        await self._send_subscribe("ticker", ticker=ticker)
        self._subscriptions.add(f"ticker:{ticker}")

    async def subscribe_orderbook(self, ticker: str) -> None:
        """Subscribe to orderbook updates."""
        await self._send_subscribe("orderbook", ticker=ticker)
        self._subscriptions.add(f"orderbook:{ticker}")

    # =========================================================================
    # Handler Registration
    # =========================================================================

    def on_fill(self, handler: Callable[[dict], Any]) -> None:
        """Register fill notification handler."""
        self._handlers[MessageType.FILL.value].append(handler)

    def on_order_update(self, handler: Callable[[dict], Any]) -> None:
        """Register order update handler."""
        self._handlers[MessageType.ORDER_UPDATE.value].append(handler)

    def on_position_update(self, handler: Callable[[dict], Any]) -> None:
        """Register position update handler."""
        self._handlers[MessageType.POSITION_UPDATE.value].append(handler)

    def on_ticker(self, handler: Callable[[dict], Any]) -> None:
        """Register ticker handler."""
        self._handlers[MessageType.TICKER.value].append(handler)

    def on_error(self, handler: Callable[[dict], Any]) -> None:
        """Register error handler."""
        self._handlers[MessageType.ERROR.value].append(handler)

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def subscriptions(self) -> set[str]:
        return self._subscriptions.copy()


# =============================================================================
# Test
# =============================================================================

async def test_websocket():
    """Test WebSocket connection."""
    print("=" * 60)
    print("KALSHI WEBSOCKET TEST")
    print("=" * 60)

    try:
        ws = KalshiWebSocket.from_env(demo=True)

        async with ws:
            print(f"Connected: {ws.is_connected}")

            # Subscribe to fills
            await ws.subscribe_fills()
            await ws.subscribe_orders()

            print("Subscribed to fills and orders")
            print("Waiting for messages (10 seconds)...")

            await asyncio.sleep(10)

            print(f"Subscriptions: {ws.subscriptions}")

        print("Test complete")

    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_websocket())
```

### Tests Required
- [ ] `test_websocket_connect()`
- [ ] `test_websocket_auth_headers()`
- [ ] `test_websocket_subscribe_fills()`
- [ ] `test_websocket_message_routing()`
- [ ] `test_websocket_reconnect()`

---

## Phase 4: Kalshi Live Executor

**File:** `app/execution/kalshi_executor.py`
**Effort:** 16 hours
**Dependencies:** Phases 1-3, `client.py`

### Purpose
Production executor that places real orders on Kalshi.

### Key Features
1. **REST API for Order Submission** - Uses existing `KalshiClient`
2. **WebSocket for Fill Monitoring** - Real-time fill notifications
3. **Order State Management** - Track pending/resting/filled states
4. **Timeout Handling** - Cancel unfilled orders after timeout
5. **Position Validation** - Verify positions after fills

### Design

```python
# app/execution/kalshi_executor.py

"""
Kalshi Live Executor - Production Order Execution

Executes real orders against Kalshi API with:
- REST API for order placement
- WebSocket for fill notifications
- Automatic timeout and cancellation
- Position verification

IMPORTANT: This executes REAL trades with REAL money.
Use PaperExecutor for testing.

Usage:
    from app.execution.kalshi_executor import KalshiExecutor

    async with KalshiExecutor.from_env() as executor:
        result = await executor.execute(order)
        print(result.summary())
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Dict

from app.execution.base import ExecutorBase, ExecutorConfig
from app.execution.models import (
    ExecutionOrder, ExecutionResult, Fill, OrderState, ExecutionMode,
    now_ns
)
from app.execution.circuit_breaker import CircuitBreaker
from app.connectors.kalshi.client import (
    KalshiClient, KalshiAsyncClient, Order, KalshiAPIError,
    InsufficientFundsError, MarketClosedError, OrderRejectedError
)
from app.connectors.kalshi.websocket import KalshiWebSocket, FillMessage


class KalshiExecutor(ExecutorBase):
    """
    Production executor for Kalshi trades.

    Uses:
    - KalshiAsyncClient for order submission
    - KalshiWebSocket for fill notifications
    - CircuitBreaker for risk controls

    Order Flow:
    1. Pre-submission risk checks (circuit breaker)
    2. Submit order via REST API
    3. Wait for fill via WebSocket or polling
    4. Update ExecutionOrder with fill details
    5. Return ExecutionResult
    """

    def __init__(
        self,
        client: KalshiAsyncClient,
        websocket: Optional[KalshiWebSocket] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
        config: Optional[ExecutorConfig] = None,
    ):
        """
        Initialize Kalshi executor.

        Args:
            client: Async Kalshi API client
            websocket: WebSocket for fill notifications (optional)
            circuit_breaker: Risk controls
            config: Execution configuration
        """
        self.client = client
        self.websocket = websocket
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.config = config or ExecutorConfig()

        self._connected = False

        # Track pending orders for fill matching
        self._pending_orders: Dict[str, ExecutionOrder] = {}
        self._fill_events: Dict[str, asyncio.Event] = {}

    @classmethod
    def from_env(cls, use_websocket: bool = True) -> "KalshiExecutor":
        """
        Create executor from environment.

        Args:
            use_websocket: Enable WebSocket for fill notifications
        """
        client = KalshiAsyncClient.from_env()

        websocket = None
        if use_websocket:
            try:
                websocket = KalshiWebSocket.from_env()
            except Exception as e:
                print(f"[KalshiExecutor] WebSocket init failed: {e}")

        circuit_breaker = CircuitBreaker.from_env()
        config = ExecutorConfig.from_env()

        return cls(
            client=client,
            websocket=websocket,
            circuit_breaker=circuit_breaker,
            config=config,
        )

    # =========================================================================
    # ExecutorBase Implementation
    # =========================================================================

    @property
    def mode(self) -> str:
        return "live"

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        """Connect to Kalshi API and WebSocket."""
        # Verify REST API connection
        try:
            balance = await self.client.get_balance()
            print(f"[KalshiExecutor] Connected. Balance: ${balance.balance_dollars:.2f}")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Kalshi API: {e}")

        # Connect WebSocket if available
        if self.websocket:
            try:
                await self.websocket.connect()
                await self.websocket.subscribe_fills()
                await self.websocket.subscribe_orders()

                # Register fill handler
                self.websocket.on_fill(self._handle_fill)

                print("[KalshiExecutor] WebSocket connected")
            except Exception as e:
                print(f"[KalshiExecutor] WebSocket failed: {e}")
                # Continue without WebSocket - will use polling

        self._connected = True

    async def disconnect(self) -> None:
        """Disconnect from Kalshi."""
        if self.websocket:
            await self.websocket.disconnect()

        await self.client.close()
        self._connected = False

    async def __aenter__(self) -> "KalshiExecutor":
        await self.connect()
        return self

    async def __aexit__(self, *args) -> None:
        await self.disconnect()

    async def execute(self, order: ExecutionOrder) -> ExecutionResult:
        """
        Execute order on Kalshi.

        Steps:
        1. Circuit breaker pre-check
        2. Submit order via REST
        3. Wait for fill (WebSocket or polling)
        4. Build and return result
        """
        order.mode = ExecutionMode.LIVE

        # Step 1: Circuit breaker
        risk_cents = order.contracts * order.limit_price
        check = self.circuit_breaker.check_trade(
            ticker=order.ticker,
            contracts=order.contracts,
            risk_cents=risk_cents,
        )

        if not check.allowed:
            order.mark_rejected(code="CIRCUIT_BREAKER", message=check.reason)
            return ExecutionResult.from_order(order, fills=[])

        # Step 2: Submit order
        try:
            kalshi_order = await self._submit_order(order)
            order.mark_submitted(kalshi_order_id=kalshi_order.order_id)
        except InsufficientFundsError as e:
            order.mark_rejected(code="INSUFFICIENT_FUNDS", message=str(e))
            return ExecutionResult.from_order(order, fills=[])
        except MarketClosedError as e:
            order.mark_rejected(code="MARKET_CLOSED", message=str(e))
            return ExecutionResult.from_order(order, fills=[])
        except OrderRejectedError as e:
            order.mark_rejected(code="ORDER_REJECTED", message=str(e))
            return ExecutionResult.from_order(order, fills=[])
        except KalshiAPIError as e:
            order.mark_rejected(code="API_ERROR", message=str(e))
            return ExecutionResult.from_order(order, fills=[])

        # Step 3: Wait for fill
        fills = await self._wait_for_fill(order)

        # Step 4: Build result
        result = ExecutionResult.from_order(order, fills=fills)

        # Record in circuit breaker
        if result.success or result.partial:
            self.circuit_breaker.record_trade(result)

        return result

    async def cancel(self, order: ExecutionOrder) -> ExecutionResult:
        """Cancel an open order."""
        if not order.kalshi_order_id:
            order.mark_canceled(reason="No Kalshi order ID")
            return ExecutionResult.from_order(order, fills=[])

        try:
            await self.client.cancel_order(order.kalshi_order_id)
            order.mark_canceled(reason="User requested")
        except KalshiAPIError as e:
            # Order may have already filled
            updated = await self.client.get_order(order.kalshi_order_id)
            if updated.is_filled:
                order.mark_filled()

        return ExecutionResult.from_order(order, fills=[])

    async def get_position(self, ticker: str) -> dict:
        """Get current position."""
        position = await self.client.get_position(ticker)
        if position:
            return {
                "ticker": ticker,
                "yes_contracts": position.yes_count,
                "no_contracts": position.no_count,
            }
        return {"ticker": ticker, "yes_contracts": 0, "no_contracts": 0}

    async def get_balance(self) -> int:
        """Get available balance in cents."""
        balance = await self.client.get_balance()
        return balance.balance

    # =========================================================================
    # Internal Methods
    # =========================================================================

    async def _submit_order(self, order: ExecutionOrder) -> Order:
        """Submit order to Kalshi API."""
        # Determine price parameter
        if order.side == "yes":
            return await self.client.create_order(
                ticker=order.ticker,
                side=order.side,
                action=order.action,
                count=order.contracts,
                type=order.order_type,
                yes_price=order.limit_price,
                client_order_id=order.client_order_id,
            )
        else:
            return await self.client.create_order(
                ticker=order.ticker,
                side=order.side,
                action=order.action,
                count=order.contracts,
                type=order.order_type,
                no_price=order.limit_price,
                client_order_id=order.client_order_id,
            )

    async def _wait_for_fill(self, order: ExecutionOrder) -> list[Fill]:
        """Wait for order to fill or timeout."""
        fills = []

        # Create event for WebSocket fill notification
        self._pending_orders[order.kalshi_order_id] = order
        self._fill_events[order.kalshi_order_id] = asyncio.Event()

        try:
            # Wait for fill with timeout
            await asyncio.wait_for(
                self._fill_events[order.kalshi_order_id].wait(),
                timeout=self.config.timeout_seconds,
            )

            # Fill received via WebSocket
            if order.state == OrderState.FILLED:
                # Get fills from order
                pass  # Fills added by WebSocket handler

        except asyncio.TimeoutError:
            # Timeout - poll for final status
            fills = await self._poll_order_status(order)

            # If still not filled, cancel
            if not order.is_terminal:
                await self.cancel(order)

        finally:
            # Cleanup
            self._pending_orders.pop(order.kalshi_order_id, None)
            self._fill_events.pop(order.kalshi_order_id, None)

        return fills

    async def _poll_order_status(self, order: ExecutionOrder) -> list[Fill]:
        """Poll Kalshi API for order status."""
        fills = []

        try:
            kalshi_order = await self.client.get_order(order.kalshi_order_id)

            if kalshi_order.is_filled:
                # Create fill from order data
                fill = order.record_fill(
                    contracts=kalshi_order.filled_count,
                    price=kalshi_order.price_cents,
                    is_maker=False,
                )
                fills.append(fill)

            elif kalshi_order.is_canceled:
                order.mark_canceled()

            elif kalshi_order.filled_count > 0:
                # Partial fill
                fill = order.record_fill(
                    contracts=kalshi_order.filled_count,
                    price=kalshi_order.price_cents,
                    is_maker=False,
                )
                fills.append(fill)

        except KalshiAPIError as e:
            print(f"[KalshiExecutor] Poll error: {e}")

        return fills

    async def _handle_fill(self, data: dict) -> None:
        """Handle fill notification from WebSocket."""
        order_id = data.get("order_id")

        if order_id not in self._pending_orders:
            return

        order = self._pending_orders[order_id]

        # Record fill
        fill = order.record_fill(
            contracts=data.get("count", 0),
            price=data.get("yes_price") or data.get("no_price") or 0,
            is_maker=not data.get("is_taker", True),
        )

        # Signal fill received
        if order_id in self._fill_events:
            self._fill_events[order_id].set()

        await self.on_fill(order, fill)

    async def on_fill(self, order: ExecutionOrder, fill: Fill) -> None:
        """Called when fill is received. Override for custom handling."""
        print(f"[KalshiExecutor] Fill: {order.ticker} {fill.contracts}@{fill.price}c")


# =============================================================================
# Factory
# =============================================================================

def create_executor(mode: str = "paper", **kwargs):
    """
    Factory function to create appropriate executor.

    Args:
        mode: "paper" or "live"

    Returns:
        ExecutorBase instance
    """
    if mode == "paper":
        from app.execution.paper_executor import PaperExecutor
        return PaperExecutor(**kwargs)
    elif mode == "live":
        return KalshiExecutor.from_env(**kwargs)
    else:
        raise ValueError(f"Unknown executor mode: {mode}")


# =============================================================================
# Test
# =============================================================================

async def test_kalshi_executor():
    """Test Kalshi executor (demo mode)."""
    print("=" * 60)
    print("KALSHI EXECUTOR TEST (DEMO)")
    print("=" * 60)

    # Force demo mode
    os.environ["PAPER_TRADING"] = "true"

    try:
        executor = KalshiExecutor.from_env(use_websocket=False)

        async with executor:
            # Check balance
            balance = await executor.get_balance()
            print(f"Balance: ${balance / 100:.2f}")

            # Create test order (small)
            order = ExecutionOrder(
                ticker="KXTEST-DEMO",  # Use a demo market
                side="yes",
                action="buy",
                limit_price=50,
                contracts=1,
            )

            print(f"Would execute: BUY 1 YES @ 50c")
            print("(Skipping actual execution in test)")

        print("Test complete")

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_kalshi_executor())
```

### Tests Required
- [ ] `test_kalshi_executor_connect()`
- [ ] `test_kalshi_executor_submit_order()`
- [ ] `test_kalshi_executor_wait_for_fill()`
- [ ] `test_kalshi_executor_timeout_cancel()`
- [ ] `test_kalshi_executor_circuit_breaker()`
- [ ] `test_kalshi_executor_websocket_fill()`

---

## Phase 5: Execution Service Orchestrator

**File:** `app/services/execution_service.py`
**Effort:** 8 hours
**Dependencies:** Phases 1-4

### Purpose
High-level service that coordinates:
1. Signal → Order conversion
2. Executor selection (paper vs live)
3. Batch execution
4. Result aggregation and logging

### Design

```python
# app/services/execution_service.py

"""
Execution Service - Orchestrates Signal to Order Flow

Provides:
- Signal → ExecutionOrder conversion
- Executor management (paper/live)
- Batch execution with concurrency control
- Result logging to QuestDB

Usage:
    from app.services.execution_service import ExecutionService

    service = ExecutionService.from_env()

    async with service:
        # Execute single signal
        result = await service.execute_signal(signal)

        # Execute batch
        results = await service.execute_batch(signals, max_concurrent=5)
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Optional, List

from app.execution.base import ExecutorBase, ExecutorConfig
from app.execution.models import ExecutionOrder, ExecutionResult, ExecutionMode
from app.execution.circuit_breaker import CircuitBreaker
from app.arb.detector import Signal
from app.data.questdb import QuestDBILPClient


@dataclass
class ExecutionStats:
    """Execution session statistics."""
    signals_received: int = 0
    orders_submitted: int = 0
    orders_filled: int = 0
    orders_rejected: int = 0
    total_cost_cents: int = 0
    total_potential_profit_cents: int = 0

    def to_dict(self) -> dict:
        return {
            "signals_received": self.signals_received,
            "orders_submitted": self.orders_submitted,
            "orders_filled": self.orders_filled,
            "orders_rejected": self.orders_rejected,
            "fill_rate": self.orders_filled / self.orders_submitted if self.orders_submitted else 0,
            "total_cost_dollars": self.total_cost_cents / 100,
            "total_potential_profit_dollars": self.total_potential_profit_cents / 100,
        }


class ExecutionService:
    """
    High-level execution orchestration service.

    Coordinates:
    - Executor lifecycle (connect/disconnect)
    - Signal to order conversion
    - Execution with circuit breaker
    - Result logging
    """

    def __init__(
        self,
        executor: ExecutorBase,
        circuit_breaker: Optional[CircuitBreaker] = None,
        log_to_questdb: bool = True,
    ):
        self.executor = executor
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.log_to_questdb = log_to_questdb

        self._stats = ExecutionStats()
        self._running = False

    @classmethod
    def from_env(cls) -> "ExecutionService":
        """Create service from environment."""
        mode = os.environ.get("EXECUTION_MODE", "paper")

        if mode == "live":
            from app.execution.kalshi_executor import KalshiExecutor
            executor = KalshiExecutor.from_env()
        else:
            from app.execution.paper_executor import PaperExecutor
            initial_balance = int(os.environ.get("PAPER_INITIAL_BALANCE", "10000_00"))
            executor = PaperExecutor(initial_balance=initial_balance)

        circuit_breaker = CircuitBreaker.from_env()
        log_enabled = os.environ.get("LOG_EXECUTIONS", "true").lower() == "true"

        return cls(
            executor=executor,
            circuit_breaker=circuit_breaker,
            log_to_questdb=log_enabled,
        )

    async def __aenter__(self) -> "ExecutionService":
        await self.start()
        return self

    async def __aexit__(self, *args) -> None:
        await self.stop()

    async def start(self) -> None:
        """Start execution service."""
        await self.executor.connect()
        self._running = True
        print(f"[ExecutionService] Started in {self.executor.mode} mode")

    async def stop(self) -> None:
        """Stop execution service."""
        self._running = False
        await self.executor.disconnect()
        print(f"[ExecutionService] Stopped. Stats: {self._stats.to_dict()}")

    async def execute_signal(self, signal: Signal) -> ExecutionResult:
        """
        Execute a single signal.

        Args:
            signal: Detection signal to execute

        Returns:
            ExecutionResult with order outcome
        """
        self._stats.signals_received += 1

        # Skip signals that shouldn't trade
        if not signal.should_trade:
            order = ExecutionOrder(
                signal_id=signal.signal_id,
                ticker=signal.kalshi.ticker,
            )
            order.mark_rejected(code="SIGNAL_NO_TRADE", message=signal.reason)
            return ExecutionResult.from_order(order, fills=[])

        # Convert signal to order
        order = ExecutionOrder.from_signal(
            signal,
            mode=ExecutionMode.PAPER if self.executor.mode == "paper" else ExecutionMode.LIVE,
        )

        self._stats.orders_submitted += 1

        # Execute
        result = await self.executor.execute(order)

        # Update stats
        if result.success:
            self._stats.orders_filled += 1
            self._stats.total_cost_cents += result.total_cost_cents
            self._stats.total_potential_profit_cents += result.potential_profit_cents
        elif result.failed:
            self._stats.orders_rejected += 1

        # Log to QuestDB
        if self.log_to_questdb:
            await self._log_execution(result)

        return result

    async def execute_batch(
        self,
        signals: List[Signal],
        max_concurrent: int = 5,
    ) -> List[ExecutionResult]:
        """
        Execute batch of signals with concurrency control.

        Args:
            signals: List of signals to execute
            max_concurrent: Max parallel executions

        Returns:
            List of ExecutionResults
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def execute_with_limit(sig: Signal) -> ExecutionResult:
            async with semaphore:
                return await self.execute_signal(sig)

        tasks = [execute_with_limit(sig) for sig in signals]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        valid_results = []
        for r in results:
            if isinstance(r, ExecutionResult):
                valid_results.append(r)
            else:
                print(f"[ExecutionService] Execution error: {r}")

        return valid_results

    async def _log_execution(self, result: ExecutionResult) -> None:
        """Log execution result to QuestDB."""
        try:
            with QuestDBILPClient() as ilp:
                # Log to executions table
                order = result.order

                tags = f"ticker={order.ticker},side={order.side},mode={order.mode.value}"
                fields = (
                    f"execution_id=\"{order.execution_id}\","
                    f"signal_id=\"{order.signal_id}\","
                    f"contracts={order.contracts}i,"
                    f"limit_price={order.limit_price}i,"
                    f"filled_contracts={order.filled_contracts}i,"
                    f"avg_fill_price={order.average_fill_price},"
                    f"state=\"{order.state.value}\","
                    f"edge_cents={order.signal_edge_cents}i,"
                    f"latency_ms={order.total_latency_ms}"
                )

                line = f"executions,{tags} {fields} {order.completed_at_ns or order.created_at_ns}"
                ilp._send(line)

        except Exception as e:
            print(f"[ExecutionService] QuestDB log error: {e}")

    @property
    def stats(self) -> ExecutionStats:
        return self._stats

    def reset_stats(self) -> None:
        self._stats = ExecutionStats()
```

---

## Phase 6: CLI Runner

**File:** `app/cli/run_executor.py`
**Effort:** 4 hours
**Dependencies:** Phase 5

### Purpose
Command-line interface for running the execution system.

### Design

```python
# app/cli/run_executor.py

"""
Execution CLI - Command Line Interface for Trading

Commands:
    python -m app.cli.run_executor paper     # Run paper trading
    python -m app.cli.run_executor live      # Run live trading (CAREFUL!)
    python -m app.cli.run_executor status    # Show current status
    python -m app.cli.run_executor test      # Test execution flow
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path


def load_env():
    """Load .env file."""
    env_file = Path(__file__).parent.parent.parent / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())


async def run_paper():
    """Run paper trading session."""
    from app.services.execution_service import ExecutionService
    from app.services.arb_pipeline import ArbPipeline

    print("=" * 60)
    print("PAPER TRADING SESSION")
    print("=" * 60)

    # Set paper mode
    os.environ["EXECUTION_MODE"] = "paper"

    service = ExecutionService.from_env()
    pipeline = ArbPipeline.from_env()

    async with service:
        print(f"Mode: {service.executor.mode}")
        print(f"Initial Balance: ${await service.executor.get_balance() / 100:.2f}")
        print()

        # Run detection cycle
        signals = pipeline.run_cycle()

        if not signals:
            print("No tradeable signals found")
            return

        print(f"Found {len(signals)} signals")

        # Filter to tradeable signals
        tradeable = [s for s in signals if s.should_trade]
        print(f"Tradeable: {len(tradeable)}")

        if not tradeable:
            print("No signals meet trading criteria")
            return

        # Execute
        results = await service.execute_batch(tradeable, max_concurrent=3)

        # Summary
        print()
        print("RESULTS:")
        print("-" * 40)
        for r in results:
            print(f"  {r.summary()}")

        print()
        print(f"Stats: {service.stats.to_dict()}")


async def run_live():
    """Run live trading session."""
    from app.services.execution_service import ExecutionService

    print("=" * 60)
    print("!!! LIVE TRADING SESSION !!!")
    print("=" * 60)
    print()
    print("WARNING: This will execute REAL trades with REAL money!")
    print()

    confirm = input("Type 'CONFIRM' to proceed: ")
    if confirm != "CONFIRM":
        print("Aborted.")
        return

    os.environ["EXECUTION_MODE"] = "live"

    # Similar to paper but with live executor
    print("Live trading not fully implemented - use with caution")


async def show_status():
    """Show current system status."""
    from app.execution.circuit_breaker import CircuitBreaker

    print("=" * 60)
    print("EXECUTION SYSTEM STATUS")
    print("=" * 60)
    print()

    # Circuit breaker status
    cb = CircuitBreaker.from_env()
    status = cb.get_status()

    print("Circuit Breaker:")
    print(f"  State: {status['state']}")
    print(f"  Daily P&L: ${status['daily_pnl_cents'] / 100:.2f}")
    print(f"  Daily Trades: {status['daily_trade_count']}")
    print(f"  Consecutive Losses: {status['consecutive_losses']}")
    print()

    # Check API connectivity
    print("API Connectivity:")
    try:
        from app.connectors.kalshi.client import KalshiClient
        client = KalshiClient.from_env()
        balance = client.get_balance()
        print(f"  Kalshi REST: OK (Balance: ${balance.balance_dollars:.2f})")
        client.close()
    except Exception as e:
        print(f"  Kalshi REST: FAILED ({e})")

    # Check environment
    print()
    print("Environment:")
    print(f"  EXECUTION_MODE: {os.environ.get('EXECUTION_MODE', 'paper')}")
    print(f"  PAPER_TRADING: {os.environ.get('PAPER_TRADING', 'true')}")
    print(f"  MAX_POSITION_SIZE: {os.environ.get('MAX_POSITION_SIZE', '100')}")
    print(f"  MAX_DAILY_LOSS: {os.environ.get('MAX_DAILY_LOSS', '500')}")


async def run_test():
    """Test execution flow with mock data."""
    from app.execution.paper_executor import PaperExecutor
    from app.execution.models import ExecutionOrder

    print("=" * 60)
    print("EXECUTION FLOW TEST")
    print("=" * 60)
    print()

    executor = PaperExecutor(initial_balance=10000_00, seed=42)

    async with executor:
        # Test order
        order = ExecutionOrder(
            ticker="KXTEST-DEMO-BUF",
            side="yes",
            action="buy",
            limit_price=48,
            contracts=25,
        )

        print(f"Test Order: BUY 25 YES @ 48c")
        print(f"Max Cost: ${25 * 48 / 100:.2f}")
        print()

        result = await executor.execute(order)

        print(f"Result: {result.summary()}")
        print()
        print(f"Stats: {executor.get_stats()}")
        print()

        if result.success:
            print("TEST PASSED")
        else:
            print("TEST COMPLETED (order may not have filled due to simulation)")


def main():
    parser = argparse.ArgumentParser(description="Kalshi Execution CLI")
    parser.add_argument(
        "command",
        choices=["paper", "live", "status", "test"],
        help="Command to run",
    )

    args = parser.parse_args()

    load_env()

    if args.command == "paper":
        asyncio.run(run_paper())
    elif args.command == "live":
        asyncio.run(run_live())
    elif args.command == "status":
        asyncio.run(show_status())
    elif args.command == "test":
        asyncio.run(run_test())


if __name__ == "__main__":
    main()
```

---

## Phase 7: Integration Tests

**File:** `tests/test_execution_integration.py`
**Effort:** 8 hours
**Dependencies:** All previous phases

### Test Cases

```python
# tests/test_execution_integration.py

"""
Integration tests for the execution layer.

Run with: pytest tests/test_execution_integration.py -v
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

from app.execution.paper_executor import PaperExecutor
from app.execution.models import ExecutionOrder, ExecutionResult, OrderState
from app.execution.circuit_breaker import CircuitBreaker
from app.arb.detector import Signal, KalshiMarket, SportsbookConsensus, Edge, Confidence


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def paper_executor():
    """Create paper executor with fixed seed."""
    return PaperExecutor(initial_balance=10000_00, seed=42)


@pytest.fixture
def mock_signal():
    """Create mock detection signal."""
    kalshi = KalshiMarket(
        ticker="KXNFL-TEST-BUF",
        yes_bid=45,
        yes_ask=47,
        no_bid=53,
        no_ask=55,
        volume=1000,
    )

    consensus = SportsbookConsensus(
        implied_prob=52.0,
        book_count=4,
        spread=2.0,
        prices={"dk": 52, "fd": 51, "mgm": 53, "czr": 52},
    )

    edge = Edge(
        yes_edge=5,
        no_edge=2,
        best_edge=5,
        best_side="yes",
    )

    confidence = Confidence(
        score=75,
        tier="HIGH",
        components={"edge": 25, "consensus": 25, "liquidity": 15, "timing": 10},
    )

    return Signal(
        signal_id="test-signal-001",
        kalshi=kalshi,
        sportsbook=consensus,
        edge=edge,
        confidence=confidence,
        confidence_score=75,
        should_trade=True,
        action="BUY_YES",
        reason="5c edge, high confidence",
        recommended_contracts=25,
    )


# =============================================================================
# PAPER EXECUTOR TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_paper_executor_buy_yes(paper_executor):
    """Test buying YES contracts in paper mode."""
    await paper_executor.connect()

    order = ExecutionOrder(
        ticker="KXTEST-BUF",
        side="yes",
        action="buy",
        limit_price=48,
        contracts=10,
    )

    result = await paper_executor.execute(order)

    assert result.order.state in [OrderState.FILLED, OrderState.EXPIRED]

    if result.success:
        assert result.order.filled_contracts == 10
        assert result.total_cost_cents > 0

    await paper_executor.disconnect()


@pytest.mark.asyncio
async def test_paper_executor_insufficient_funds(paper_executor):
    """Test rejection when balance insufficient."""
    await paper_executor.connect()

    # Try to buy more than balance allows
    order = ExecutionOrder(
        ticker="KXTEST-BUF",
        side="yes",
        action="buy",
        limit_price=90,
        contracts=200,  # 200 * 90 = 18000c > 10000c balance
    )

    result = await paper_executor.execute(order)

    assert result.order.state == OrderState.REJECTED
    assert "INSUFFICIENT_FUNDS" in result.order.error_code

    await paper_executor.disconnect()


@pytest.mark.asyncio
async def test_paper_executor_position_tracking(paper_executor):
    """Test position updates after fills."""
    await paper_executor.connect()

    # Execute buy
    order = ExecutionOrder(
        ticker="KXTEST-BUF",
        side="yes",
        action="buy",
        limit_price=50,
        contracts=10,
    )

    result = await paper_executor.execute(order)

    if result.success:
        position = await paper_executor.get_position("KXTEST-BUF")
        assert position["yes_contracts"] == 10

    await paper_executor.disconnect()


@pytest.mark.asyncio
async def test_paper_executor_settlement():
    """Test position settlement on market resolution."""
    executor = PaperExecutor(initial_balance=10000_00, seed=42)
    await executor.connect()

    # Buy YES at 40c
    order = ExecutionOrder(
        ticker="KXTEST-BUF",
        side="yes",
        action="buy",
        limit_price=40,
        contracts=10,
    )

    result = await executor.execute(order)

    if result.success:
        initial_balance = await executor.get_balance()

        # Settle as YES wins
        pnl = executor.settle_position("KXTEST-BUF", "yes")

        # Should profit: 10 contracts * (100 - 40) = 600c
        final_balance = await executor.get_balance()

        # Note: slippage affects actual fill price
        assert final_balance > initial_balance or pnl > 0

    await executor.disconnect()


# =============================================================================
# CIRCUIT BREAKER INTEGRATION
# =============================================================================

@pytest.mark.asyncio
async def test_circuit_breaker_blocks_trade():
    """Test circuit breaker blocks risky trades."""
    # Set up circuit breaker with low limits
    cb = CircuitBreaker()
    cb.limits.max_position_size = 5  # Max 5 contracts

    executor = PaperExecutor(
        initial_balance=10000_00,
        circuit_breaker=cb,
        seed=42,
    )

    await executor.connect()

    # Try to buy more than limit
    order = ExecutionOrder(
        ticker="KXTEST-BUF",
        side="yes",
        action="buy",
        limit_price=50,
        contracts=10,  # Exceeds limit of 5
    )

    result = await executor.execute(order)

    assert result.order.state == OrderState.REJECTED
    assert "CIRCUIT_BREAKER" in result.order.error_code

    await executor.disconnect()


# =============================================================================
# SIGNAL TO ORDER CONVERSION
# =============================================================================

def test_order_from_signal(mock_signal):
    """Test creating ExecutionOrder from Signal."""
    order = ExecutionOrder.from_signal(mock_signal)

    assert order.ticker == "KXNFL-TEST-BUF"
    assert order.side == "yes"
    assert order.action == "buy"
    assert order.contracts == 25
    assert order.limit_price == 48  # yes_ask (47) + 1 offset
    assert order.signal_edge_cents == 5
    assert order.signal_confidence == 75


# =============================================================================
# EXECUTION RESULT CALCULATIONS
# =============================================================================

def test_execution_result_metrics():
    """Test ExecutionResult metric calculations."""
    order = ExecutionOrder(
        ticker="KXTEST-BUF",
        side="yes",
        action="buy",
        limit_price=50,
        contracts=10,
        signal_edge_cents=5,
    )

    # Simulate fill
    fill = order.record_fill(contracts=10, price=50)

    result = ExecutionResult.from_order(order, fills=[fill])

    assert result.success
    assert result.total_cost_cents == 500  # 10 * 50
    assert result.potential_profit_cents == 500  # 10 * (100 - 50)
    assert result.expected_value_cents == 50  # 5% edge * 10 * 100


# =============================================================================
# FULL FLOW INTEGRATION
# =============================================================================

@pytest.mark.asyncio
async def test_full_detection_to_execution_flow(mock_signal):
    """Test complete flow from detection signal to execution result."""
    executor = PaperExecutor(initial_balance=10000_00, seed=42)

    async with executor:
        # Convert signal to order
        order = ExecutionOrder.from_signal(mock_signal)

        # Execute
        result = await executor.execute(order)

        # Verify result structure
        assert result.order.signal_id == "test-signal-001"
        assert result.order.ticker == "KXNFL-TEST-BUF"

        # Check stats
        stats = executor.get_stats()
        assert stats["total_trades"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

---

## File Structure After Implementation

```
app/
├── execution/
│   ├── __init__.py              # Export all execution components
│   ├── base.py                  # Phase 1: ExecutorBase interface
│   ├── models.py                # (EXISTS) Order/Fill/Result models
│   ├── circuit_breaker.py       # (EXISTS) Risk controls
│   ├── paper_executor.py        # Phase 2: Paper trading
│   └── kalshi_executor.py       # Phase 4: Live trading
│
├── connectors/
│   └── kalshi/
│       ├── __init__.py
│       ├── auth.py              # (EXISTS) RSA-PSS auth
│       ├── client.py            # (EXISTS) REST API client
│       └── websocket.py         # Phase 3: WebSocket client
│
├── services/
│   ├── arb_pipeline.py          # (EXISTS) Detection pipeline
│   ├── execution_service.py     # Phase 5: Orchestrator
│   └── ...
│
└── cli/
    ├── run_ingest.py            # (EXISTS) Data ingestion
    └── run_executor.py          # Phase 6: Execution CLI
```

---

## Environment Variables to Add

```bash
# Add to .env.example

# Execution Configuration
EXECUTION_MODE=paper                    # paper or live
PAPER_INITIAL_BALANCE=10000_00          # Starting balance for paper trading (cents)
PAPER_FILL_PROB=0.95                    # Paper fill probability
PAPER_FILL_DELAY_MS=50                  # Paper simulated latency
PAPER_SLIPPAGE_CENTS=1                  # Paper price slippage

# Timeouts
EXECUTION_TIMEOUT=30                    # Order timeout in seconds
EXECUTION_MAX_RETRIES=3                 # Max retry attempts

# Logging
LOG_EXECUTIONS=true                     # Log to QuestDB

# WebSocket (optional)
KALSHI_WS_URL=wss://api.elections.kalshi.com/trade-api/ws/v2
```

---

## run.bat Updates

Add these commands to `run.bat`:

```batch
:: Execution commands
if "%COMMAND%"=="execute" goto :execute
if "%COMMAND%"=="execute-paper" goto :execute-paper
if "%COMMAND%"=="execute-live" goto :execute-live
if "%COMMAND%"=="execute-test" goto :execute-test
if "%COMMAND%"=="execute-status" goto :execute-status

:execute
echo Running execution in %EXECUTION_MODE% mode...
python -m app.cli.run_executor %EXECUTION_MODE%
goto :eof

:execute-paper
echo Running paper trading...
python -m app.cli.run_executor paper
goto :eof

:execute-live
echo Running LIVE trading...
echo WARNING: This uses real money!
python -m app.cli.run_executor live
goto :eof

:execute-test
echo Running execution tests...
python -m app.cli.run_executor test
goto :eof

:execute-status
echo Checking execution status...
python -m app.cli.run_executor status
goto :eof
```

---

## Implementation Checklist

### Phase 1: Base Interface (4 hours)
- [ ] Create `app/execution/base.py`
- [ ] Define `ExecutorBase` abstract class
- [ ] Define `ExecutorConfig` dataclass
- [ ] Add unit tests

### Phase 2: Paper Executor (8 hours)
- [ ] Create `app/execution/paper_executor.py`
- [ ] Implement fill simulation
- [ ] Implement position tracking
- [ ] Implement balance management
- [ ] Implement settlement
- [ ] Add unit tests
- [ ] Test with mock signals

### Phase 3: WebSocket Client (12 hours)
- [ ] Create `app/connectors/kalshi/websocket.py`
- [ ] Implement connection management
- [ ] Implement authentication
- [ ] Implement subscription handling
- [ ] Implement fill message parsing
- [ ] Implement auto-reconnect
- [ ] Add unit tests
- [ ] Test with demo API

### Phase 4: Kalshi Executor (16 hours)
- [ ] Create `app/execution/kalshi_executor.py`
- [ ] Integrate REST client
- [ ] Integrate WebSocket client
- [ ] Implement order submission
- [ ] Implement fill waiting
- [ ] Implement timeout/cancel
- [ ] Add unit tests
- [ ] Test with demo API

### Phase 5: Execution Service (8 hours)
- [ ] Create `app/services/execution_service.py`
- [ ] Implement signal→order conversion
- [ ] Implement batch execution
- [ ] Implement QuestDB logging
- [ ] Add unit tests

### Phase 6: CLI Runner (4 hours)
- [ ] Create `app/cli/run_executor.py`
- [ ] Implement `paper` command
- [ ] Implement `live` command
- [ ] Implement `status` command
- [ ] Implement `test` command
- [ ] Update `run.bat`

### Phase 7: Integration Tests (8 hours)
- [ ] Create `tests/test_execution_integration.py`
- [ ] Test paper executor flow
- [ ] Test circuit breaker integration
- [ ] Test signal→order→result flow
- [ ] Test settlement calculations
- [ ] Achieve >80% coverage

---

## Risk Mitigation

### Before Going Live

1. **Paper Trading Period** - Run paper trading for at least 1 week
2. **Demo API Testing** - Test all flows against Kalshi demo environment
3. **Small Position Limits** - Start with 1-5 contract limits
4. **Circuit Breaker Tuning** - Set conservative loss limits
5. **Manual Override** - Test kill switch functionality
6. **Monitoring** - Set up alerts for anomalies

### Emergency Procedures

1. **Kill Switch** - `CircuitBreaker.halt("reason")`
2. **Cancel All Orders** - `KalshiClient.cancel_all_orders()`
3. **Stop Service** - `ExecutionService.stop()`

---

## Sources

- [Kalshi WebSocket Documentation](https://docs.kalshi.com/getting_started/quick_start_websockets)
- [Kalshi API Developer Guide](https://zuplo.com/learning-center/kalshi-api)
- [Kalshi Help Center](https://help.kalshi.com/kalshi-api)

---

*Plan Created: 2026-01-08*
*Estimated Total Effort: 60 hours*
*Ready for Implementation: Yes*
