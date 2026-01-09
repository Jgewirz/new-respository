# Execution Layer Implementation Plan

## Executive Summary

This document outlines the step-by-step plan to build a **low-latency execution layer** for the Kalshi arbitrage system. The execution layer receives `Signal` objects from the `ArbPipeline` and manages the complete order lifecycle: validation → submission → monitoring → reconciliation.

**Target Latency**: Signal → Order Submitted < 50ms | Signal → Fill Confirmed < 200ms

---

## Current System State

### Implemented Components (Ready to Use)
```
app/connectors/kalshi/client.py     ✅ Full Kalshi API (sync + async)
app/connectors/kalshi/auth.py       ✅ RSA-PSS authentication
app/arb/detector.py                 ✅ Edge detection + Signal generation
app/services/arb_pipeline.py        ✅ Unified orchestration
app/mapping/resolver.py             ✅ Event mapping
app/data/questdb.py                 ✅ Time-series storage
```

### Data Flow (Current → Execution)
```
ArbPipeline.run_cycle()
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ Signal(                                                       │
│   should_trade=True,                                          │
│   action="BUY_YES",                                           │
│   kalshi=KalshiMarket(ticker="KXNFL-26JAN11-BUF", yes_ask=48),│
│   edge=EdgeCalculation(cents=7, percent=14.6),                │
│   confidence=ConfidenceScore(score=75, tier="HIGH"),          │
│   position=PositionSize(contracts=25, risk_amount=1200),      │
│ )                                                             │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│                    EXECUTION LAYER (TO BUILD)                 │
│                                                               │
│   Signal → Validate → Submit → Monitor → Reconcile → Log     │
└──────────────────────────────────────────────────────────────┘
```

---

## Files to Create

```
arb-kalshi-sportsbook/app/
├── execution/
│   ├── __init__.py                 # Module exports
│   ├── models.py                   # ExecutionOrder, Fill, ExecutionResult dataclasses
│   ├── circuit_breaker.py          # Risk controls + kill switch
│   ├── order_manager.py            # Order lifecycle management
│   ├── paper_executor.py           # Simulated execution for testing
│   ├── kalshi_executor.py          # Live execution against Kalshi API
│   ├── websocket_client.py         # Real-time fills via WebSocket
│   ├── reconciliation.py           # Position + P&L reconciliation
│   └── metrics.py                  # Execution metrics + latency tracking
│
├── cli/
│   └── run_executor.py             # Main execution loop CLI
│
└── services/
    └── execution_service.py        # High-level execution orchestration
```

---

## Phase 1: Core Models & Infrastructure

### File: `app/execution/models.py`

Defines the data structures for order lifecycle tracking.

```python
"""
Execution Layer Data Models

Dataclasses for tracking orders from signal through settlement.
All timestamps in nanoseconds for QuestDB compatibility.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
import uuid


class ExecutionMode(str, Enum):
    """Execution environment."""
    PAPER = "paper"       # Simulated fills
    LIVE = "live"         # Real Kalshi orders


class OrderState(str, Enum):
    """Order lifecycle states."""
    PENDING = "pending"           # Created, not yet submitted
    SUBMITTED = "submitted"       # Sent to Kalshi
    RESTING = "resting"           # On book, awaiting fill
    PARTIALLY_FILLED = "partial"  # Some contracts filled
    FILLED = "filled"             # Fully executed
    CANCELED = "canceled"         # Canceled by us
    REJECTED = "rejected"         # Rejected by Kalshi
    EXPIRED = "expired"           # TTL expired


@dataclass
class ExecutionOrder:
    """
    Complete order record with full lifecycle tracking.

    Created from Signal, tracks through submission and fill.
    """
    # Identifiers
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    client_order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    kalshi_order_id: Optional[str] = None

    # Signal origin
    signal_id: str = ""
    signal_edge_cents: int = 0
    signal_confidence: int = 0

    # Order details
    ticker: str = ""
    side: str = ""           # "yes" or "no"
    action: str = ""         # "buy" or "sell"
    order_type: str = "limit"

    # Pricing
    limit_price: int = 0     # Price in cents (1-99)
    contracts: int = 0       # Number of contracts

    # State
    state: OrderState = OrderState.PENDING
    mode: ExecutionMode = ExecutionMode.PAPER

    # Fill tracking
    filled_contracts: int = 0
    remaining_contracts: int = 0
    average_fill_price: float = 0.0

    # Timing (nanoseconds for QuestDB)
    created_at_ns: int = 0
    submitted_at_ns: int = 0
    first_fill_at_ns: int = 0
    completed_at_ns: int = 0

    # Error tracking
    error_code: Optional[str] = None
    error_message: Optional[str] = None

    # Retry tracking
    retry_count: int = 0
    max_retries: int = 3

    def __post_init__(self):
        if self.created_at_ns == 0:
            self.created_at_ns = int(datetime.now(timezone.utc).timestamp() * 1e9)
        self.remaining_contracts = self.contracts

    @property
    def is_terminal(self) -> bool:
        """Order is in a final state."""
        return self.state in (
            OrderState.FILLED,
            OrderState.CANCELED,
            OrderState.REJECTED,
            OrderState.EXPIRED,
        )

    @property
    def fill_rate(self) -> float:
        """Percentage of order filled."""
        if self.contracts == 0:
            return 0.0
        return self.filled_contracts / self.contracts

    @property
    def submission_latency_ms(self) -> float:
        """Time from creation to submission in milliseconds."""
        if self.submitted_at_ns == 0:
            return 0.0
        return (self.submitted_at_ns - self.created_at_ns) / 1e6

    @property
    def total_latency_ms(self) -> float:
        """Time from creation to completion in milliseconds."""
        if self.completed_at_ns == 0:
            return 0.0
        return (self.completed_at_ns - self.created_at_ns) / 1e6


@dataclass
class Fill:
    """Individual fill event."""
    fill_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    execution_id: str = ""
    kalshi_order_id: str = ""

    contracts: int = 0
    price: int = 0           # Fill price in cents

    timestamp_ns: int = 0
    is_maker: bool = False   # True if we provided liquidity

    @property
    def notional_cents(self) -> int:
        """Total value in cents."""
        return self.contracts * self.price


@dataclass
class ExecutionResult:
    """
    Final result of an execution attempt.

    Returned by executors after order completes or fails.
    """
    order: ExecutionOrder
    fills: list[Fill] = field(default_factory=list)

    # P&L calculation
    total_cost_cents: int = 0      # What we paid
    potential_profit_cents: int = 0 # If outcome is YES (100c - cost)

    # Execution quality
    slippage_cents: float = 0.0    # Difference from expected price

    @property
    def success(self) -> bool:
        """Order filled successfully."""
        return self.order.state == OrderState.FILLED

    @property
    def partial(self) -> bool:
        """Order partially filled."""
        return self.order.filled_contracts > 0 and not self.success
```

---

## Phase 2: Circuit Breaker (Risk Controls)

### File: `app/execution/circuit_breaker.py`

The circuit breaker is **critical safety infrastructure** that prevents runaway losses.

```python
"""
Circuit Breaker - Execution Risk Controls

Implements multiple layers of protection:
1. Per-trade risk limits
2. Daily loss limits
3. Position concentration limits
4. Rate limiting
5. Manual kill switch

The circuit breaker MUST be checked before every order submission.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Optional
import os
import asyncio


class BreakerState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"            # Halted - no trades
    HALF_OPEN = "half_open"  # Testing after cooldown


@dataclass
class RiskLimits:
    """Configurable risk parameters."""
    # Per-trade limits
    max_position_size: int = 100          # Max contracts per trade
    max_risk_per_trade: int = 500_00      # Max risk in cents ($5)

    # Daily limits
    max_daily_loss: int = 1000_00         # Daily loss limit in cents ($10)
    max_daily_trades: int = 50            # Max trades per day
    max_daily_volume: int = 5000          # Max contracts per day

    # Concentration limits
    max_position_per_event: int = 500     # Max contracts per sporting event
    max_exposure_per_sport: float = 0.20  # Max 20% of bankroll per sport

    # Rate limits
    max_orders_per_minute: int = 10       # Order submission rate
    min_time_between_orders_ms: int = 100 # Minimum gap between orders

    # Consecutive loss circuit breaker
    max_consecutive_losses: int = 5       # Halt after 5 consecutive losses

    @classmethod
    def from_env(cls) -> RiskLimits:
        """Load limits from environment variables."""
        return cls(
            max_position_size=int(os.environ.get("MAX_POSITION_SIZE", 100)),
            max_risk_per_trade=int(os.environ.get("MAX_RISK_PER_TRADE", 500_00)),
            max_daily_loss=int(os.environ.get("MAX_DAILY_LOSS", 1000_00)),
            max_daily_trades=int(os.environ.get("MAX_DAILY_TRADES", 50)),
        )


@dataclass
class DailyStats:
    """Daily trading statistics."""
    date: str = ""                    # YYYY-MM-DD
    trades_count: int = 0
    contracts_traded: int = 0
    realized_pnl_cents: int = 0
    consecutive_losses: int = 0
    last_trade_time: Optional[datetime] = None

    def __post_init__(self):
        if not self.date:
            self.date = datetime.now(timezone.utc).strftime("%Y-%m-%d")


@dataclass
class CircuitBreaker:
    """
    Multi-layered execution risk control system.

    Usage:
        breaker = CircuitBreaker.from_env()

        # Before every trade
        can_trade, reason = breaker.check_trade(signal)
        if not can_trade:
            log.warning(f"Trade blocked: {reason}")
            return

        # After every trade
        breaker.record_result(result)
    """
    limits: RiskLimits = field(default_factory=RiskLimits)
    state: BreakerState = BreakerState.CLOSED
    daily_stats: DailyStats = field(default_factory=DailyStats)

    # Position tracking
    positions_by_event: dict[str, int] = field(default_factory=dict)
    positions_by_sport: dict[str, int] = field(default_factory=dict)

    # Rate limiting
    _order_timestamps: list[datetime] = field(default_factory=list)
    _last_order_time: Optional[datetime] = None

    # Manual controls
    _manual_halt: bool = False
    _halt_reason: str = ""

    @classmethod
    def from_env(cls) -> CircuitBreaker:
        """Create circuit breaker from environment."""
        return cls(limits=RiskLimits.from_env())

    def check_trade(
        self,
        ticker: str,
        contracts: int,
        risk_cents: int,
        sport: str = "",
        event_id: str = "",
    ) -> tuple[bool, str]:
        """
        Check if trade is allowed.

        Returns:
            (allowed, reason) - reason is empty if allowed
        """
        # Check manual halt
        if self._manual_halt:
            return False, f"Manual halt: {self._halt_reason}"

        # Check circuit breaker state
        if self.state == BreakerState.OPEN:
            return False, "Circuit breaker OPEN - trading halted"

        # Reset daily stats if new day
        self._check_day_rollover()

        # Check per-trade limits
        if contracts > self.limits.max_position_size:
            return False, f"Position size {contracts} exceeds limit {self.limits.max_position_size}"

        if risk_cents > self.limits.max_risk_per_trade:
            return False, f"Risk ${risk_cents/100:.2f} exceeds limit ${self.limits.max_risk_per_trade/100:.2f}"

        # Check daily limits
        if self.daily_stats.trades_count >= self.limits.max_daily_trades:
            return False, f"Daily trade limit reached ({self.limits.max_daily_trades})"

        if self.daily_stats.contracts_traded + contracts > self.limits.max_daily_volume:
            return False, f"Daily volume limit would be exceeded"

        if -self.daily_stats.realized_pnl_cents >= self.limits.max_daily_loss:
            return False, f"Daily loss limit reached (${self.limits.max_daily_loss/100:.2f})"

        # Check consecutive losses
        if self.daily_stats.consecutive_losses >= self.limits.max_consecutive_losses:
            return False, f"Consecutive loss limit reached ({self.limits.max_consecutive_losses})"

        # Check rate limits
        if not self._check_rate_limit():
            return False, "Rate limit exceeded - slow down"

        # Check concentration limits
        if event_id:
            current = self.positions_by_event.get(event_id, 0)
            if current + contracts > self.limits.max_position_per_event:
                return False, f"Event position limit would be exceeded"

        return True, ""

    def record_result(self, result) -> None:
        """
        Record execution result and update state.

        Args:
            result: ExecutionResult from executor
        """
        self._check_day_rollover()

        if result.success or result.partial:
            self.daily_stats.trades_count += 1
            self.daily_stats.contracts_traded += result.order.filled_contracts
            self.daily_stats.last_trade_time = datetime.now(timezone.utc)

        # P&L tracking (simplified - full implementation in reconciliation)
        # Negative cost = profit, positive cost = loss for this tracking
        if result.total_cost_cents > 0:
            # We spent money - update realized when settled
            pass

    def record_settlement(self, pnl_cents: int) -> None:
        """Record P&L from settled position."""
        self.daily_stats.realized_pnl_cents += pnl_cents

        if pnl_cents < 0:
            self.daily_stats.consecutive_losses += 1

            # Check if we should trip the breaker
            if self.daily_stats.consecutive_losses >= self.limits.max_consecutive_losses:
                self._trip_breaker("Consecutive loss limit reached")
        else:
            self.daily_stats.consecutive_losses = 0

    def halt(self, reason: str) -> None:
        """Manual halt - stop all trading."""
        self._manual_halt = True
        self._halt_reason = reason
        self.state = BreakerState.OPEN

    def resume(self) -> None:
        """Resume trading after manual halt."""
        self._manual_halt = False
        self._halt_reason = ""
        self.state = BreakerState.CLOSED

    def _trip_breaker(self, reason: str) -> None:
        """Trip the circuit breaker."""
        self.state = BreakerState.OPEN
        self._halt_reason = reason

    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        now = datetime.now(timezone.utc)

        # Check minimum time between orders
        if self._last_order_time:
            elapsed_ms = (now - self._last_order_time).total_seconds() * 1000
            if elapsed_ms < self.limits.min_time_between_orders_ms:
                return False

        # Check orders per minute
        one_minute_ago = now - timedelta(minutes=1)
        self._order_timestamps = [
            ts for ts in self._order_timestamps
            if ts > one_minute_ago
        ]

        if len(self._order_timestamps) >= self.limits.max_orders_per_minute:
            return False

        # Update tracking
        self._order_timestamps.append(now)
        self._last_order_time = now

        return True

    def _check_day_rollover(self) -> None:
        """Reset stats if new trading day."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.daily_stats.date != today:
            self.daily_stats = DailyStats(date=today)
            # Don't clear manual halt on day rollover
            if not self._manual_halt:
                self.state = BreakerState.CLOSED
```

---

## Phase 3: Order Manager

### File: `app/execution/order_manager.py`

Manages order lifecycle with retry logic and state transitions.

```python
"""
Order Manager - Order Lifecycle Management

Handles:
1. Order creation from signals
2. State machine transitions
3. Retry logic with price adjustment
4. Timeout management
5. Order cancellation
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Callable, Awaitable
import asyncio

from app.arb.detector import Signal
from app.execution.models import ExecutionOrder, OrderState, ExecutionMode, Fill


@dataclass
class OrderConfig:
    """Order execution configuration."""
    # Limit order offset from signal price
    limit_offset_cents: int = 1       # Bid 1c above current ask for speed

    # Timeout settings
    fill_timeout_seconds: float = 30   # Max wait for fill
    partial_fill_wait: float = 5       # Extra wait if partial

    # Retry settings
    max_retries: int = 3
    retry_price_step: int = 1          # Improve price by 1c on retry
    max_price_improvement: int = 3     # Max 3c from original

    # Slippage protection
    max_slippage_cents: int = 2        # Abort if market moves > 2c


class OrderManager:
    """
    Manages order lifecycle from signal to completion.

    Usage:
        manager = OrderManager(config)
        order = manager.create_order(signal)

        # Submit and wait
        result = await manager.execute_order(order, submit_fn)
    """

    def __init__(self, config: Optional[OrderConfig] = None):
        self.config = config or OrderConfig()
        self._active_orders: dict[str, ExecutionOrder] = {}

    def create_order_from_signal(
        self,
        signal: Signal,
        mode: ExecutionMode = ExecutionMode.PAPER,
    ) -> ExecutionOrder:
        """
        Create ExecutionOrder from detection Signal.

        Applies limit offset and configures retry parameters.
        """
        # Determine side and price from signal
        if signal.action == "BUY_YES":
            side = "yes"
            action = "buy"
            # Buy at ask + offset for speed
            base_price = signal.kalshi.yes_ask
            limit_price = min(99, base_price + self.config.limit_offset_cents)
        elif signal.action == "BUY_NO":
            side = "no"
            action = "buy"
            base_price = signal.kalshi.no_ask
            limit_price = min(99, base_price + self.config.limit_offset_cents)
        else:
            raise ValueError(f"Unknown action: {signal.action}")

        order = ExecutionOrder(
            signal_id=signal.id if hasattr(signal, 'id') else "",
            signal_edge_cents=signal.edge.cents,
            signal_confidence=signal.confidence.score,
            ticker=signal.kalshi.ticker,
            side=side,
            action=action,
            limit_price=limit_price,
            contracts=signal.position.contracts,
            mode=mode,
            max_retries=self.config.max_retries,
        )

        self._active_orders[order.execution_id] = order
        return order

    def update_state(
        self,
        order: ExecutionOrder,
        new_state: OrderState,
        kalshi_order_id: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        """Update order state with timestamp."""
        now_ns = int(datetime.now(timezone.utc).timestamp() * 1e9)

        order.state = new_state

        if kalshi_order_id:
            order.kalshi_order_id = kalshi_order_id

        if new_state == OrderState.SUBMITTED:
            order.submitted_at_ns = now_ns
        elif new_state in (OrderState.FILLED, OrderState.CANCELED, OrderState.REJECTED):
            order.completed_at_ns = now_ns

        if error:
            order.error_message = error

    def record_fill(
        self,
        order: ExecutionOrder,
        contracts: int,
        price: int,
        is_maker: bool = False,
    ) -> Fill:
        """Record a fill event."""
        fill = Fill(
            execution_id=order.execution_id,
            kalshi_order_id=order.kalshi_order_id or "",
            contracts=contracts,
            price=price,
            timestamp_ns=int(datetime.now(timezone.utc).timestamp() * 1e9),
            is_maker=is_maker,
        )

        # Update order state
        if order.first_fill_at_ns == 0:
            order.first_fill_at_ns = fill.timestamp_ns

        order.filled_contracts += contracts
        order.remaining_contracts -= contracts

        # Update average price
        total_value = (order.average_fill_price * (order.filled_contracts - contracts) +
                       price * contracts)
        order.average_fill_price = total_value / order.filled_contracts

        # Update state
        if order.remaining_contracts <= 0:
            self.update_state(order, OrderState.FILLED)
        else:
            self.update_state(order, OrderState.PARTIALLY_FILLED)

        return fill

    def should_retry(self, order: ExecutionOrder) -> bool:
        """Check if order should be retried."""
        if order.retry_count >= order.max_retries:
            return False

        if order.state not in (OrderState.CANCELED, OrderState.EXPIRED):
            return False

        return True

    def prepare_retry(self, order: ExecutionOrder) -> ExecutionOrder:
        """
        Prepare order for retry with improved price.

        Returns new order with adjusted price.
        """
        order.retry_count += 1

        # Improve price up to max
        new_price = order.limit_price + self.config.retry_price_step
        max_price = order.limit_price + self.config.max_price_improvement

        order.limit_price = min(new_price, max_price, 99)
        order.state = OrderState.PENDING
        order.kalshi_order_id = None

        return order

    def check_slippage(
        self,
        order: ExecutionOrder,
        current_ask: int,
    ) -> tuple[bool, int]:
        """
        Check if market has moved too much.

        Returns:
            (ok, slippage_cents)
        """
        # Calculate slippage from original signal price
        expected_price = order.limit_price - self.config.limit_offset_cents
        slippage = current_ask - expected_price

        if slippage > self.config.max_slippage_cents:
            return False, slippage

        return True, slippage

    def get_active_orders(self) -> list[ExecutionOrder]:
        """Get all non-terminal orders."""
        return [
            order for order in self._active_orders.values()
            if not order.is_terminal
        ]

    def cleanup_completed(self) -> list[ExecutionOrder]:
        """Remove and return completed orders."""
        completed = [
            order for order in self._active_orders.values()
            if order.is_terminal
        ]

        for order in completed:
            del self._active_orders[order.execution_id]

        return completed
```

---

## Phase 4: Paper Executor

### File: `app/execution/paper_executor.py`

Simulated execution for testing without real money.

```python
"""
Paper Trading Executor

Simulates order execution for strategy validation:
- Realistic fill simulation based on market conditions
- Latency simulation
- Partial fill modeling
- Settlement simulation

Use paper trading to validate:
1. Edge detection accuracy
2. Position sizing appropriateness
3. Risk management effectiveness
4. System reliability
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
import asyncio
import random

from app.execution.models import (
    ExecutionOrder,
    ExecutionResult,
    Fill,
    OrderState,
    ExecutionMode,
)
from app.execution.order_manager import OrderManager


@dataclass
class PaperConfig:
    """Paper trading simulation parameters."""
    # Fill probability (higher = more fills)
    fill_probability: float = 0.85

    # Partial fill settings
    partial_fill_probability: float = 0.15
    min_partial_pct: float = 0.3
    max_partial_pct: float = 0.8

    # Latency simulation (milliseconds)
    min_latency_ms: int = 10
    max_latency_ms: int = 50

    # Slippage simulation
    slippage_probability: float = 0.1
    max_slippage_cents: int = 2

    # Price improvement probability (maker rebates)
    price_improvement_probability: float = 0.05


class PaperExecutor:
    """
    Simulated executor for paper trading.

    Usage:
        executor = PaperExecutor()
        result = await executor.execute(order)
    """

    def __init__(
        self,
        config: Optional[PaperConfig] = None,
        order_manager: Optional[OrderManager] = None,
    ):
        self.config = config or PaperConfig()
        self.order_manager = order_manager or OrderManager()

        # Track paper positions
        self.positions: dict[str, int] = {}   # ticker -> net contracts
        self.cash_balance: int = 1_000_000    # Start with $10,000 (in cents)

    async def execute(self, order: ExecutionOrder) -> ExecutionResult:
        """
        Execute order with simulated fills.

        Simulates:
        1. Network latency
        2. Fill probability based on order aggressiveness
        3. Partial fills
        4. Price slippage
        """
        order.mode = ExecutionMode.PAPER
        fills = []

        # Simulate submission latency
        latency = random.uniform(
            self.config.min_latency_ms,
            self.config.max_latency_ms
        )
        await asyncio.sleep(latency / 1000)

        # Update to submitted
        self.order_manager.update_state(
            order,
            OrderState.SUBMITTED,
            kalshi_order_id=f"paper_{order.client_order_id}",
        )

        # Simulate fill decision
        if random.random() > self.config.fill_probability:
            # No fill - order rests then expires
            await asyncio.sleep(0.1)
            self.order_manager.update_state(order, OrderState.EXPIRED)
            return ExecutionResult(order=order, fills=[])

        # Determine fill size
        if random.random() < self.config.partial_fill_probability:
            # Partial fill
            fill_pct = random.uniform(
                self.config.min_partial_pct,
                self.config.max_partial_pct
            )
            fill_contracts = max(1, int(order.contracts * fill_pct))
        else:
            # Full fill
            fill_contracts = order.contracts

        # Determine fill price
        fill_price = order.limit_price

        if random.random() < self.config.slippage_probability:
            # Adverse slippage
            slippage = random.randint(1, self.config.max_slippage_cents)
            fill_price = min(99, fill_price + slippage)
        elif random.random() < self.config.price_improvement_probability:
            # Price improvement
            fill_price = max(1, fill_price - 1)

        # Record fill
        fill = self.order_manager.record_fill(
            order,
            contracts=fill_contracts,
            price=fill_price,
            is_maker=random.random() < 0.3,
        )
        fills.append(fill)

        # Update paper positions
        self._update_position(order, fill)

        # Calculate result metrics
        total_cost = sum(f.notional_cents for f in fills)
        potential_profit = sum(f.contracts * (100 - f.price) for f in fills)
        slippage = fill_price - (order.limit_price - 1)  # vs expected

        return ExecutionResult(
            order=order,
            fills=fills,
            total_cost_cents=total_cost,
            potential_profit_cents=potential_profit,
            slippage_cents=slippage,
        )

    def _update_position(self, order: ExecutionOrder, fill: Fill) -> None:
        """Update paper position tracking."""
        ticker = order.ticker
        current = self.positions.get(ticker, 0)

        # Determine position delta
        if order.action == "buy":
            if order.side == "yes":
                delta = fill.contracts
            else:  # no
                delta = -fill.contracts
        else:  # sell
            if order.side == "yes":
                delta = -fill.contracts
            else:
                delta = fill.contracts

        self.positions[ticker] = current + delta

        # Update cash
        cost = fill.notional_cents
        if order.action == "buy":
            self.cash_balance -= cost
        else:
            self.cash_balance += cost

    def get_portfolio_value(self) -> dict:
        """Get current paper portfolio state."""
        return {
            "cash_balance_cents": self.cash_balance,
            "cash_balance_dollars": self.cash_balance / 100,
            "positions": self.positions.copy(),
            "position_count": len([p for p in self.positions.values() if p != 0]),
        }
```

---

## Phase 5: Live Kalshi Executor

### File: `app/execution/kalshi_executor.py`

Production executor using the Kalshi API.

```python
"""
Live Kalshi Executor

Production order execution against Kalshi API:
- Uses existing KalshiAsyncClient
- Implements order submission + monitoring
- Handles fills via polling (WebSocket in Phase 6)
- Includes emergency cancel functionality

CRITICAL: This executes REAL trades with REAL money.
Always test thoroughly with paper trading first.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
import asyncio

from app.connectors.kalshi.client import (
    KalshiAsyncClient,
    Order,
    OrderStatus,
    KalshiAPIError,
    InsufficientFundsError,
    MarketClosedError,
    OrderRejectedError,
)
from app.execution.models import (
    ExecutionOrder,
    ExecutionResult,
    Fill,
    OrderState,
    ExecutionMode,
)
from app.execution.order_manager import OrderManager, OrderConfig


@dataclass
class LiveExecutorConfig:
    """Live execution configuration."""
    # Polling settings (until WebSocket implemented)
    poll_interval_ms: int = 200      # Check order status every 200ms
    fill_timeout_seconds: float = 30  # Max wait for fill

    # Retry settings
    max_submit_retries: int = 2
    retry_delay_ms: int = 100

    # Safety settings
    require_balance_check: bool = True
    min_balance_cents: int = 100_00   # Require $1 minimum


class KalshiExecutor:
    """
    Live execution against Kalshi API.

    Usage:
        async with KalshiAsyncClient.from_env() as client:
            executor = KalshiExecutor(client)
            result = await executor.execute(order)
    """

    def __init__(
        self,
        client: KalshiAsyncClient,
        config: Optional[LiveExecutorConfig] = None,
        order_manager: Optional[OrderManager] = None,
    ):
        self.client = client
        self.config = config or LiveExecutorConfig()
        self.order_manager = order_manager or OrderManager()

    async def execute(self, order: ExecutionOrder) -> ExecutionResult:
        """
        Execute order against Kalshi.

        Flow:
        1. Pre-flight checks (balance, market status)
        2. Submit order via API
        3. Poll for fill status
        4. Handle partial fills, timeouts, errors
        """
        order.mode = ExecutionMode.LIVE
        fills = []

        try:
            # Pre-flight checks
            if self.config.require_balance_check:
                balance = await self.client.get_balance()
                if balance.balance < self.config.min_balance_cents:
                    self.order_manager.update_state(
                        order,
                        OrderState.REJECTED,
                        error="Insufficient balance",
                    )
                    return ExecutionResult(order=order)

            # Submit order
            kalshi_order = await self._submit_order(order)

            if kalshi_order is None:
                return ExecutionResult(order=order)

            self.order_manager.update_state(
                order,
                OrderState.SUBMITTED,
                kalshi_order_id=kalshi_order.order_id,
            )

            # Monitor for fills
            fills = await self._monitor_order(order, kalshi_order)

            # Calculate metrics
            total_cost = sum(f.notional_cents for f in fills)
            potential_profit = sum(f.contracts * (100 - f.price) for f in fills)

            return ExecutionResult(
                order=order,
                fills=fills,
                total_cost_cents=total_cost,
                potential_profit_cents=potential_profit,
            )

        except Exception as e:
            self.order_manager.update_state(
                order,
                OrderState.REJECTED,
                error=str(e),
            )
            return ExecutionResult(order=order)

    async def _submit_order(self, order: ExecutionOrder) -> Optional[Order]:
        """Submit order to Kalshi with retries."""
        for attempt in range(self.config.max_submit_retries):
            try:
                # Prepare order parameters
                kwargs = {
                    "ticker": order.ticker,
                    "side": order.side,
                    "action": order.action,
                    "count": order.contracts,
                    "type": order.order_type,
                    "client_order_id": order.client_order_id,
                }

                # Add price based on side
                if order.side == "yes":
                    kwargs["yes_price"] = order.limit_price
                else:
                    kwargs["no_price"] = order.limit_price

                kalshi_order = await self.client.create_order(**kwargs)
                return kalshi_order

            except InsufficientFundsError as e:
                self.order_manager.update_state(
                    order,
                    OrderState.REJECTED,
                    error="Insufficient funds",
                )
                return None

            except MarketClosedError as e:
                self.order_manager.update_state(
                    order,
                    OrderState.REJECTED,
                    error="Market closed",
                )
                return None

            except OrderRejectedError as e:
                self.order_manager.update_state(
                    order,
                    OrderState.REJECTED,
                    error=str(e),
                )
                return None

            except KalshiAPIError as e:
                if attempt < self.config.max_submit_retries - 1:
                    await asyncio.sleep(self.config.retry_delay_ms / 1000)
                    continue
                raise

        return None

    async def _monitor_order(
        self,
        order: ExecutionOrder,
        kalshi_order: Order,
    ) -> list[Fill]:
        """Monitor order until filled, canceled, or timeout."""
        fills = []
        start_time = datetime.now(timezone.utc)
        last_filled = 0

        while True:
            # Check timeout
            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
            if elapsed > self.config.fill_timeout_seconds:
                # Timeout - cancel remaining
                if order.remaining_contracts > 0:
                    await self._cancel_order(kalshi_order.order_id)
                break

            # Poll order status
            try:
                updated = await self.client.get_order(kalshi_order.order_id)
            except KalshiAPIError:
                await asyncio.sleep(self.config.poll_interval_ms / 1000)
                continue

            # Check for new fills
            if updated.filled_count > last_filled:
                new_fills = updated.filled_count - last_filled

                fill = self.order_manager.record_fill(
                    order,
                    contracts=new_fills,
                    price=updated.price_cents,  # Approximate
                )
                fills.append(fill)
                last_filled = updated.filled_count

            # Check terminal states
            if updated.is_filled:
                self.order_manager.update_state(order, OrderState.FILLED)
                break
            elif updated.is_canceled:
                self.order_manager.update_state(order, OrderState.CANCELED)
                break

            await asyncio.sleep(self.config.poll_interval_ms / 1000)

        return fills

    async def _cancel_order(self, order_id: str) -> bool:
        """Cancel order - returns True if successful."""
        try:
            await self.client.cancel_order(order_id)
            return True
        except KalshiAPIError:
            return False

    async def cancel_all_orders(self, ticker: Optional[str] = None) -> int:
        """Emergency cancel all open orders."""
        open_orders = await self.client.get_open_orders(ticker=ticker)
        canceled = 0

        for order in open_orders:
            if await self._cancel_order(order.order_id):
                canceled += 1

        return canceled
```

---

## Phase 6: WebSocket Client (Real-Time Fills)

### File: `app/execution/websocket_client.py`

WebSocket connection for real-time fill notifications (lower latency than polling).

```python
"""
Kalshi WebSocket Client

Real-time data streaming for:
1. Fill notifications (instant vs 200ms polling)
2. Market price updates
3. Position changes

Based on Kalshi WebSocket API:
- Heartbeat: Ping every 10s, respond with Pong
- Channels: fill, ticker, orderbook_delta
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Callable, Awaitable
import asyncio
import json

# Note: Requires websockets library
# pip install websockets


@dataclass
class WebSocketConfig:
    """WebSocket configuration."""
    url: str = "wss://trading-api.kalshi.com/trade-api/ws/v2"
    demo_url: str = "wss://demo-api.kalshi.co/trade-api/ws/v2"

    heartbeat_interval: float = 10.0
    reconnect_delay: float = 1.0
    max_reconnect_delay: float = 30.0

    use_demo: bool = False


FillCallback = Callable[[dict], Awaitable[None]]
TickerCallback = Callable[[dict], Awaitable[None]]


class KalshiWebSocket:
    """
    WebSocket client for real-time Kalshi data.

    Usage:
        ws = KalshiWebSocket(auth)

        async def on_fill(data):
            print(f"Fill: {data}")

        ws.on_fill = on_fill
        await ws.connect()
        await ws.subscribe_fills()
    """

    def __init__(
        self,
        auth,  # KalshiAuth instance
        config: Optional[WebSocketConfig] = None,
    ):
        self.auth = auth
        self.config = config or WebSocketConfig()

        self._ws = None
        self._connected = False
        self._subscriptions: set[str] = set()

        # Callbacks
        self.on_fill: Optional[FillCallback] = None
        self.on_ticker: Optional[TickerCallback] = None
        self.on_disconnect: Optional[Callable[[], Awaitable[None]]] = None

        # Background tasks
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._receive_task: Optional[asyncio.Task] = None

    async def connect(self) -> None:
        """Establish WebSocket connection."""
        import websockets

        url = self.config.demo_url if self.config.use_demo else self.config.url

        # Get auth headers
        headers = self.auth.get_headers("GET", "/trade-api/ws/v2")

        self._ws = await websockets.connect(
            url,
            extra_headers=headers,
        )
        self._connected = True

        # Start background tasks
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._receive_task = asyncio.create_task(self._receive_loop())

    async def disconnect(self) -> None:
        """Close WebSocket connection."""
        self._connected = False

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        if self._receive_task:
            self._receive_task.cancel()

        if self._ws:
            await self._ws.close()

    async def subscribe_fills(self) -> None:
        """Subscribe to fill notifications."""
        await self._subscribe("fill")

    async def subscribe_ticker(self, ticker: str) -> None:
        """Subscribe to market ticker updates."""
        await self._send({
            "type": "subscribe",
            "channel": "ticker",
            "params": {"ticker": ticker},
        })
        self._subscriptions.add(f"ticker:{ticker}")

    async def _subscribe(self, channel: str) -> None:
        """Send subscription message."""
        await self._send({
            "type": "subscribe",
            "channel": channel,
        })
        self._subscriptions.add(channel)

    async def _send(self, message: dict) -> None:
        """Send JSON message."""
        if self._ws and self._connected:
            await self._ws.send(json.dumps(message))

    async def _heartbeat_loop(self) -> None:
        """Send periodic pings to keep connection alive."""
        while self._connected:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)
                if self._ws:
                    await self._ws.ping(b"heartbeat")
            except Exception:
                break

    async def _receive_loop(self) -> None:
        """Process incoming messages."""
        while self._connected:
            try:
                message = await self._ws.recv()
                data = json.loads(message)

                await self._handle_message(data)

            except Exception as e:
                if self._connected:
                    # Connection lost - trigger reconnect
                    if self.on_disconnect:
                        await self.on_disconnect()
                break

    async def _handle_message(self, data: dict) -> None:
        """Route message to appropriate handler."""
        channel = data.get("channel", "")

        if channel == "fill" and self.on_fill:
            await self.on_fill(data)
        elif channel == "ticker" and self.on_ticker:
            await self.on_ticker(data)
```

---

## Phase 7: Execution Service (Orchestration)

### File: `app/services/execution_service.py`

High-level orchestration of the execution layer.

```python
"""
Execution Service - High-Level Orchestration

Ties together all execution components:
1. Circuit breaker checks
2. Order management
3. Paper vs live routing
4. Result logging to QuestDB
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
import os

from app.arb.detector import Signal
from app.execution.models import ExecutionOrder, ExecutionResult, ExecutionMode
from app.execution.circuit_breaker import CircuitBreaker
from app.execution.order_manager import OrderManager
from app.execution.paper_executor import PaperExecutor
from app.execution.kalshi_executor import KalshiExecutor
from app.connectors.kalshi.client import KalshiAsyncClient


class ExecutionService:
    """
    Unified execution service.

    Usage:
        service = ExecutionService.from_env()

        for signal in pipeline.run_cycle():
            if signal.should_trade:
                result = await service.execute_signal(signal)
    """

    def __init__(
        self,
        mode: ExecutionMode = ExecutionMode.PAPER,
        circuit_breaker: Optional[CircuitBreaker] = None,
        order_manager: Optional[OrderManager] = None,
        kalshi_client: Optional[KalshiAsyncClient] = None,
    ):
        self.mode = mode
        self.circuit_breaker = circuit_breaker or CircuitBreaker.from_env()
        self.order_manager = order_manager or OrderManager()

        # Initialize appropriate executor
        if mode == ExecutionMode.PAPER:
            self.executor = PaperExecutor(order_manager=self.order_manager)
        else:
            if kalshi_client is None:
                raise ValueError("Live mode requires KalshiAsyncClient")
            self.executor = KalshiExecutor(
                client=kalshi_client,
                order_manager=self.order_manager,
            )

    @classmethod
    def from_env(cls, kalshi_client: Optional[KalshiAsyncClient] = None):
        """Create service from environment configuration."""
        paper_mode = os.environ.get("PAPER_TRADING", "true").lower() == "true"
        mode = ExecutionMode.PAPER if paper_mode else ExecutionMode.LIVE

        return cls(mode=mode, kalshi_client=kalshi_client)

    async def execute_signal(self, signal: Signal) -> ExecutionResult:
        """
        Execute a trading signal.

        Flow:
        1. Check circuit breaker
        2. Create order from signal
        3. Execute via appropriate executor
        4. Record result
        5. Return result
        """
        # Extract risk parameters
        risk_cents = signal.position.contracts * signal.kalshi.yes_ask

        # Check circuit breaker
        can_trade, reason = self.circuit_breaker.check_trade(
            ticker=signal.kalshi.ticker,
            contracts=signal.position.contracts,
            risk_cents=risk_cents,
        )

        if not can_trade:
            # Create rejected order
            order = self.order_manager.create_order_from_signal(signal, self.mode)
            self.order_manager.update_state(
                order,
                ExecutionOrder.OrderState.REJECTED,
                error=f"Circuit breaker: {reason}",
            )
            return ExecutionResult(order=order)

        # Create and execute order
        order = self.order_manager.create_order_from_signal(signal, self.mode)
        result = await self.executor.execute(order)

        # Record result with circuit breaker
        self.circuit_breaker.record_result(result)

        return result

    async def emergency_halt(self, reason: str = "Manual halt") -> None:
        """Emergency stop - halt all trading and cancel orders."""
        self.circuit_breaker.halt(reason)

        if self.mode == ExecutionMode.LIVE:
            await self.executor.cancel_all_orders()
```

---

## Phase 8: Main Execution Loop CLI

### File: `app/cli/run_executor.py`

Main entry point for running the execution system.

```python
"""
Execution Loop CLI

Main entry point for the arbitrage execution system.

Usage:
    # Paper trading (default)
    python -m app.cli.run_executor

    # Live trading
    PAPER_TRADING=false python -m app.cli.run_executor

    # With specific profile
    TRADING_PROFILE=AGGRESSIVE python -m app.cli.run_executor
"""

from __future__ import annotations
import asyncio
import os
import signal
import sys
from datetime import datetime, timezone

# Load environment
from pathlib import Path
env_file = Path(__file__).parent.parent.parent / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())

from app.services.arb_pipeline import ArbPipeline
from app.services.execution_service import ExecutionService
from app.execution.models import ExecutionMode
from app.connectors.kalshi.client import KalshiAsyncClient


# Configuration
CYCLE_INTERVAL_MS = int(os.environ.get("CYCLE_INTERVAL_MS", 1000))
MAX_SIGNALS_PER_CYCLE = int(os.environ.get("MAX_SIGNALS_PER_CYCLE", 3))


async def main():
    """Main execution loop."""
    print("=" * 60)
    print("KALSHI ARBITRAGE EXECUTION ENGINE")
    print("=" * 60)
    print()

    # Determine mode
    paper_mode = os.environ.get("PAPER_TRADING", "true").lower() == "true"
    mode = ExecutionMode.PAPER if paper_mode else ExecutionMode.LIVE

    print(f"Mode: {'PAPER TRADING' if paper_mode else 'LIVE TRADING'}")
    print(f"Profile: {os.environ.get('TRADING_PROFILE', 'STANDARD')}")
    print(f"Cycle Interval: {CYCLE_INTERVAL_MS}ms")
    print()

    if not paper_mode:
        print("WARNING: LIVE TRADING MODE")
        print("Real money will be used!")
        response = input("Type 'CONFIRM' to continue: ")
        if response != "CONFIRM":
            print("Aborted.")
            return

    # Initialize components
    kalshi_client = None
    if not paper_mode:
        kalshi_client = KalshiAsyncClient.from_env()
        await kalshi_client.__aenter__()

    pipeline = ArbPipeline.from_env()
    execution_service = ExecutionService.from_env(kalshi_client=kalshi_client)

    # Setup graceful shutdown
    shutdown_event = asyncio.Event()

    def shutdown_handler(sig, frame):
        print("\nShutdown requested...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    # Main loop
    print("Starting execution loop...")
    print("-" * 60)

    cycle_count = 0
    total_signals = 0
    total_trades = 0

    try:
        while not shutdown_event.is_set():
            cycle_start = datetime.now(timezone.utc)
            cycle_count += 1

            # Run detection cycle
            try:
                signals = pipeline.run_cycle()
            except Exception as e:
                print(f"[{cycle_start}] Detection error: {e}")
                await asyncio.sleep(CYCLE_INTERVAL_MS / 1000)
                continue

            # Filter and sort signals
            tradeable = [s for s in signals if s.should_trade]
            tradeable.sort(key=lambda s: -s.edge.cents)

            total_signals += len(tradeable)

            # Execute top signals
            for signal in tradeable[:MAX_SIGNALS_PER_CYCLE]:
                result = await execution_service.execute_signal(signal)

                if result.success:
                    total_trades += 1
                    print(f"[{datetime.now()}] TRADE: {signal.action} "
                          f"{result.order.filled_contracts}x {signal.kalshi.ticker} "
                          f"@ {result.order.average_fill_price}c "
                          f"(edge: {signal.edge.cents}c)")
                elif result.order.error_message:
                    print(f"[{datetime.now()}] BLOCKED: {signal.kalshi.ticker} - "
                          f"{result.order.error_message}")

            # Print cycle summary periodically
            if cycle_count % 60 == 0:
                print(f"[Stats] Cycles: {cycle_count} | "
                      f"Signals: {total_signals} | "
                      f"Trades: {total_trades}")

            # Wait for next cycle
            elapsed = (datetime.now(timezone.utc) - cycle_start).total_seconds()
            sleep_time = max(0, (CYCLE_INTERVAL_MS / 1000) - elapsed)
            await asyncio.sleep(sleep_time)

    finally:
        # Cleanup
        print("\nShutting down...")

        if not paper_mode and kalshi_client:
            # Cancel any open orders
            canceled = await execution_service.executor.cancel_all_orders()
            print(f"Canceled {canceled} open orders")
            await kalshi_client.__aexit__(None, None, None)

        print(f"Final Stats: {cycle_count} cycles, {total_signals} signals, {total_trades} trades")
        print("Shutdown complete.")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Implementation Checklist

### Phase 1: Core Infrastructure (Day 1)
- [ ] Create `app/execution/__init__.py`
- [ ] Create `app/execution/models.py` - ExecutionOrder, Fill, ExecutionResult
- [ ] Unit tests for models

### Phase 2: Risk Management (Day 2)
- [ ] Create `app/execution/circuit_breaker.py`
- [ ] Implement all risk limit checks
- [ ] Unit tests for circuit breaker

### Phase 3: Order Management (Day 3)
- [ ] Create `app/execution/order_manager.py`
- [ ] Implement state machine transitions
- [ ] Implement retry logic
- [ ] Unit tests

### Phase 4: Paper Trading (Day 4)
- [ ] Create `app/execution/paper_executor.py`
- [ ] Test with simulated signals
- [ ] Validate P&L tracking

### Phase 5: Live Execution (Day 5)
- [ ] Create `app/execution/kalshi_executor.py`
- [ ] Test against Kalshi demo API
- [ ] Implement emergency cancel

### Phase 6: WebSocket (Day 6)
- [ ] Create `app/execution/websocket_client.py`
- [ ] Integrate with executor for real-time fills
- [ ] Test reconnection logic

### Phase 7: Integration (Day 7)
- [ ] Create `app/services/execution_service.py`
- [ ] Create `app/cli/run_executor.py`
- [ ] End-to-end integration tests

### Phase 8: Production Readiness (Day 8+)
- [ ] Add QuestDB logging for execution metrics
- [ ] Add alerting (email/Slack on errors)
- [ ] Load testing
- [ ] Paper trade for 7+ days
- [ ] Code review
- [ ] Deploy to production

---

## Latency Budget

| Component | Target | Method |
|-----------|--------|--------|
| Signal creation | 0ms | In-memory from pipeline |
| Circuit breaker check | <1ms | In-memory state |
| Order creation | <1ms | In-memory |
| Order submission | <50ms | HTTP to Kalshi |
| Fill detection (polling) | <200ms | HTTP polling |
| Fill detection (WebSocket) | <50ms | WebSocket push |
| **Total (polling)** | **<300ms** | Signal → Fill confirmed |
| **Total (WebSocket)** | **<150ms** | Signal → Fill confirmed |

---

## Testing Strategy

### Unit Tests
```bash
pytest tests/execution/test_models.py
pytest tests/execution/test_circuit_breaker.py
pytest tests/execution/test_order_manager.py
```

### Integration Tests
```bash
# Paper trading integration
pytest tests/execution/test_paper_executor.py

# Full pipeline integration (paper mode)
pytest tests/execution/test_full_pipeline.py
```

### Manual Testing Checklist
- [ ] Paper mode executes simulated fills
- [ ] Circuit breaker blocks over-limit trades
- [ ] Circuit breaker trips on consecutive losses
- [ ] Order retries work correctly
- [ ] Graceful shutdown cancels orders
- [ ] WebSocket reconnects on disconnect
- [ ] Live mode authenticates successfully
- [ ] Live mode submits and receives fills

---

## Risk Warnings

1. **NEVER deploy live trading without extensive paper testing**
2. **ALWAYS have circuit breaker enabled**
3. **ALWAYS test emergency halt procedure before going live**
4. **START with conservative risk limits**
5. **MONITOR actively during initial live trading**
6. **HAVE manual intervention capability ready**

---

## Dependencies

Add to `pyproject.toml`:
```toml
dependencies = [
    # ... existing deps
    "websockets>=12.0",  # For WebSocket client
]
```
