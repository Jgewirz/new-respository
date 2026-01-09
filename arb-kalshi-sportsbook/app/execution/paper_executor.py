"""
Paper Executor - Simulated Order Execution for Testing

High-fidelity paper trading executor that simulates Kalshi order execution
for strategy validation before live deployment.

Key Features:
    - Realistic fill simulation with configurable probability
    - Slippage modeling (adverse price movement)
    - Partial fill support
    - Price improvement simulation
    - Position tracking across multiple markets
    - Balance management with margin calculation
    - Settlement logic for position resolution

Integration:
    - Extends ExecutorBase from app.execution.base
    - Uses ExecutorConfig for simulation parameters
    - Uses CircuitBreaker for pre-trade risk checks
    - Updates ExecutionOrder/Fill/ExecutionResult from app.execution.models

Usage:
    from app.execution import PaperExecutor, ExecutorConfig

    # Create executor
    executor = PaperExecutor(initial_balance=10000_00)  # $100.00

    # Execute order
    async with executor:
        result = await executor.execute(order)
        print(result.summary())

    # Or use factory
    executor = create_paper_executor(initial_balance=10000_00)

Author: Claude Code
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any
import logging

from app.execution.base import ExecutorBase, ExecutorConfig
from app.execution.models import (
    ExecutionOrder,
    ExecutionResult,
    Fill,
    ExecutionMode,
    OrderState,
    now_ns,
    generate_id,
)
from app.execution.circuit_breaker import CircuitBreaker, TradeCheckResult

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# PAPER POSITION TRACKING
# =============================================================================

@dataclass
class PaperPosition:
    """
    Tracks position in a single market.

    Supports both YES and NO positions simultaneously
    (though this is typically avoided in practice).

    Attributes:
        ticker: Market ticker (e.g., "KXNFL-26JAN11-BUF")
        yes_contracts: Number of YES contracts held
        no_contracts: Number of NO contracts held
        avg_yes_price: Volume-weighted average YES entry price
        avg_no_price: Volume-weighted average NO entry price
        total_yes_cost: Total cost of YES position in cents
        total_no_cost: Total cost of NO position in cents
        created_at: When position was opened
        updated_at: Last update timestamp
    """
    ticker: str
    yes_contracts: int = 0
    no_contracts: int = 0
    avg_yes_price: float = 0.0
    avg_no_price: float = 0.0
    total_yes_cost: int = 0
    total_no_cost: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def has_position(self) -> bool:
        """Check if any position exists."""
        return self.yes_contracts > 0 or self.no_contracts > 0

    @property
    def net_contracts(self) -> int:
        """Net position (YES - NO)."""
        return self.yes_contracts - self.no_contracts

    @property
    def total_cost(self) -> int:
        """Total cost of all positions in cents."""
        return self.total_yes_cost + self.total_no_cost

    @property
    def max_profit_cents(self) -> int:
        """Maximum profit if position wins."""
        yes_profit = self.yes_contracts * 100 - self.total_yes_cost if self.yes_contracts > 0 else 0
        no_profit = self.no_contracts * 100 - self.total_no_cost if self.no_contracts > 0 else 0
        return max(yes_profit, no_profit)

    def add_contracts(
        self,
        side: str,
        contracts: int,
        price: int,
    ) -> None:
        """
        Add contracts to position.

        Args:
            side: "yes" or "no"
            contracts: Number of contracts to add
            price: Fill price in cents
        """
        self.updated_at = datetime.now(timezone.utc)
        cost = contracts * price

        if side == "yes":
            # Update average price
            if self.yes_contracts == 0:
                self.avg_yes_price = float(price)
            else:
                total_value = (self.avg_yes_price * self.yes_contracts) + (price * contracts)
                self.avg_yes_price = total_value / (self.yes_contracts + contracts)

            self.yes_contracts += contracts
            self.total_yes_cost += cost

        elif side == "no":
            if self.no_contracts == 0:
                self.avg_no_price = float(price)
            else:
                total_value = (self.avg_no_price * self.no_contracts) + (price * contracts)
                self.avg_no_price = total_value / (self.no_contracts + contracts)

            self.no_contracts += contracts
            self.total_no_cost += cost

    def remove_contracts(self, side: str, contracts: int) -> int:
        """
        Remove contracts from position.

        Args:
            side: "yes" or "no"
            contracts: Number of contracts to remove

        Returns:
            Actual contracts removed (may be less if position is smaller)
        """
        self.updated_at = datetime.now(timezone.utc)

        if side == "yes":
            actual = min(contracts, self.yes_contracts)
            if actual > 0 and self.yes_contracts > 0:
                cost_per_contract = self.total_yes_cost / self.yes_contracts
                self.total_yes_cost -= int(cost_per_contract * actual)
            self.yes_contracts -= actual
            if self.yes_contracts == 0:
                self.avg_yes_price = 0.0
                self.total_yes_cost = 0
            return actual

        elif side == "no":
            actual = min(contracts, self.no_contracts)
            if actual > 0 and self.no_contracts > 0:
                cost_per_contract = self.total_no_cost / self.no_contracts
                self.total_no_cost -= int(cost_per_contract * actual)
            self.no_contracts -= actual
            if self.no_contracts == 0:
                self.avg_no_price = 0.0
                self.total_no_cost = 0
            return actual

        return 0

    def settle(self, outcome: str) -> int:
        """
        Settle position based on market outcome.

        Args:
            outcome: "yes" or "no" (which side won)

        Returns:
            P&L in cents (positive = profit, negative = loss)
        """
        pnl = 0

        if outcome == "yes":
            # YES wins: YES contracts pay out, NO contracts lose
            if self.yes_contracts > 0:
                payout = self.yes_contracts * 100
                pnl += payout - self.total_yes_cost
            if self.no_contracts > 0:
                pnl -= self.total_no_cost  # Total loss

        elif outcome == "no":
            # NO wins: NO contracts pay out, YES contracts lose
            if self.no_contracts > 0:
                payout = self.no_contracts * 100
                pnl += payout - self.total_no_cost
            if self.yes_contracts > 0:
                pnl -= self.total_yes_cost  # Total loss

        # Clear position
        self.yes_contracts = 0
        self.no_contracts = 0
        self.avg_yes_price = 0.0
        self.avg_no_price = 0.0
        self.total_yes_cost = 0
        self.total_no_cost = 0

        return pnl

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "ticker": self.ticker,
            "yes_contracts": self.yes_contracts,
            "no_contracts": self.no_contracts,
            "avg_yes_price": round(self.avg_yes_price, 2),
            "avg_no_price": round(self.avg_no_price, 2),
            "total_yes_cost_cents": self.total_yes_cost,
            "total_no_cost_cents": self.total_no_cost,
            "total_cost_cents": self.total_cost,
            "net_contracts": self.net_contracts,
            "max_profit_cents": self.max_profit_cents,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


# =============================================================================
# PAPER EXECUTOR STATE
# =============================================================================

@dataclass
class PaperExecutorState:
    """
    Complete state of the paper executor.

    Tracks balance, positions, order history, and trading statistics.

    Attributes:
        balance_cents: Available cash balance in cents
        initial_balance_cents: Starting balance for ROI calculations
        positions: Dict of ticker -> PaperPosition
        orders: Dict of execution_id -> ExecutionOrder
        fills: List of all Fill events
        stats: Trading statistics
    """
    balance_cents: int
    initial_balance_cents: int
    positions: Dict[str, PaperPosition] = field(default_factory=dict)
    orders: Dict[str, ExecutionOrder] = field(default_factory=dict)
    fills: List[Fill] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=lambda: {
        "total_orders": 0,
        "filled_orders": 0,
        "partial_fills": 0,
        "canceled_orders": 0,
        "rejected_orders": 0,
        "expired_orders": 0,
        "total_contracts_traded": 0,
        "total_volume_cents": 0,
        "realized_pnl_cents": 0,
        "total_fills": 0,
        "avg_fill_time_ms": 0.0,
        "slippage_total_cents": 0,
        "price_improvements": 0,
    })
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def total_position_cost(self) -> int:
        """Total cost locked in positions."""
        return sum(p.total_cost for p in self.positions.values())

    @property
    def equity_cents(self) -> int:
        """Total equity (balance + position value)."""
        return self.balance_cents + self.total_position_cost

    @property
    def roi_pct(self) -> float:
        """Return on investment percentage."""
        if self.initial_balance_cents == 0:
            return 0.0
        return ((self.equity_cents - self.initial_balance_cents) / self.initial_balance_cents) * 100

    @property
    def fill_rate(self) -> float:
        """Order fill rate."""
        if self.stats["total_orders"] == 0:
            return 0.0
        return self.stats["filled_orders"] / self.stats["total_orders"]

    def get_or_create_position(self, ticker: str) -> PaperPosition:
        """Get existing position or create new one."""
        if ticker not in self.positions:
            self.positions[ticker] = PaperPosition(ticker=ticker)
        return self.positions[ticker]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "balance_cents": self.balance_cents,
            "balance_dollars": self.balance_cents / 100,
            "initial_balance_cents": self.initial_balance_cents,
            "equity_cents": self.equity_cents,
            "equity_dollars": self.equity_cents / 100,
            "roi_pct": round(self.roi_pct, 2),
            "total_position_cost_cents": self.total_position_cost,
            "positions_count": len(self.positions),
            "positions": {k: v.to_dict() for k, v in self.positions.items()},
            "orders_count": len(self.orders),
            "fills_count": len(self.fills),
            "fill_rate": round(self.fill_rate, 3),
            "stats": self.stats,
            "created_at": self.created_at.isoformat(),
        }


# =============================================================================
# PAPER EXECUTOR
# =============================================================================

class PaperExecutor(ExecutorBase):
    """
    Paper trading executor for strategy validation.

    Simulates Kalshi order execution with configurable:
    - Fill probability
    - Network latency
    - Price slippage
    - Partial fills
    - Price improvement

    Thread-safe via asyncio Lock.

    Example:
        # Create with default config
        executor = PaperExecutor(initial_balance=10000_00)

        # Or with custom config
        config = ExecutorConfig.paper_realistic()
        executor = PaperExecutor(initial_balance=10000_00, config=config)

        async with executor:
            result = await executor.execute(order)
            print(result.summary())
    """

    def __init__(
        self,
        initial_balance: int = 10000_00,  # $100.00 default
        config: Optional[ExecutorConfig] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize paper executor.

        Args:
            initial_balance: Starting balance in cents
            config: ExecutorConfig for simulation parameters
            circuit_breaker: Optional CircuitBreaker for risk checks
            seed: Random seed for reproducible simulations
        """
        self._config = config or ExecutorConfig.paper_default()
        self._circuit_breaker = circuit_breaker
        self._connected = False
        self._lock = asyncio.Lock()

        # Initialize state
        self._state = PaperExecutorState(
            balance_cents=initial_balance,
            initial_balance_cents=initial_balance,
        )

        # Random generator for simulation
        self._rng = random.Random(seed)

        # Order ID counter for paper orders
        self._order_counter = 0

        logger.info(
            f"PaperExecutor initialized: balance=${initial_balance/100:.2f}, "
            f"fill_prob={self._config.fill_probability}, "
            f"slippage={self._config.slippage_cents}c"
        )

    # =========================================================================
    # ABSTRACT PROPERTY IMPLEMENTATIONS
    # =========================================================================

    @property
    def mode(self) -> str:
        """Return executor mode."""
        return "paper"

    @property
    def is_connected(self) -> bool:
        """Check if executor is ready."""
        return self._connected

    @property
    def config(self) -> ExecutorConfig:
        """Get executor configuration."""
        return self._config

    @property
    def state(self) -> PaperExecutorState:
        """Get executor state."""
        return self._state

    # =========================================================================
    # LIFECYCLE METHODS
    # =========================================================================

    async def connect(self) -> None:
        """Connect paper executor (no-op for paper trading)."""
        async with self._lock:
            self._connected = True
            logger.debug("PaperExecutor connected")

    async def disconnect(self) -> None:
        """Disconnect paper executor."""
        async with self._lock:
            self._connected = False
            logger.debug("PaperExecutor disconnected")

    # =========================================================================
    # EXECUTION METHODS
    # =========================================================================

    async def execute(self, order: ExecutionOrder) -> ExecutionResult:
        """
        Execute an order with simulated fills.

        Simulation process:
        1. Pre-trade validation (balance, circuit breaker)
        2. Simulate network latency
        3. Determine fill outcome (fill, partial, reject)
        4. Apply slippage or price improvement
        5. Update positions and balance
        6. Return ExecutionResult

        Args:
            order: ExecutionOrder to execute

        Returns:
            ExecutionResult with outcome
        """
        async with self._lock:
            start_time = now_ns()
            fills: List[Fill] = []

            # Set execution mode
            order.mode = ExecutionMode.PAPER

            # Track order
            self._state.orders[order.execution_id] = order
            self._state.stats["total_orders"] += 1

            # Pre-trade validation
            validation_result = await self._validate_order(order)
            if not validation_result.allowed:
                order.mark_rejected(
                    code="VALIDATION_FAILED",
                    message=validation_result.reason,
                )
                self._state.stats["rejected_orders"] += 1
                await self.on_order_update(order)
                return ExecutionResult.from_order(order, fills=[])

            # Generate paper order ID
            self._order_counter += 1
            paper_order_id = f"paper_{self._order_counter:06d}"

            # Mark submitted
            order.mark_submitted(kalshi_order_id=paper_order_id)
            await self.on_order_update(order)

            # Simulate network latency
            if self._config.fill_delay_ms > 0:
                delay_sec = self._config.fill_delay_ms / 1000.0
                # Add some variance
                delay_sec *= (0.8 + self._rng.random() * 0.4)
                await asyncio.sleep(delay_sec)

            # Determine fill outcome
            fill_roll = self._rng.random()

            if fill_roll > self._config.fill_probability:
                # No fill - order expires
                order.mark_expired()
                self._state.stats["expired_orders"] += 1
                await self.on_order_update(order)
                return ExecutionResult.from_order(order, fills=[])

            # Order will fill - determine price and quantity
            fill_price = self._simulate_fill_price(order)
            fill_contracts = self._simulate_fill_quantity(order)

            if fill_contracts <= 0:
                order.mark_expired()
                self._state.stats["expired_orders"] += 1
                await self.on_order_update(order)
                return ExecutionResult.from_order(order, fills=[])

            # Calculate cost
            fill_cost = fill_contracts * fill_price

            # Check balance
            if fill_cost > self._state.balance_cents:
                order.mark_rejected(
                    code="INSUFFICIENT_BALANCE",
                    message=f"Need ${fill_cost/100:.2f}, have ${self._state.balance_cents/100:.2f}",
                )
                self._state.stats["rejected_orders"] += 1
                await self.on_order_update(order)
                return ExecutionResult.from_order(order, fills=[])

            # Execute fill
            fill = order.record_fill(
                contracts=fill_contracts,
                price=fill_price,
                is_maker=self._rng.random() < 0.3,  # 30% maker fills
            )
            fills.append(fill)
            self._state.fills.append(fill)

            # Update balance
            self._state.balance_cents -= fill_cost

            # Update position
            position = self._state.get_or_create_position(order.ticker)
            position.add_contracts(order.side, fill_contracts, fill_price)

            # Update stats
            self._state.stats["total_contracts_traded"] += fill_contracts
            self._state.stats["total_volume_cents"] += fill_cost
            self._state.stats["total_fills"] += 1

            # Track slippage
            expected_price = order.limit_price
            slippage = fill_price - expected_price
            if slippage > 0:
                self._state.stats["slippage_total_cents"] += slippage * fill_contracts
            elif slippage < 0:
                self._state.stats["price_improvements"] += 1

            # Track fill status
            if order.state == OrderState.FILLED:
                self._state.stats["filled_orders"] += 1
            elif order.state == OrderState.PARTIALLY_FILLED:
                self._state.stats["partial_fills"] += 1

            # Update average fill time
            fill_time_ms = (now_ns() - start_time) / 1e6
            n = self._state.stats["total_fills"]
            old_avg = self._state.stats["avg_fill_time_ms"]
            self._state.stats["avg_fill_time_ms"] = old_avg + (fill_time_ms - old_avg) / n

            # Trigger hooks
            await self.on_fill(order, fill)
            await self.on_order_update(order)

            # Handle partial fill - simulate remaining fill
            if order.state == OrderState.PARTIALLY_FILLED:
                # Second fill attempt for remaining contracts
                remaining_fill = await self._attempt_remaining_fill(order, fills)
                if remaining_fill:
                    fills.append(remaining_fill)

            return ExecutionResult.from_order(order, fills=fills)

    async def cancel(self, order: ExecutionOrder) -> ExecutionResult:
        """
        Cancel an open order.

        Args:
            order: ExecutionOrder to cancel

        Returns:
            ExecutionResult with final state
        """
        async with self._lock:
            if order.is_terminal:
                logger.warning(f"Cannot cancel terminal order: {order.execution_id}")
                return ExecutionResult.from_order(order, fills=[])

            # Get any fills that occurred
            fills = [f for f in self._state.fills if f.execution_id == order.execution_id]

            order.mark_canceled(reason="User requested cancellation")
            self._state.stats["canceled_orders"] += 1

            await self.on_order_update(order)

            return ExecutionResult.from_order(order, fills=fills)

    # =========================================================================
    # QUERY METHODS
    # =========================================================================

    async def get_position(self, ticker: str) -> dict:
        """
        Get position for a ticker.

        Args:
            ticker: Market ticker

        Returns:
            Position dictionary
        """
        async with self._lock:
            if ticker in self._state.positions:
                return self._state.positions[ticker].to_dict()

            return {
                "ticker": ticker,
                "yes_contracts": 0,
                "no_contracts": 0,
                "avg_yes_price": 0.0,
                "avg_no_price": 0.0,
            }

    async def get_balance(self) -> int:
        """
        Get available balance in cents.

        Returns:
            Balance in cents
        """
        async with self._lock:
            return self._state.balance_cents

    async def get_all_positions(self) -> Dict[str, dict]:
        """
        Get all positions.

        Returns:
            Dict of ticker -> position dict
        """
        async with self._lock:
            return {k: v.to_dict() for k, v in self._state.positions.items()}

    # =========================================================================
    # SETTLEMENT METHODS
    # =========================================================================

    async def settle_position(self, ticker: str, outcome: str) -> int:
        """
        Settle a position based on market outcome.

        Args:
            ticker: Market ticker
            outcome: "yes" or "no" (which side won)

        Returns:
            P&L in cents (positive = profit)
        """
        async with self._lock:
            if ticker not in self._state.positions:
                logger.warning(f"No position to settle: {ticker}")
                return 0

            position = self._state.positions[ticker]
            pnl = position.settle(outcome)

            # Update balance with payout
            if pnl > 0:
                # Profit: get back cost + profit
                self._state.balance_cents += position.total_cost + pnl
            else:
                # Loss: already paid cost, no return
                pass

            # Actually for settlement: balance gets payout
            # If YES wins and we have YES: payout = contracts * 100
            # We already deducted cost when buying, so add back payout
            if outcome == "yes" and position.yes_contracts > 0:
                self._state.balance_cents += position.yes_contracts * 100
            elif outcome == "no" and position.no_contracts > 0:
                self._state.balance_cents += position.no_contracts * 100

            # Update stats
            self._state.stats["realized_pnl_cents"] += pnl

            # Record with circuit breaker if present
            if self._circuit_breaker:
                self._circuit_breaker.record_settlement(pnl)

            # Remove empty position
            if not position.has_position:
                del self._state.positions[ticker]

            logger.info(f"Settled {ticker} ({outcome}): P&L ${pnl/100:.2f}")

            return pnl

    async def settle_all(self, outcomes: Dict[str, str]) -> int:
        """
        Settle multiple positions.

        Args:
            outcomes: Dict of ticker -> outcome ("yes" or "no")

        Returns:
            Total P&L in cents
        """
        total_pnl = 0
        for ticker, outcome in outcomes.items():
            pnl = await self.settle_position(ticker, outcome)
            total_pnl += pnl
        return total_pnl

    # =========================================================================
    # STATS AND MANAGEMENT
    # =========================================================================

    def get_stats(self) -> dict:
        """
        Get trading statistics.

        Returns:
            Statistics dictionary
        """
        return {
            **self._state.stats,
            "balance_cents": self._state.balance_cents,
            "balance_dollars": self._state.balance_cents / 100,
            "equity_cents": self._state.equity_cents,
            "equity_dollars": self._state.equity_cents / 100,
            "roi_pct": round(self._state.roi_pct, 2),
            "fill_rate": round(self._state.fill_rate, 3),
            "positions_count": len(self._state.positions),
        }

    def get_state(self) -> dict:
        """
        Get complete executor state.

        Returns:
            State dictionary
        """
        return self._state.to_dict()

    async def reset(self, initial_balance: Optional[int] = None) -> None:
        """
        Reset paper trading state.

        Args:
            initial_balance: New starting balance (uses original if not provided)
        """
        async with self._lock:
            balance = initial_balance or self._state.initial_balance_cents

            self._state = PaperExecutorState(
                balance_cents=balance,
                initial_balance_cents=balance,
            )
            self._order_counter = 0

            logger.info(f"PaperExecutor reset: balance=${balance/100:.2f}")

    # =========================================================================
    # SIMULATION HELPERS
    # =========================================================================

    async def _validate_order(self, order: ExecutionOrder) -> TradeCheckResult:
        """Validate order before execution."""
        # Check balance
        max_cost = order.contracts * order.limit_price
        if max_cost > self._state.balance_cents:
            return TradeCheckResult(
                allowed=False,
                reason=f"Insufficient balance: need ${max_cost/100:.2f}, have ${self._state.balance_cents/100:.2f}",
            )

        # Check circuit breaker
        if self._config.require_circuit_breaker and self._circuit_breaker:
            return self._circuit_breaker.check_trade(
                ticker=order.ticker,
                contracts=order.contracts,
                risk_cents=max_cost,
                balance_cents=self._state.balance_cents,
            )

        return TradeCheckResult(allowed=True, reason="Validation passed")

    def _simulate_fill_price(self, order: ExecutionOrder) -> int:
        """
        Simulate fill price with slippage or improvement.

        Returns:
            Fill price in cents
        """
        base_price = order.limit_price

        # Check for price improvement
        if self._rng.random() < self._config.price_improvement_probability:
            # Price improvement: fill better than limit
            improvement = self._rng.randint(1, 2)
            return max(1, base_price - improvement)

        # Apply slippage (adverse for buyer)
        if self._config.slippage_cents > 0:
            # Slippage varies from 0 to max
            slippage = self._rng.randint(0, self._config.slippage_cents)
            return min(99, base_price + slippage)

        return base_price

    def _simulate_fill_quantity(self, order: ExecutionOrder) -> int:
        """
        Simulate fill quantity (full or partial).

        Returns:
            Number of contracts to fill
        """
        remaining = order.remaining_contracts

        # Check for partial fill
        if self._rng.random() < self._config.partial_fill_probability:
            # Partial fill: 30-90% of remaining
            fill_pct = 0.3 + self._rng.random() * 0.6
            return max(1, int(remaining * fill_pct))

        # Full fill
        return remaining

    async def _attempt_remaining_fill(
        self,
        order: ExecutionOrder,
        existing_fills: List[Fill],
    ) -> Optional[Fill]:
        """
        Attempt to fill remaining contracts after partial fill.

        Args:
            order: Order with partial fill
            existing_fills: Fills already recorded

        Returns:
            Additional Fill if successful, None otherwise
        """
        if order.remaining_contracts <= 0:
            return None

        # 80% chance of filling remainder
        if self._rng.random() > 0.8:
            return None

        # Small delay
        await asyncio.sleep(self._config.fill_delay_ms / 2000.0)

        fill_price = self._simulate_fill_price(order)
        fill_contracts = order.remaining_contracts

        fill_cost = fill_contracts * fill_price
        if fill_cost > self._state.balance_cents:
            return None

        # Execute remaining fill
        fill = order.record_fill(
            contracts=fill_contracts,
            price=fill_price,
            is_maker=False,
        )

        # Update balance and position
        self._state.balance_cents -= fill_cost
        position = self._state.get_or_create_position(order.ticker)
        position.add_contracts(order.side, fill_contracts, fill_price)

        # Update stats
        self._state.stats["total_contracts_traded"] += fill_contracts
        self._state.stats["total_volume_cents"] += fill_cost
        self._state.stats["total_fills"] += 1
        self._state.fills.append(fill)

        if order.state == OrderState.FILLED:
            self._state.stats["filled_orders"] += 1
            # Decrement partial fills since it's now complete
            self._state.stats["partial_fills"] -= 1

        await self.on_fill(order, fill)

        return fill

    # =========================================================================
    # HOOKS
    # =========================================================================

    async def on_fill(self, order: ExecutionOrder, fill: Fill) -> None:
        """Log fill event."""
        logger.debug(
            f"Fill: {fill.contracts}@{fill.price}c for {order.ticker} "
            f"(order: {order.execution_id})"
        )

    async def on_order_update(self, order: ExecutionOrder) -> None:
        """Log order state change."""
        logger.debug(f"Order {order.execution_id}: {order.state.value}")

    async def on_error(self, order: ExecutionOrder, error: Exception) -> None:
        """Log error."""
        logger.error(f"Error executing {order.ticker}: {error}")


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_paper_executor(
    initial_balance: int = 10000_00,
    config: Optional[ExecutorConfig] = None,
    circuit_breaker: Optional[CircuitBreaker] = None,
    use_realistic_config: bool = False,
    seed: Optional[int] = None,
    **kwargs,
) -> PaperExecutor:
    """
    Factory function to create a PaperExecutor.

    Args:
        initial_balance: Starting balance in cents (default $100.00)
        config: Custom ExecutorConfig (uses default if not provided)
        circuit_breaker: Optional CircuitBreaker for risk checks
        use_realistic_config: Use realistic settings instead of defaults
        seed: Random seed for reproducible simulations
        **kwargs: Additional config overrides

    Returns:
        Configured PaperExecutor

    Example:
        # Basic usage
        executor = create_paper_executor(initial_balance=50000_00)  # $500

        # With realistic settings
        executor = create_paper_executor(
            initial_balance=100000_00,  # $1000
            use_realistic_config=True,
        )

        # With custom settings
        executor = create_paper_executor(
            initial_balance=10000_00,
            config=ExecutorConfig(
                fill_probability=0.90,
                slippage_cents=2,
            ),
        )
    """
    # Determine config
    if config is None:
        if use_realistic_config:
            config = ExecutorConfig.paper_realistic()
        else:
            config = ExecutorConfig.paper_default()

    # Apply any overrides from kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Create circuit breaker if needed
    if circuit_breaker is None and config.require_circuit_breaker:
        circuit_breaker = CircuitBreaker.from_env()

    return PaperExecutor(
        initial_balance=initial_balance,
        config=config,
        circuit_breaker=circuit_breaker,
        seed=seed,
    )


# =============================================================================
# CLI TEST FUNCTION
# =============================================================================

async def test_paper_executor():
    """
    Test the paper executor with simulated trades.

    Demonstrates:
    - Order execution
    - Position tracking
    - Settlement
    - Statistics
    """
    print("=" * 70)
    print("PAPER EXECUTOR TEST")
    print("=" * 70)
    print()

    # Create executor with reproducible seed
    executor = create_paper_executor(
        initial_balance=10000_00,  # $100.00
        seed=42,  # For reproducibility
    )

    async with executor:
        print(f"Initial State:")
        print(f"  Balance: ${executor.state.balance_cents / 100:.2f}")
        print(f"  Mode: {executor.mode}")
        print(f"  Connected: {executor.is_connected}")
        print()

        # Create test orders
        orders = [
            ExecutionOrder(
                ticker="KXNFL-TEST-001",
                side="yes",
                action="buy",
                limit_price=45,
                contracts=10,
            ),
            ExecutionOrder(
                ticker="KXNFL-TEST-002",
                side="no",
                action="buy",
                limit_price=52,
                contracts=5,
            ),
            ExecutionOrder(
                ticker="KXNFL-TEST-001",  # Same market
                side="yes",
                action="buy",
                limit_price=47,
                contracts=5,
            ),
        ]

        # Execute orders
        print("Executing Orders:")
        print("-" * 50)

        for i, order in enumerate(orders, 1):
            result = await executor.execute(order)
            status = "FILLED" if result.success else "PARTIAL" if result.partial else "FAILED"
            print(
                f"  Order {i}: {order.ticker} {order.side.upper()} "
                f"{order.contracts}@{order.limit_price}c -> {status}"
            )
            if result.success or result.partial:
                print(f"           Filled: {order.filled_contracts}@{order.average_fill_price:.1f}c")
                print(f"           Cost: ${result.total_cost_cents / 100:.2f}")

        print()

        # Show positions
        print("Positions:")
        print("-" * 50)
        positions = await executor.get_all_positions()
        for ticker, pos in positions.items():
            print(f"  {ticker}:")
            if pos["yes_contracts"] > 0:
                print(f"    YES: {pos['yes_contracts']} @ avg {pos['avg_yes_price']:.1f}c")
            if pos["no_contracts"] > 0:
                print(f"    NO: {pos['no_contracts']} @ avg {pos['avg_no_price']:.1f}c")
            print(f"    Total Cost: ${pos['total_cost_cents'] / 100:.2f}")

        print()

        # Show balance
        balance = await executor.get_balance()
        print(f"Balance After Trades: ${balance / 100:.2f}")
        print()

        # Settle positions
        print("Settling Positions:")
        print("-" * 50)

        # KXNFL-TEST-001 settles YES
        pnl1 = await executor.settle_position("KXNFL-TEST-001", "yes")
        print(f"  KXNFL-TEST-001 (YES wins): P&L ${pnl1 / 100:.2f}")

        # KXNFL-TEST-002 settles NO
        pnl2 = await executor.settle_position("KXNFL-TEST-002", "no")
        print(f"  KXNFL-TEST-002 (NO wins): P&L ${pnl2 / 100:.2f}")

        print()

        # Final stats
        print("Final Statistics:")
        print("-" * 50)
        stats = executor.get_stats()
        print(f"  Total Orders: {stats['total_orders']}")
        print(f"  Filled Orders: {stats['filled_orders']}")
        print(f"  Fill Rate: {stats['fill_rate'] * 100:.1f}%")
        print(f"  Contracts Traded: {stats['total_contracts_traded']}")
        print(f"  Volume: ${stats['total_volume_cents'] / 100:.2f}")
        print(f"  Realized P&L: ${stats['realized_pnl_cents'] / 100:.2f}")
        print(f"  Final Balance: ${stats['balance_dollars']:.2f}")
        print(f"  ROI: {stats['roi_pct']:.2f}%")

        print()
        print("=" * 70)
        print("TEST COMPLETE")
        print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_paper_executor())
