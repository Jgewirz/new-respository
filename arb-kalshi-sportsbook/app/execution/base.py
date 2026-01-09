"""
Base Executor Interface - Abstract Foundation for Order Execution

Defines the interface that all executors (Paper, Live, Demo) must implement.
This enables executor swapping without code changes and ensures consistent behavior.

Key Classes:
    - ExecutorBase: Abstract base class defining executor contract
    - ExecutorConfig: Configuration dataclass for executor settings

Integration Points:
    - Input: ExecutionOrder from app.execution.models
    - Input: Signal from app.arb.detector (via ExecutionOrder.from_signal)
    - Output: ExecutionResult with order outcome and fills
    - Risk: CircuitBreaker from app.execution.circuit_breaker

Data Flow:
    Signal → ExecutionOrder → Executor.execute() → ExecutionResult

Usage:
    from app.execution.base import ExecutorBase, ExecutorConfig

    # Create config from environment
    config = ExecutorConfig.from_env()

    # Executors implement this interface
    class MyExecutor(ExecutorBase):
        async def execute(self, order: ExecutionOrder) -> ExecutionResult:
            ...

Design Principles:
    1. All executors are interchangeable via this interface
    2. Async-first for non-blocking execution
    3. Context manager support for clean resource management
    4. Lifecycle hooks for customization (on_fill, on_order_update)
    5. Comprehensive error handling and state tracking

Author: Claude Code
Version: 1.0.0
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Any, TYPE_CHECKING
import os

# Type checking imports to avoid circular dependencies
if TYPE_CHECKING:
    from app.execution.models import ExecutionOrder, ExecutionResult, Fill


# =============================================================================
# EXECUTOR CONFIGURATION
# =============================================================================

@dataclass
class ExecutorConfig:
    """
    Configuration for order executors.

    Contains settings for both paper and live execution modes.
    Paper-specific settings are ignored by live executors.

    Attributes:
        mode: Execution mode ("paper" or "live")

        Paper Trading Settings:
            fill_probability: Probability of fill in paper mode (0.0-1.0)
            fill_delay_ms: Simulated network latency in milliseconds
            slippage_cents: Price slippage in cents (adverse direction)
            partial_fill_probability: Probability of partial fill (0.0-1.0)

        General Settings:
            timeout_seconds: Max time to wait for fill before canceling
            max_retries: Max retry attempts on transient failures
            retry_delay_ms: Delay between retry attempts

        Safety Settings:
            require_circuit_breaker: Require circuit breaker check before trades
            validate_balance: Check balance before submitting orders
            min_balance_cents: Minimum required balance to trade

    Environment Variables:
        EXECUTION_MODE: "paper" or "live"
        PAPER_FILL_PROB: Fill probability for paper trading
        PAPER_FILL_DELAY_MS: Simulated latency
        PAPER_SLIPPAGE_CENTS: Price slippage
        PAPER_PARTIAL_FILL_PROB: Partial fill probability
        EXECUTION_TIMEOUT: Order timeout in seconds
        EXECUTION_MAX_RETRIES: Max retry attempts
        EXECUTION_RETRY_DELAY_MS: Delay between retries
        REQUIRE_CIRCUIT_BREAKER: "true" to require risk checks
        VALIDATE_BALANCE: "true" to check balance
        MIN_BALANCE: Minimum balance in dollars
    """
    # Mode selection
    mode: str = "paper"  # "paper" or "live"

    # Paper trading simulation settings
    fill_probability: float = 0.95          # 95% fill rate default
    fill_delay_ms: float = 50.0             # 50ms simulated latency
    slippage_cents: int = 1                 # 1c adverse slippage
    partial_fill_probability: float = 0.10  # 10% partial fills

    # Timeout and retry settings
    timeout_seconds: float = 30.0           # 30s fill timeout
    max_retries: int = 3                    # 3 retry attempts
    retry_delay_ms: float = 100.0           # 100ms between retries

    # Safety settings
    require_circuit_breaker: bool = True    # Require risk check
    validate_balance: bool = True           # Check balance before trade
    min_balance_cents: int = 100_00         # $1.00 minimum balance

    # Price improvement settings (for paper simulation)
    price_improvement_probability: float = 0.05  # 5% chance of better price

    @classmethod
    def from_env(cls) -> "ExecutorConfig":
        """
        Load configuration from environment variables.

        All environment variables are optional; defaults are used if not set.

        Returns:
            ExecutorConfig with settings from environment

        Example:
            # In shell or .env file:
            export EXECUTION_MODE=paper
            export PAPER_FILL_PROB=0.90
            export EXECUTION_TIMEOUT=60

            # In code:
            config = ExecutorConfig.from_env()
        """
        def get_float(key: str, default: float) -> float:
            val = os.environ.get(key)
            if val:
                try:
                    return float(val)
                except ValueError:
                    pass
            return default

        def get_int(key: str, default: int) -> int:
            val = os.environ.get(key)
            if val:
                try:
                    return int(val)
                except ValueError:
                    pass
            return default

        def get_bool(key: str, default: bool) -> bool:
            val = os.environ.get(key, "").lower()
            if val in ("true", "1", "yes"):
                return True
            elif val in ("false", "0", "no"):
                return False
            return default

        def get_cents(key: str, default_dollars: float) -> int:
            """Convert dollar amount from env to cents."""
            val = os.environ.get(key)
            if val:
                try:
                    return int(float(val) * 100)
                except ValueError:
                    pass
            return int(default_dollars * 100)

        return cls(
            # Mode
            mode=os.environ.get("EXECUTION_MODE", "paper").lower(),

            # Paper trading settings
            fill_probability=get_float("PAPER_FILL_PROB", 0.95),
            fill_delay_ms=get_float("PAPER_FILL_DELAY_MS", 50.0),
            slippage_cents=get_int("PAPER_SLIPPAGE_CENTS", 1),
            partial_fill_probability=get_float("PAPER_PARTIAL_FILL_PROB", 0.10),

            # Timeout and retry
            timeout_seconds=get_float("EXECUTION_TIMEOUT", 30.0),
            max_retries=get_int("EXECUTION_MAX_RETRIES", 3),
            retry_delay_ms=get_float("EXECUTION_RETRY_DELAY_MS", 100.0),

            # Safety
            require_circuit_breaker=get_bool("REQUIRE_CIRCUIT_BREAKER", True),
            validate_balance=get_bool("VALIDATE_BALANCE", True),
            min_balance_cents=get_cents("MIN_BALANCE", 1.00),

            # Price improvement
            price_improvement_probability=get_float("PAPER_PRICE_IMPROVEMENT_PROB", 0.05),
        )

    @classmethod
    def paper_default(cls) -> "ExecutorConfig":
        """Create default paper trading configuration."""
        return cls(mode="paper")

    @classmethod
    def paper_realistic(cls) -> "ExecutorConfig":
        """
        Create realistic paper trading configuration.

        Lower fill probability, more slippage, more partial fills.
        Better for stress testing strategies.
        """
        return cls(
            mode="paper",
            fill_probability=0.85,
            fill_delay_ms=100.0,
            slippage_cents=2,
            partial_fill_probability=0.20,
            timeout_seconds=30.0,
        )

    @classmethod
    def paper_optimistic(cls) -> "ExecutorConfig":
        """
        Create optimistic paper trading configuration.

        High fill probability, low slippage.
        Good for initial strategy validation.
        """
        return cls(
            mode="paper",
            fill_probability=0.99,
            fill_delay_ms=20.0,
            slippage_cents=0,
            partial_fill_probability=0.05,
            price_improvement_probability=0.10,
        )

    @classmethod
    def live_default(cls) -> "ExecutorConfig":
        """Create default live trading configuration."""
        return cls(
            mode="live",
            timeout_seconds=30.0,
            max_retries=2,
            require_circuit_breaker=True,
            validate_balance=True,
        )

    @property
    def is_paper(self) -> bool:
        """Check if configured for paper trading."""
        return self.mode == "paper"

    @property
    def is_live(self) -> bool:
        """Check if configured for live trading."""
        return self.mode == "live"

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/serialization."""
        return {
            "mode": self.mode,
            "fill_probability": self.fill_probability,
            "fill_delay_ms": self.fill_delay_ms,
            "slippage_cents": self.slippage_cents,
            "partial_fill_probability": self.partial_fill_probability,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "retry_delay_ms": self.retry_delay_ms,
            "require_circuit_breaker": self.require_circuit_breaker,
            "validate_balance": self.validate_balance,
            "min_balance_cents": self.min_balance_cents,
            "price_improvement_probability": self.price_improvement_probability,
        }

    def __repr__(self) -> str:
        return (
            f"ExecutorConfig(mode={self.mode!r}, "
            f"fill_prob={self.fill_probability}, "
            f"timeout={self.timeout_seconds}s)"
        )


# =============================================================================
# ABSTRACT EXECUTOR BASE
# =============================================================================

class ExecutorBase(ABC):
    """
    Abstract base class for order executors.

    All executors (Paper, Live, Demo) MUST implement this interface.
    This allows the ExecutionService to swap executors without code changes.

    Required Methods:
        mode: Property returning executor mode ("paper", "live", "demo")
        is_connected: Property checking if executor can accept orders
        connect(): Establish connection (WebSocket, API session, etc.)
        disconnect(): Clean shutdown
        execute(order): Execute an order and return result
        cancel(order): Cancel an open order
        get_position(ticker): Get current position for a ticker
        get_balance(): Get available balance in cents

    Optional Hooks:
        on_fill(order, fill): Called when a fill is received
        on_order_update(order): Called when order state changes
        on_error(order, error): Called when an error occurs

    Context Manager:
        Executors support async context manager for clean resource management:

            async with MyExecutor() as executor:
                result = await executor.execute(order)

    Lifecycle:
        1. Create executor
        2. Call connect() or use context manager
        3. Execute orders
        4. Call disconnect() or exit context

    Example Implementation:
        class MyExecutor(ExecutorBase):
            @property
            def mode(self) -> str:
                return "my_mode"

            @property
            def is_connected(self) -> bool:
                return self._connected

            async def connect(self) -> None:
                self._connected = True

            async def disconnect(self) -> None:
                self._connected = False

            async def execute(self, order: ExecutionOrder) -> ExecutionResult:
                # Implementation here
                pass

            async def cancel(self, order: ExecutionOrder) -> ExecutionResult:
                # Implementation here
                pass

            async def get_position(self, ticker: str) -> dict:
                return {"ticker": ticker, "yes_contracts": 0, "no_contracts": 0}

            async def get_balance(self) -> int:
                return 10000_00  # $100.00 in cents
    """

    # =========================================================================
    # ABSTRACT PROPERTIES
    # =========================================================================

    @property
    @abstractmethod
    def mode(self) -> str:
        """
        Return executor mode identifier.

        Returns:
            str: One of "paper", "live", "demo", or custom mode

        Example:
            @property
            def mode(self) -> str:
                return "paper"
        """
        pass

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """
        Check if executor can accept orders.

        Returns:
            bool: True if connected and ready to accept orders

        Note:
            This should return True only when the executor is fully
            initialized and can process execute() calls.

        Example:
            @property
            def is_connected(self) -> bool:
                return self._ws is not None and self._ws.open
        """
        pass

    # =========================================================================
    # ABSTRACT LIFECYCLE METHODS
    # =========================================================================

    @abstractmethod
    async def connect(self) -> None:
        """
        Establish connection (WebSocket, API session, etc.).

        Called automatically when using context manager.
        Must be idempotent (safe to call multiple times).

        Raises:
            ConnectionError: If connection cannot be established
            AuthenticationError: If authentication fails

        Example:
            async def connect(self) -> None:
                if self._connected:
                    return
                self._session = aiohttp.ClientSession()
                await self._verify_auth()
                self._connected = True
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """
        Clean shutdown - close connections and cleanup resources.

        Called automatically when exiting context manager.
        Must be idempotent and handle already-disconnected state.

        Should:
            - Cancel pending orders (optional, configurable)
            - Close WebSocket connections
            - Close HTTP sessions
            - Release any locks or resources

        Example:
            async def disconnect(self) -> None:
                self._connected = False
                if self._ws:
                    await self._ws.close()
                if self._session:
                    await self._session.close()
        """
        pass

    # =========================================================================
    # ABSTRACT EXECUTION METHODS
    # =========================================================================

    @abstractmethod
    async def execute(self, order: "ExecutionOrder") -> "ExecutionResult":
        """
        Execute an order and return result.

        This is the main entry point for order execution.
        Implementations should handle the complete order lifecycle:
        1. Pre-submission validation
        2. Submit order to exchange/simulator
        3. Wait for fills or timeout
        4. Handle retries if configured
        5. Return ExecutionResult with final state

        Args:
            order: ExecutionOrder to execute (from models.py)

        Returns:
            ExecutionResult with order outcome, fills, and metrics

        Raises:
            ExecutorError: If a fatal error prevents execution

        State Transitions:
            PENDING → SUBMITTED → RESTING → FILLED
                                         → PARTIALLY_FILLED → FILLED
                                         → CANCELED
                   → REJECTED
                   → EXPIRED

        Example:
            async def execute(self, order: ExecutionOrder) -> ExecutionResult:
                order.mode = ExecutionMode.PAPER

                # Submit order
                order.mark_submitted(kalshi_order_id="paper_123")

                # Simulate fill
                if random.random() < 0.95:
                    fill = order.record_fill(
                        contracts=order.contracts,
                        price=order.limit_price
                    )
                    return ExecutionResult.from_order(order, fills=[fill])
                else:
                    order.mark_expired()
                    return ExecutionResult.from_order(order, fills=[])
        """
        pass

    @abstractmethod
    async def cancel(self, order: "ExecutionOrder") -> "ExecutionResult":
        """
        Cancel an open order.

        Args:
            order: ExecutionOrder to cancel

        Returns:
            ExecutionResult with canceled state and any partial fills

        Note:
            Order may have partially filled before cancellation.
            Check order.filled_contracts for partial execution.

        Example:
            async def cancel(self, order: ExecutionOrder) -> ExecutionResult:
                if order.is_terminal:
                    return ExecutionResult.from_order(order, fills=[])

                await self._client.cancel_order(order.kalshi_order_id)
                order.mark_canceled(reason="User requested")
                return ExecutionResult.from_order(order, fills=[])
        """
        pass

    # =========================================================================
    # ABSTRACT QUERY METHODS
    # =========================================================================

    @abstractmethod
    async def get_position(self, ticker: str) -> dict:
        """
        Get current position for a ticker.

        Args:
            ticker: Market ticker (e.g., "KXNFL-26JAN11-BUF")

        Returns:
            dict with position details:
                {
                    "ticker": str,
                    "yes_contracts": int,
                    "no_contracts": int,
                    "avg_yes_price": float,  # Optional
                    "avg_no_price": float,   # Optional
                    "unrealized_pnl": int,   # Optional, cents
                }

        Note:
            Return empty position (0 contracts) if no position exists.

        Example:
            async def get_position(self, ticker: str) -> dict:
                pos = self._positions.get(ticker)
                if pos:
                    return {
                        "ticker": ticker,
                        "yes_contracts": pos.yes_count,
                        "no_contracts": pos.no_count,
                    }
                return {"ticker": ticker, "yes_contracts": 0, "no_contracts": 0}
        """
        pass

    @abstractmethod
    async def get_balance(self) -> int:
        """
        Get available balance in cents.

        Returns:
            int: Available balance in cents (e.g., 10000 = $100.00)

        Note:
            This should return the balance available for new orders,
            excluding any funds locked in pending orders.

        Example:
            async def get_balance(self) -> int:
                balance = await self._client.get_balance()
                return balance.available_cents
        """
        pass

    # =========================================================================
    # OPTIONAL LIFECYCLE HOOKS
    # =========================================================================

    async def on_fill(self, order: "ExecutionOrder", fill: "Fill") -> None:
        """
        Called when a fill is received.

        Override to add custom fill handling (logging, notifications, etc.).
        Default implementation does nothing.

        Args:
            order: The order that received a fill
            fill: The fill details

        Example:
            async def on_fill(self, order: ExecutionOrder, fill: Fill) -> None:
                print(f"Fill: {fill.contracts}@{fill.price}c for {order.ticker}")
                await self._metrics.record_fill(fill)
        """
        pass

    async def on_order_update(self, order: "ExecutionOrder") -> None:
        """
        Called when order state changes.

        Override for custom state change handling (logging, tracking, etc.).
        Default implementation does nothing.

        Args:
            order: The order with updated state

        Example:
            async def on_order_update(self, order: ExecutionOrder) -> None:
                print(f"Order {order.execution_id}: {order.state.value}")
                await self._database.update_order(order)
        """
        pass

    async def on_error(self, order: "ExecutionOrder", error: Exception) -> None:
        """
        Called when an error occurs during execution.

        Override for custom error handling (alerting, recovery, etc.).
        Default implementation does nothing.

        Args:
            order: The order that encountered an error
            error: The exception that occurred

        Example:
            async def on_error(self, order: ExecutionOrder, error: Exception) -> None:
                print(f"Error executing {order.ticker}: {error}")
                await self._alerter.send_alert(f"Execution error: {error}")
        """
        pass

    # =========================================================================
    # CONTEXT MANAGER SUPPORT
    # =========================================================================

    async def __aenter__(self) -> "ExecutorBase":
        """
        Async context manager entry.

        Automatically connects the executor.

        Usage:
            async with MyExecutor() as executor:
                result = await executor.execute(order)
        """
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        """
        Async context manager exit.

        Automatically disconnects the executor.
        Does not suppress exceptions.
        """
        await self.disconnect()

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(mode={self.mode!r}, connected={self.is_connected})"


# =============================================================================
# TYPE ALIASES FOR CONVENIENCE
# =============================================================================

# Callback type for fill notifications
FillCallback = Any  # Callable[[ExecutionOrder, Fill], Awaitable[None]]

# Callback type for order updates
OrderUpdateCallback = Any  # Callable[[ExecutionOrder], Awaitable[None]]


# =============================================================================
# VALIDATION / TEST
# =============================================================================

def validate_config():
    """Validate ExecutorConfig functionality."""
    print("=" * 60)
    print("EXECUTOR CONFIG VALIDATION")
    print("=" * 60)
    print()

    # Test default config
    default = ExecutorConfig()
    print("Default Config:")
    print(f"  Mode: {default.mode}")
    print(f"  Fill Probability: {default.fill_probability}")
    print(f"  Timeout: {default.timeout_seconds}s")
    print()

    # Test from_env (uses defaults since env not set)
    from_env = ExecutorConfig.from_env()
    print("From Environment:")
    print(f"  Mode: {from_env.mode}")
    print(f"  Is Paper: {from_env.is_paper}")
    print()

    # Test presets
    print("Presets:")
    for name, factory in [
        ("paper_default", ExecutorConfig.paper_default),
        ("paper_realistic", ExecutorConfig.paper_realistic),
        ("paper_optimistic", ExecutorConfig.paper_optimistic),
        ("live_default", ExecutorConfig.live_default),
    ]:
        config = factory()
        print(f"  {name}: {config}")
    print()

    # Test serialization
    print("Serialization (to_dict):")
    print(f"  Keys: {list(default.to_dict().keys())[:5]}...")
    print()

    print("CONFIG VALIDATION PASSED")
    print("=" * 60)


if __name__ == "__main__":
    validate_config()
