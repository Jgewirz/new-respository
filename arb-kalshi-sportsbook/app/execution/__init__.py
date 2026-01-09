"""
Execution Layer Module

Handles order lifecycle from Signal to Fill:
- ExecutorBase: Abstract interface for all executors
- ExecutorConfig: Configuration for executors
- ExecutionOrder: Complete order record with lifecycle tracking
- Fill: Individual fill events
- ExecutionResult: Final execution outcome
- CircuitBreaker: Risk controls and trading limits

Integration Points:
- Input: Signal from app.arb.detector
- Output: Orders submitted via app.connectors.kalshi.client
- Storage: Metrics logged to QuestDB via app.data.questdb

Usage:
    from app.execution import (
        ExecutorBase, ExecutorConfig,
        ExecutionOrder, Fill, ExecutionResult, OrderState,
        CircuitBreaker, RiskLimits,
    )

    # Setup circuit breaker
    breaker = CircuitBreaker.from_env()

    # Check before trading
    result = breaker.check_trade(ticker="KXNFL-26JAN11-BUF", contracts=25, risk_cents=1200)
    if not result.allowed:
        print(f"Blocked: {result.reason}")
        return

    # Create order from signal
    order = ExecutionOrder.from_signal(signal)

    # Track fills
    fill = order.record_fill(contracts=25, price=48)

    # Build result
    exec_result = ExecutionResult.from_order(order, fills=[fill])

    # Record with circuit breaker
    breaker.record_trade(exec_result)

Executor Implementation:
    class MyExecutor(ExecutorBase):
        @property
        def mode(self) -> str:
            return "my_mode"

        async def execute(self, order: ExecutionOrder) -> ExecutionResult:
            # Implementation here
            ...
"""

from app.execution.models import (
    # Enums
    ExecutionMode,
    OrderState,
    OrderSide,
    OrderAction,
    # Dataclasses
    ExecutionOrder,
    Fill,
    ExecutionResult,
    # Utilities
    now_ns,
    ns_to_datetime,
    generate_id,
)

from app.execution.circuit_breaker import (
    # Enums
    BreakerState,
    BlockReason,
    # Config
    RiskLimits,
    # Stats
    DailyStats,
    # Result
    TradeCheckResult,
    # Main class
    CircuitBreaker,
)

from app.execution.base import (
    # Config
    ExecutorConfig,
    # Abstract Base
    ExecutorBase,
)

from app.execution.paper_executor import (
    # Dataclasses
    PaperPosition,
    PaperExecutorState,
    # Main class
    PaperExecutor,
    # Factory
    create_paper_executor,
)

__all__ = [
    # Base Executor Interface (Phase 1)
    "ExecutorBase",
    "ExecutorConfig",
    # Paper Executor (Phase 2)
    "PaperExecutor",
    "PaperPosition",
    "PaperExecutorState",
    "create_paper_executor",
    # Model Enums
    "ExecutionMode",
    "OrderState",
    "OrderSide",
    "OrderAction",
    # Model Dataclasses
    "ExecutionOrder",
    "Fill",
    "ExecutionResult",
    # Model Utilities
    "now_ns",
    "ns_to_datetime",
    "generate_id",
    # Circuit Breaker Enums
    "BreakerState",
    "BlockReason",
    # Circuit Breaker Config
    "RiskLimits",
    # Circuit Breaker Stats
    "DailyStats",
    # Circuit Breaker Result
    "TradeCheckResult",
    # Circuit Breaker Main
    "CircuitBreaker",
]
