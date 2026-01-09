"""
Execution Layer Data Models

Core dataclasses for order lifecycle tracking from Signal through settlement.

Data Flow:
    Signal (from detector) → ExecutionOrder → Fill(s) → ExecutionResult

Timestamp Convention:
    All timestamps use nanoseconds (int) for QuestDB ILP compatibility.
    Use now_ns() for current time, ns_to_datetime() to convert back.

Integration with Existing Components:
    - Signal: from app.arb.detector (input)
    - KalshiClient: from app.connectors.kalshi.client (execution)
    - QuestDB: from app.data.questdb (logging)

Usage:
    from app.execution.models import ExecutionOrder, Fill, ExecutionResult

    # Create from signal
    order = ExecutionOrder.from_signal(signal, mode=ExecutionMode.PAPER)

    # Record fill
    fill = order.record_fill(contracts=10, price=48)

    # Build result
    result = ExecutionResult.from_order(order, fills=[fill])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, TYPE_CHECKING
import uuid

# Type checking imports (avoid circular imports)
if TYPE_CHECKING:
    from app.arb.detector import Signal


# =============================================================================
# UTILITIES
# =============================================================================

def now_ns() -> int:
    """Current UTC time in nanoseconds (for QuestDB)."""
    return int(datetime.now(timezone.utc).timestamp() * 1e9)


def ns_to_datetime(ns: int) -> datetime:
    """Convert nanoseconds to datetime."""
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc)


def generate_id() -> str:
    """Generate unique ID for orders/fills."""
    return str(uuid.uuid4())


# =============================================================================
# ENUMS
# =============================================================================

class ExecutionMode(str, Enum):
    """
    Execution environment.

    PAPER: Simulated fills for testing
    LIVE: Real orders against Kalshi API
    """
    PAPER = "paper"
    LIVE = "live"


class OrderState(str, Enum):
    """
    Order lifecycle states.

    State transitions:
        PENDING → SUBMITTED → RESTING → FILLED
                           ↘ PARTIALLY_FILLED → FILLED
                           ↘ CANCELED
                ↘ REJECTED
                ↘ EXPIRED
    """
    PENDING = "pending"              # Created, not yet submitted
    SUBMITTED = "submitted"          # Sent to Kalshi, awaiting ack
    RESTING = "resting"              # On order book, awaiting fill
    PARTIALLY_FILLED = "partial"     # Some contracts filled
    FILLED = "filled"                # Fully executed
    CANCELED = "canceled"            # Canceled by user or system
    REJECTED = "rejected"            # Rejected by Kalshi
    EXPIRED = "expired"              # TTL expired without fill


class OrderSide(str, Enum):
    """
    Contract side - YES or NO.

    Aligned with Kalshi API and detector Signal.action:
        Signal.action="BUY_YES" → OrderSide.YES
        Signal.action="BUY_NO" → OrderSide.NO
    """
    YES = "yes"
    NO = "no"


class OrderAction(str, Enum):
    """
    Order action - BUY or SELL.

    For arbitrage, we typically BUY when we detect edge.
    SELL is used for position exit.
    """
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """
    Order type.

    LIMIT: Specify max price (recommended for low latency)
    MARKET: Fill at any price (not recommended - slippage risk)
    """
    LIMIT = "limit"
    MARKET = "market"


# =============================================================================
# EXECUTION ORDER
# =============================================================================

@dataclass
class ExecutionOrder:
    """
    Complete order record with full lifecycle tracking.

    Created from Signal, tracks through submission and fill.
    All timestamps in nanoseconds for QuestDB compatibility.

    Attributes:
        execution_id: Internal unique ID for this execution attempt
        client_order_id: UUID sent to Kalshi for idempotency
        kalshi_order_id: Order ID returned by Kalshi after submission

        signal_id: ID of the Signal that triggered this order
        signal_edge_cents: Edge in cents at time of signal
        signal_confidence: Confidence score (0-100) at time of signal

        ticker: Kalshi market ticker (e.g., "KXNFL-26JAN11-BUF")
        side: "yes" or "no"
        action: "buy" or "sell"
        order_type: "limit" or "market"

        limit_price: Price in cents (1-99)
        contracts: Number of contracts ordered

        state: Current order state
        mode: PAPER or LIVE

        filled_contracts: Contracts filled so far
        remaining_contracts: Contracts still pending
        average_fill_price: Volume-weighted average fill price

        created_at_ns: Order creation timestamp (nanoseconds)
        submitted_at_ns: When sent to Kalshi
        first_fill_at_ns: First fill received
        completed_at_ns: Terminal state reached

        error_code: Error code if rejected
        error_message: Error message if rejected

        retry_count: Number of retry attempts
        max_retries: Maximum retries allowed
    """
    # Identifiers
    execution_id: str = field(default_factory=generate_id)
    client_order_id: str = field(default_factory=generate_id)
    kalshi_order_id: Optional[str] = None

    # Signal origin (for tracking)
    signal_id: str = ""
    signal_edge_cents: int = 0
    signal_confidence: int = 0

    # Order details
    ticker: str = ""
    side: str = ""              # "yes" or "no"
    action: str = "buy"         # "buy" or "sell"
    order_type: str = "limit"   # "limit" or "market"

    # Pricing
    limit_price: int = 0        # Price in cents (1-99)
    contracts: int = 0          # Number of contracts

    # State tracking
    state: OrderState = OrderState.PENDING
    mode: ExecutionMode = ExecutionMode.PAPER

    # Fill tracking
    filled_contracts: int = 0
    remaining_contracts: int = 0
    average_fill_price: float = 0.0

    # Timestamps (nanoseconds for QuestDB)
    created_at_ns: int = field(default_factory=now_ns)
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
        """Initialize remaining contracts from total."""
        if self.remaining_contracts == 0 and self.contracts > 0:
            self.remaining_contracts = self.contracts

    # -------------------------------------------------------------------------
    # Factory Methods
    # -------------------------------------------------------------------------

    @classmethod
    def from_signal(
        cls,
        signal: "Signal",
        mode: ExecutionMode = ExecutionMode.PAPER,
        price_offset: int = 1,
    ) -> "ExecutionOrder":
        """
        Create ExecutionOrder from detection Signal.

        Args:
            signal: Signal from EdgeDetector
            mode: PAPER or LIVE execution
            price_offset: Cents to add to ask for faster fill (default 1)

        Returns:
            ExecutionOrder ready for submission

        Example:
            signal = detector.detect(kalshi_market, sportsbook)
            if signal.should_trade:
                order = ExecutionOrder.from_signal(signal)
        """
        # Determine side from action
        if signal.action == "BUY_YES":
            side = OrderSide.YES.value
            base_price = signal.kalshi.yes_ask
        elif signal.action == "BUY_NO":
            side = OrderSide.NO.value
            base_price = signal.kalshi.no_ask
        else:
            raise ValueError(f"Cannot create order from action: {signal.action}")

        # Add offset for faster fill, cap at 99
        limit_price = min(99, base_price + price_offset)

        return cls(
            signal_id=signal.signal_id,
            signal_edge_cents=signal.edge.best_edge,
            signal_confidence=signal.confidence_score,
            ticker=signal.kalshi.ticker,
            side=side,
            action=OrderAction.BUY.value,
            order_type=OrderType.LIMIT.value,
            limit_price=limit_price,
            contracts=signal.recommended_contracts,
            mode=mode,
        )

    # -------------------------------------------------------------------------
    # State Properties
    # -------------------------------------------------------------------------

    @property
    def is_terminal(self) -> bool:
        """Order is in a final state (no more updates expected)."""
        return self.state in (
            OrderState.FILLED,
            OrderState.CANCELED,
            OrderState.REJECTED,
            OrderState.EXPIRED,
        )

    @property
    def is_active(self) -> bool:
        """Order is active (submitted and awaiting fill)."""
        return self.state in (
            OrderState.SUBMITTED,
            OrderState.RESTING,
            OrderState.PARTIALLY_FILLED,
        )

    @property
    def is_fillable(self) -> bool:
        """Order can still receive fills."""
        return self.state in (
            OrderState.RESTING,
            OrderState.PARTIALLY_FILLED,
        )

    @property
    def fill_rate(self) -> float:
        """Percentage of order filled (0.0 to 1.0)."""
        if self.contracts == 0:
            return 0.0
        return self.filled_contracts / self.contracts

    @property
    def fill_rate_pct(self) -> float:
        """Percentage of order filled (0 to 100)."""
        return self.fill_rate * 100

    # -------------------------------------------------------------------------
    # Timing Properties
    # -------------------------------------------------------------------------

    @property
    def created_at(self) -> datetime:
        """Creation time as datetime."""
        return ns_to_datetime(self.created_at_ns)

    @property
    def submitted_at(self) -> Optional[datetime]:
        """Submission time as datetime."""
        if self.submitted_at_ns == 0:
            return None
        return ns_to_datetime(self.submitted_at_ns)

    @property
    def completed_at(self) -> Optional[datetime]:
        """Completion time as datetime."""
        if self.completed_at_ns == 0:
            return None
        return ns_to_datetime(self.completed_at_ns)

    @property
    def submission_latency_ms(self) -> float:
        """Time from creation to submission in milliseconds."""
        if self.submitted_at_ns == 0:
            return 0.0
        return (self.submitted_at_ns - self.created_at_ns) / 1e6

    @property
    def fill_latency_ms(self) -> float:
        """Time from submission to first fill in milliseconds."""
        if self.first_fill_at_ns == 0 or self.submitted_at_ns == 0:
            return 0.0
        return (self.first_fill_at_ns - self.submitted_at_ns) / 1e6

    @property
    def total_latency_ms(self) -> float:
        """Time from creation to completion in milliseconds."""
        if self.completed_at_ns == 0:
            return 0.0
        return (self.completed_at_ns - self.created_at_ns) / 1e6

    # -------------------------------------------------------------------------
    # Financial Properties
    # -------------------------------------------------------------------------

    @property
    def total_cost_cents(self) -> int:
        """Total cost of filled contracts in cents."""
        if self.filled_contracts == 0:
            return 0
        return int(self.filled_contracts * self.average_fill_price)

    @property
    def total_cost_dollars(self) -> float:
        """Total cost of filled contracts in dollars."""
        return self.total_cost_cents / 100

    @property
    def potential_profit_cents(self) -> int:
        """Potential profit if position wins (100c - cost per contract)."""
        if self.filled_contracts == 0:
            return 0
        return int(self.filled_contracts * (100 - self.average_fill_price))

    @property
    def potential_profit_dollars(self) -> float:
        """Potential profit in dollars."""
        return self.potential_profit_cents / 100

    @property
    def risk_reward_ratio(self) -> float:
        """Risk/reward ratio (cost / potential profit)."""
        if self.potential_profit_cents == 0:
            return 0.0
        return self.total_cost_cents / self.potential_profit_cents

    # -------------------------------------------------------------------------
    # State Mutation Methods
    # -------------------------------------------------------------------------

    def mark_submitted(self, kalshi_order_id: Optional[str] = None) -> None:
        """Mark order as submitted to Kalshi."""
        self.state = OrderState.SUBMITTED
        self.submitted_at_ns = now_ns()
        if kalshi_order_id:
            self.kalshi_order_id = kalshi_order_id

    def mark_resting(self) -> None:
        """Mark order as resting on book."""
        self.state = OrderState.RESTING

    def mark_filled(self) -> None:
        """Mark order as fully filled."""
        self.state = OrderState.FILLED
        self.completed_at_ns = now_ns()
        self.remaining_contracts = 0

    def mark_canceled(self, reason: str = "") -> None:
        """Mark order as canceled."""
        self.state = OrderState.CANCELED
        self.completed_at_ns = now_ns()
        if reason:
            self.error_message = reason

    def mark_rejected(self, code: str = "", message: str = "") -> None:
        """Mark order as rejected by Kalshi."""
        self.state = OrderState.REJECTED
        self.completed_at_ns = now_ns()
        self.error_code = code
        self.error_message = message

    def mark_expired(self) -> None:
        """Mark order as expired (timeout)."""
        self.state = OrderState.EXPIRED
        self.completed_at_ns = now_ns()

    def record_fill(self, contracts: int, price: int, is_maker: bool = False) -> "Fill":
        """
        Record a fill event and update order state.

        Args:
            contracts: Number of contracts filled
            price: Fill price in cents
            is_maker: True if we provided liquidity

        Returns:
            Fill object for this fill event
        """
        # Record first fill time
        if self.first_fill_at_ns == 0:
            self.first_fill_at_ns = now_ns()

        # Update average price (volume-weighted)
        if self.filled_contracts == 0:
            self.average_fill_price = float(price)
        else:
            total_value = (self.average_fill_price * self.filled_contracts) + (price * contracts)
            self.average_fill_price = total_value / (self.filled_contracts + contracts)

        # Update counts
        self.filled_contracts += contracts
        self.remaining_contracts = max(0, self.remaining_contracts - contracts)

        # Update state
        if self.remaining_contracts <= 0:
            self.mark_filled()
        else:
            self.state = OrderState.PARTIALLY_FILLED

        # Create fill record
        return Fill(
            execution_id=self.execution_id,
            kalshi_order_id=self.kalshi_order_id or "",
            contracts=contracts,
            price=price,
            is_maker=is_maker,
        )

    # -------------------------------------------------------------------------
    # Retry Methods
    # -------------------------------------------------------------------------

    def can_retry(self) -> bool:
        """Check if order can be retried."""
        return (
            self.retry_count < self.max_retries
            and self.state in (OrderState.CANCELED, OrderState.EXPIRED)
        )

    def prepare_retry(self, price_improvement: int = 1) -> None:
        """
        Prepare order for retry with improved price.

        Args:
            price_improvement: Cents to improve limit price
        """
        self.retry_count += 1
        self.state = OrderState.PENDING
        self.limit_price = min(99, self.limit_price + price_improvement)
        self.kalshi_order_id = None
        self.client_order_id = generate_id()  # New ID for idempotency
        self.submitted_at_ns = 0

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/serialization."""
        return {
            "execution_id": self.execution_id,
            "client_order_id": self.client_order_id,
            "kalshi_order_id": self.kalshi_order_id,
            "signal_id": self.signal_id,
            "signal_edge_cents": self.signal_edge_cents,
            "signal_confidence": self.signal_confidence,
            "ticker": self.ticker,
            "side": self.side,
            "action": self.action,
            "order_type": self.order_type,
            "limit_price": self.limit_price,
            "contracts": self.contracts,
            "state": self.state.value,
            "mode": self.mode.value,
            "filled_contracts": self.filled_contracts,
            "remaining_contracts": self.remaining_contracts,
            "average_fill_price": round(self.average_fill_price, 2),
            "total_cost_cents": self.total_cost_cents,
            "potential_profit_cents": self.potential_profit_cents,
            "created_at": self.created_at.isoformat(),
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "submission_latency_ms": round(self.submission_latency_ms, 2),
            "total_latency_ms": round(self.total_latency_ms, 2),
            "error_code": self.error_code,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
        }

    def to_ilp_fields(self) -> dict:
        """
        Get fields for QuestDB ILP write.

        Returns fields formatted for ILP ingestion.
        """
        return {
            "execution_id": self.execution_id,
            "kalshi_order_id": self.kalshi_order_id or "",
            "signal_id": self.signal_id,
            "ticker": self.ticker,
            "side": self.side,
            "action": self.action,
            "limit_price": self.limit_price,
            "contracts": self.contracts,
            "filled_contracts": self.filled_contracts,
            "average_fill_price": self.average_fill_price,
            "state": self.state.value,
            "mode": self.mode.value,
            "signal_edge_cents": self.signal_edge_cents,
            "submission_latency_ms": self.submission_latency_ms,
            "total_latency_ms": self.total_latency_ms,
        }


# =============================================================================
# FILL
# =============================================================================

@dataclass
class Fill:
    """
    Individual fill event.

    Represents a single execution of contracts within an order.
    An order may have multiple fills (partial fills).

    Attributes:
        fill_id: Unique ID for this fill
        execution_id: Parent order's execution ID
        kalshi_order_id: Kalshi's order ID
        contracts: Number of contracts in this fill
        price: Fill price in cents
        timestamp_ns: Fill time in nanoseconds
        is_maker: True if we provided liquidity (better fees)
    """
    fill_id: str = field(default_factory=generate_id)
    execution_id: str = ""
    kalshi_order_id: str = ""

    contracts: int = 0
    price: int = 0              # Fill price in cents

    timestamp_ns: int = field(default_factory=now_ns)
    is_maker: bool = False      # True if we provided liquidity

    @property
    def timestamp(self) -> datetime:
        """Fill time as datetime."""
        return ns_to_datetime(self.timestamp_ns)

    @property
    def notional_cents(self) -> int:
        """Total value in cents (contracts * price)."""
        return self.contracts * self.price

    @property
    def notional_dollars(self) -> float:
        """Total value in dollars."""
        return self.notional_cents / 100

    @property
    def potential_profit_cents(self) -> int:
        """Potential profit if position wins."""
        return self.contracts * (100 - self.price)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "fill_id": self.fill_id,
            "execution_id": self.execution_id,
            "kalshi_order_id": self.kalshi_order_id,
            "contracts": self.contracts,
            "price": self.price,
            "timestamp": self.timestamp.isoformat(),
            "is_maker": self.is_maker,
            "notional_cents": self.notional_cents,
            "potential_profit_cents": self.potential_profit_cents,
        }

    def to_ilp_fields(self) -> dict:
        """Get fields for QuestDB ILP write."""
        return {
            "fill_id": self.fill_id,
            "execution_id": self.execution_id,
            "kalshi_order_id": self.kalshi_order_id,
            "contracts": self.contracts,
            "price": self.price,
            "is_maker": self.is_maker,
            "notional_cents": self.notional_cents,
        }


# =============================================================================
# EXECUTION RESULT
# =============================================================================

@dataclass
class ExecutionResult:
    """
    Final result of an execution attempt.

    Aggregates order state and fills after execution completes.
    Returned by executors (PaperExecutor, KalshiExecutor).

    Attributes:
        order: The ExecutionOrder (with final state)
        fills: List of Fill events
        total_cost_cents: Total amount paid
        potential_profit_cents: Profit if position wins
        slippage_cents: Price difference from expected
        execution_time_ms: Total execution duration
    """
    order: ExecutionOrder
    fills: list[Fill] = field(default_factory=list)

    # Calculated metrics
    total_cost_cents: int = 0
    potential_profit_cents: int = 0
    slippage_cents: float = 0.0
    execution_time_ms: float = 0.0

    def __post_init__(self):
        """Calculate metrics from order and fills."""
        if self.fills and self.total_cost_cents == 0:
            self.total_cost_cents = sum(f.notional_cents for f in self.fills)

        if self.fills and self.potential_profit_cents == 0:
            self.potential_profit_cents = sum(f.potential_profit_cents for f in self.fills)

        if self.execution_time_ms == 0:
            self.execution_time_ms = self.order.total_latency_ms

    @classmethod
    def from_order(
        cls,
        order: ExecutionOrder,
        fills: Optional[list[Fill]] = None,
    ) -> "ExecutionResult":
        """
        Create result from completed order.

        Args:
            order: Completed ExecutionOrder
            fills: List of fills (optional, will calculate from order if not provided)

        Returns:
            ExecutionResult with calculated metrics
        """
        fills = fills or []

        # Calculate slippage (actual avg price vs intended limit price)
        if order.filled_contracts > 0:
            expected_price = order.limit_price - 1  # We added offset
            slippage = order.average_fill_price - expected_price
        else:
            slippage = 0.0

        return cls(
            order=order,
            fills=fills,
            total_cost_cents=order.total_cost_cents,
            potential_profit_cents=order.potential_profit_cents,
            slippage_cents=slippage,
            execution_time_ms=order.total_latency_ms,
        )

    # -------------------------------------------------------------------------
    # Status Properties
    # -------------------------------------------------------------------------

    @property
    def success(self) -> bool:
        """Order was fully filled."""
        return self.order.state == OrderState.FILLED

    @property
    def partial(self) -> bool:
        """Order was partially filled."""
        return (
            self.order.filled_contracts > 0
            and self.order.state != OrderState.FILLED
        )

    @property
    def failed(self) -> bool:
        """Order failed (rejected or no fills)."""
        return (
            self.order.state == OrderState.REJECTED
            or (self.order.is_terminal and self.order.filled_contracts == 0)
        )

    @property
    def fill_count(self) -> int:
        """Number of fill events."""
        return len(self.fills)

    # -------------------------------------------------------------------------
    # Financial Properties
    # -------------------------------------------------------------------------

    @property
    def total_cost_dollars(self) -> float:
        """Total cost in dollars."""
        return self.total_cost_cents / 100

    @property
    def potential_profit_dollars(self) -> float:
        """Potential profit in dollars."""
        return self.potential_profit_cents / 100

    @property
    def expected_value_cents(self) -> float:
        """
        Expected value based on signal edge.

        EV = (edge / 100) * contracts * 100
        """
        if self.order.filled_contracts == 0:
            return 0.0

        edge_pct = self.order.signal_edge_cents / 100
        return edge_pct * self.order.filled_contracts * 100

    @property
    def expected_value_dollars(self) -> float:
        """Expected value in dollars."""
        return self.expected_value_cents / 100

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "order": self.order.to_dict(),
            "fills": [f.to_dict() for f in self.fills],
            "metrics": {
                "success": self.success,
                "partial": self.partial,
                "failed": self.failed,
                "fill_count": self.fill_count,
                "total_cost_cents": self.total_cost_cents,
                "total_cost_dollars": self.total_cost_dollars,
                "potential_profit_cents": self.potential_profit_cents,
                "potential_profit_dollars": self.potential_profit_dollars,
                "slippage_cents": round(self.slippage_cents, 2),
                "execution_time_ms": round(self.execution_time_ms, 2),
                "expected_value_cents": round(self.expected_value_cents, 2),
            },
        }

    def summary(self) -> str:
        """Human-readable summary string."""
        status = "FILLED" if self.success else "PARTIAL" if self.partial else "FAILED"
        return (
            f"{status}: {self.order.filled_contracts}/{self.order.contracts} contracts "
            f"@ avg {self.order.average_fill_price:.1f}c "
            f"(cost: ${self.total_cost_dollars:.2f}, "
            f"potential: ${self.potential_profit_dollars:.2f}, "
            f"slippage: {self.slippage_cents:.1f}c, "
            f"latency: {self.execution_time_ms:.1f}ms)"
        )
