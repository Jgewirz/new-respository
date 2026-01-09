"""
Circuit Breaker - Execution Risk Controls

Multi-layered protection system that prevents runaway losses:

1. Per-Trade Limits
   - Max position size (contracts)
   - Max risk per trade (dollars)

2. Daily Limits
   - Max daily loss
   - Max daily trades
   - Max daily volume

3. Position Concentration
   - Max exposure per event
   - Max exposure per sport

4. Rate Limiting
   - Orders per minute
   - Minimum time between orders

5. Consecutive Loss Protection
   - Halt after N consecutive losses

6. Manual Controls
   - Kill switch for emergency halt
   - Resume capability

The circuit breaker MUST be checked before EVERY order submission.

Usage:
    from app.execution.circuit_breaker import CircuitBreaker

    breaker = CircuitBreaker.from_env()

    # Before every trade
    allowed, reason = breaker.check_trade(
        ticker="KXNFL-26JAN11-BUF",
        contracts=25,
        risk_cents=1200,
        sport="nfl",
    )
    if not allowed:
        print(f"Trade blocked: {reason}")
        return

    # After execution completes
    breaker.record_trade(result)

    # After settlement
    breaker.record_settlement(pnl_cents=-500)

    # Emergency halt
    breaker.halt("Unusual market conditions")
    breaker.resume()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Optional, TYPE_CHECKING
import os
import threading

if TYPE_CHECKING:
    from app.execution.models import ExecutionResult


# =============================================================================
# ENUMS
# =============================================================================

class BreakerState(str, Enum):
    """
    Circuit breaker states.

    CLOSED: Normal operation, trades allowed
    OPEN: Halted, no trades allowed
    HALF_OPEN: Testing after cooldown (future use)
    """
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class BlockReason(str, Enum):
    """Reasons why a trade might be blocked."""
    MANUAL_HALT = "manual_halt"
    BREAKER_OPEN = "breaker_open"
    POSITION_SIZE = "position_size_exceeded"
    RISK_PER_TRADE = "risk_per_trade_exceeded"
    DAILY_LOSS = "daily_loss_limit"
    DAILY_TRADES = "daily_trade_limit"
    DAILY_VOLUME = "daily_volume_limit"
    CONSECUTIVE_LOSSES = "consecutive_loss_limit"
    RATE_LIMIT = "rate_limit_exceeded"
    EVENT_CONCENTRATION = "event_concentration_limit"
    SPORT_CONCENTRATION = "sport_concentration_limit"
    INSUFFICIENT_BALANCE = "insufficient_balance"


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class RiskLimits:
    """
    Configurable risk parameters.

    All monetary values in cents for precision.
    Load from environment or use defaults.
    """
    # Per-trade limits
    max_position_size: int = 100              # Max contracts per trade
    max_risk_per_trade_cents: int = 500_00    # Max risk per trade ($5.00)

    # Daily limits
    max_daily_loss_cents: int = 10000         # Daily loss limit ($100.00)
    max_daily_trades: int = 500               # Max trades per day
    max_daily_volume: int = 50000             # Max contracts per day

    # Concentration limits
    max_position_per_event: int = 5000        # Max contracts per sporting event
    max_exposure_per_sport_pct: float = 0.30  # Max 30% of bankroll per sport

    # Rate limits
    max_orders_per_minute: int = 20           # Order submission rate
    min_time_between_orders_ms: int = 100     # Minimum 100ms between orders

    # Consecutive loss circuit breaker
    max_consecutive_losses: int = 100         # Halt after 100 consecutive losses

    # Cooldown after breaker trips
    cooldown_minutes: int = 30                # Wait 30min before auto-resume

    # Balance requirements
    min_balance_cents: int = 100_00           # Require $1.00 minimum balance

    @classmethod
    def from_env(cls) -> "RiskLimits":
        """
        Load risk limits from environment variables.

        Environment variables (all optional, defaults used if not set):
            MAX_POSITION_SIZE: Max contracts per trade
            MAX_RISK_PER_TRADE: Max risk in dollars (converted to cents)
            MAX_DAILY_LOSS: Daily loss limit in dollars
            MAX_DAILY_TRADES: Max trades per day
            MAX_DAILY_VOLUME: Max contracts per day
            MAX_POSITION_PER_EVENT: Max contracts per event
            MAX_ORDERS_PER_MINUTE: Rate limit
            MAX_CONSECUTIVE_LOSSES: Consecutive loss limit
            CIRCUIT_BREAKER_COOLDOWN: Cooldown minutes
        """
        def get_cents(key: str, default_dollars: float) -> int:
            """Get value as cents from dollar amount in env."""
            val = os.environ.get(key)
            if val:
                return int(float(val) * 100)
            return int(default_dollars * 100)

        return cls(
            max_position_size=int(os.environ.get("MAX_POSITION_SIZE", 100)),
            max_risk_per_trade_cents=get_cents("MAX_RISK_PER_TRADE", 5.00),
            max_daily_loss_cents=get_cents("MAX_DAILY_LOSS", 100.00),
            max_daily_trades=int(os.environ.get("MAX_DAILY_TRADES", 500)),
            max_daily_volume=int(os.environ.get("MAX_DAILY_VOLUME", 50000)),
            max_position_per_event=int(os.environ.get("MAX_POSITION_PER_EVENT", 5000)),
            max_orders_per_minute=int(os.environ.get("MAX_ORDERS_PER_MINUTE", 20)),
            max_consecutive_losses=int(os.environ.get("MAX_CONSECUTIVE_LOSSES", 100)),
            cooldown_minutes=int(os.environ.get("CIRCUIT_BREAKER_COOLDOWN", 30)),
            min_balance_cents=get_cents("MIN_BALANCE", 1.00),
        )

    @classmethod
    def conservative(cls) -> "RiskLimits":
        """Conservative risk limits for initial testing."""
        return cls(
            max_position_size=25,
            max_risk_per_trade_cents=100_00,    # $1.00
            max_daily_loss_cents=500_00,         # $5.00
            max_daily_trades=20,
            max_daily_volume=500,
            max_position_per_event=100,
            max_orders_per_minute=5,
            max_consecutive_losses=3,
        )

    @classmethod
    def aggressive(cls) -> "RiskLimits":
        """Aggressive risk limits for experienced trading."""
        return cls(
            max_position_size=500,
            max_risk_per_trade_cents=2500_00,    # $25.00
            max_daily_loss_cents=5000_00,        # $50.00
            max_daily_trades=200,
            max_daily_volume=20000,
            max_position_per_event=2000,
            max_orders_per_minute=20,
            max_consecutive_losses=10,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "max_position_size": self.max_position_size,
            "max_risk_per_trade_dollars": self.max_risk_per_trade_cents / 100,
            "max_daily_loss_dollars": self.max_daily_loss_cents / 100,
            "max_daily_trades": self.max_daily_trades,
            "max_daily_volume": self.max_daily_volume,
            "max_position_per_event": self.max_position_per_event,
            "max_exposure_per_sport_pct": self.max_exposure_per_sport_pct,
            "max_orders_per_minute": self.max_orders_per_minute,
            "min_time_between_orders_ms": self.min_time_between_orders_ms,
            "max_consecutive_losses": self.max_consecutive_losses,
            "cooldown_minutes": self.cooldown_minutes,
        }


# =============================================================================
# DAILY STATISTICS
# =============================================================================

@dataclass
class DailyStats:
    """
    Daily trading statistics.

    Reset at midnight UTC each day.
    """
    date: str = ""                            # YYYY-MM-DD (UTC)
    trades_count: int = 0                     # Completed trades
    contracts_traded: int = 0                 # Total contracts
    volume_cents: int = 0                     # Total notional volume
    realized_pnl_cents: int = 0               # Settled P&L
    unrealized_pnl_cents: int = 0             # Open position P&L estimate
    consecutive_losses: int = 0               # Current loss streak
    consecutive_wins: int = 0                 # Current win streak
    max_drawdown_cents: int = 0               # Worst intraday drawdown
    peak_pnl_cents: int = 0                   # Best intraday P&L
    orders_submitted: int = 0                 # Orders sent (including unfilled)
    orders_filled: int = 0                    # Orders that filled
    orders_rejected: int = 0                  # Orders rejected
    last_trade_time: Optional[datetime] = None

    def __post_init__(self):
        """Initialize date if not set."""
        if not self.date:
            self.date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    @property
    def fill_rate(self) -> float:
        """Percentage of orders that filled."""
        if self.orders_submitted == 0:
            return 0.0
        return self.orders_filled / self.orders_submitted

    @property
    def win_rate(self) -> float:
        """Win rate based on settled trades."""
        total = self.consecutive_wins + self.consecutive_losses
        if total == 0:
            return 0.0
        # This is approximate - would need full trade history for accuracy
        return 0.5  # Default to 50% if unknown

    @property
    def is_profitable(self) -> bool:
        """Currently profitable today."""
        return self.realized_pnl_cents > 0

    def record_trade(
        self,
        contracts: int,
        cost_cents: int,
        filled: bool = True,
    ) -> None:
        """Record a completed trade."""
        self.orders_submitted += 1

        if filled:
            self.orders_filled += 1
            self.trades_count += 1
            self.contracts_traded += contracts
            self.volume_cents += cost_cents
            self.last_trade_time = datetime.now(timezone.utc)

    def record_rejection(self) -> None:
        """Record a rejected order."""
        self.orders_submitted += 1
        self.orders_rejected += 1

    def record_settlement(self, pnl_cents: int) -> None:
        """
        Record P&L from a settled position.

        Args:
            pnl_cents: Profit (positive) or loss (negative) in cents
        """
        self.realized_pnl_cents += pnl_cents

        # Update win/loss streaks
        if pnl_cents > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        elif pnl_cents < 0:
            self.consecutive_losses += 1
            self.consecutive_wins = 0

        # Update peak and drawdown
        if self.realized_pnl_cents > self.peak_pnl_cents:
            self.peak_pnl_cents = self.realized_pnl_cents

        drawdown = self.peak_pnl_cents - self.realized_pnl_cents
        if drawdown > self.max_drawdown_cents:
            self.max_drawdown_cents = drawdown

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "date": self.date,
            "trades_count": self.trades_count,
            "contracts_traded": self.contracts_traded,
            "volume_dollars": self.volume_cents / 100,
            "realized_pnl_dollars": self.realized_pnl_cents / 100,
            "consecutive_losses": self.consecutive_losses,
            "consecutive_wins": self.consecutive_wins,
            "max_drawdown_dollars": self.max_drawdown_cents / 100,
            "orders_submitted": self.orders_submitted,
            "orders_filled": self.orders_filled,
            "fill_rate_pct": round(self.fill_rate * 100, 1),
            "last_trade_time": self.last_trade_time.isoformat() if self.last_trade_time else None,
        }


# =============================================================================
# TRADE CHECK RESULT
# =============================================================================

@dataclass
class TradeCheckResult:
    """
    Result of a circuit breaker trade check.

    Contains whether trade is allowed and detailed reasoning.
    """
    allowed: bool
    reason: str = ""
    block_reason: Optional[BlockReason] = None
    checks_performed: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def blocked(self) -> bool:
        """Trade was blocked."""
        return not self.allowed

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "block_reason": self.block_reason.value if self.block_reason else None,
            "checks": self.checks_performed,
            "timestamp": self.timestamp.isoformat(),
        }


# =============================================================================
# CIRCUIT BREAKER
# =============================================================================

class CircuitBreaker:
    """
    Multi-layered execution risk control system.

    Thread-safe implementation for concurrent access.

    Usage:
        breaker = CircuitBreaker.from_env()

        # Before every trade
        result = breaker.check_trade(
            ticker="KXNFL-26JAN11-BUF",
            contracts=25,
            risk_cents=1200,
        )
        if not result.allowed:
            log.warning(f"Trade blocked: {result.reason}")
            return

        # After trade completes
        breaker.record_trade(execution_result)

        # After settlement
        breaker.record_settlement(pnl_cents=-500)

        # Emergency controls
        breaker.halt("Market anomaly detected")
        breaker.resume()
    """

    def __init__(
        self,
        limits: Optional[RiskLimits] = None,
        bankroll_cents: int = 1_000_000,  # $10,000 default
    ):
        """
        Initialize circuit breaker.

        Args:
            limits: RiskLimits configuration
            bankroll_cents: Total bankroll for concentration calculations
        """
        self.limits = limits or RiskLimits()
        self.bankroll_cents = bankroll_cents

        # State
        self._state = BreakerState.CLOSED
        self._daily_stats = DailyStats()

        # Position tracking
        self._positions_by_event: dict[str, int] = {}   # event_id -> contracts
        self._positions_by_sport: dict[str, int] = {}   # sport -> contracts
        self._exposure_by_sport: dict[str, int] = {}    # sport -> exposure_cents

        # Rate limiting
        self._order_timestamps: list[datetime] = []
        self._last_order_time: Optional[datetime] = None

        # Manual controls
        self._manual_halt = False
        self._halt_reason = ""
        self._halt_time: Optional[datetime] = None

        # Thread safety
        self._lock = threading.RLock()

    @classmethod
    def from_env(cls, bankroll_cents: Optional[int] = None) -> "CircuitBreaker":
        """
        Create circuit breaker from environment configuration.

        Args:
            bankroll_cents: Override bankroll (default from TRADING_BANKROLL env)
        """
        limits = RiskLimits.from_env()

        if bankroll_cents is None:
            bankroll_dollars = float(os.environ.get("TRADING_BANKROLL", 10000))
            bankroll_cents = int(bankroll_dollars * 100)

        return cls(limits=limits, bankroll_cents=bankroll_cents)

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def state(self) -> BreakerState:
        """Current breaker state."""
        return self._state

    @property
    def is_open(self) -> bool:
        """Breaker is open (trading halted)."""
        return self._state == BreakerState.OPEN

    @property
    def is_closed(self) -> bool:
        """Breaker is closed (trading allowed)."""
        return self._state == BreakerState.CLOSED

    @property
    def daily_stats(self) -> DailyStats:
        """Current daily statistics."""
        self._check_day_rollover()
        return self._daily_stats

    @property
    def is_halted(self) -> bool:
        """Trading is halted (manual or automatic)."""
        return self._manual_halt or self._state == BreakerState.OPEN

    # -------------------------------------------------------------------------
    # Main Check Method
    # -------------------------------------------------------------------------

    def check_trade(
        self,
        ticker: str,
        contracts: int,
        risk_cents: int,
        sport: str = "",
        event_id: str = "",
        balance_cents: Optional[int] = None,
    ) -> TradeCheckResult:
        """
        Check if a trade is allowed.

        This method MUST be called before every order submission.
        All checks are performed atomically.

        Args:
            ticker: Kalshi market ticker
            contracts: Number of contracts
            risk_cents: Risk amount in cents (contracts * price)
            sport: Sport category (e.g., "nfl", "nba")
            event_id: Unique event identifier
            balance_cents: Current account balance (optional)

        Returns:
            TradeCheckResult with allowed status and reasoning
        """
        with self._lock:
            checks = {}
            self._check_day_rollover()

            # 1. Check manual halt
            if self._manual_halt:
                return TradeCheckResult(
                    allowed=False,
                    reason=f"Manual halt: {self._halt_reason}",
                    block_reason=BlockReason.MANUAL_HALT,
                    checks_performed={"manual_halt": False},
                )
            checks["manual_halt"] = True

            # 2. Check breaker state
            if self._state == BreakerState.OPEN:
                return TradeCheckResult(
                    allowed=False,
                    reason="Circuit breaker is OPEN - trading halted",
                    block_reason=BlockReason.BREAKER_OPEN,
                    checks_performed={"breaker_state": False},
                )
            checks["breaker_state"] = True

            # 3. Check position size
            if contracts > self.limits.max_position_size:
                return TradeCheckResult(
                    allowed=False,
                    reason=f"Position size {contracts} exceeds limit {self.limits.max_position_size}",
                    block_reason=BlockReason.POSITION_SIZE,
                    checks_performed={**checks, "position_size": False},
                )
            checks["position_size"] = True

            # 4. Check risk per trade
            if risk_cents > self.limits.max_risk_per_trade_cents:
                return TradeCheckResult(
                    allowed=False,
                    reason=f"Risk ${risk_cents/100:.2f} exceeds limit ${self.limits.max_risk_per_trade_cents/100:.2f}",
                    block_reason=BlockReason.RISK_PER_TRADE,
                    checks_performed={**checks, "risk_per_trade": False},
                )
            checks["risk_per_trade"] = True

            # 5. Check daily loss limit
            if -self._daily_stats.realized_pnl_cents >= self.limits.max_daily_loss_cents:
                return TradeCheckResult(
                    allowed=False,
                    reason=f"Daily loss limit ${self.limits.max_daily_loss_cents/100:.2f} reached",
                    block_reason=BlockReason.DAILY_LOSS,
                    checks_performed={**checks, "daily_loss": False},
                )
            checks["daily_loss"] = True

            # 6. Check daily trade count
            if self._daily_stats.trades_count >= self.limits.max_daily_trades:
                return TradeCheckResult(
                    allowed=False,
                    reason=f"Daily trade limit {self.limits.max_daily_trades} reached",
                    block_reason=BlockReason.DAILY_TRADES,
                    checks_performed={**checks, "daily_trades": False},
                )
            checks["daily_trades"] = True

            # 7. Check daily volume
            if self._daily_stats.contracts_traded + contracts > self.limits.max_daily_volume:
                return TradeCheckResult(
                    allowed=False,
                    reason=f"Daily volume limit {self.limits.max_daily_volume} contracts would be exceeded",
                    block_reason=BlockReason.DAILY_VOLUME,
                    checks_performed={**checks, "daily_volume": False},
                )
            checks["daily_volume"] = True

            # 8. Check consecutive losses
            if self._daily_stats.consecutive_losses >= self.limits.max_consecutive_losses:
                return TradeCheckResult(
                    allowed=False,
                    reason=f"Consecutive loss limit {self.limits.max_consecutive_losses} reached",
                    block_reason=BlockReason.CONSECUTIVE_LOSSES,
                    checks_performed={**checks, "consecutive_losses": False},
                )
            checks["consecutive_losses"] = True

            # 9. Check rate limit
            rate_ok, rate_reason = self._check_rate_limit()
            if not rate_ok:
                return TradeCheckResult(
                    allowed=False,
                    reason=rate_reason,
                    block_reason=BlockReason.RATE_LIMIT,
                    checks_performed={**checks, "rate_limit": False},
                )
            checks["rate_limit"] = True

            # 10. Check event concentration (if event_id provided)
            if event_id:
                current_event_position = self._positions_by_event.get(event_id, 0)
                if current_event_position + contracts > self.limits.max_position_per_event:
                    return TradeCheckResult(
                        allowed=False,
                        reason=f"Event position limit {self.limits.max_position_per_event} would be exceeded",
                        block_reason=BlockReason.EVENT_CONCENTRATION,
                        checks_performed={**checks, "event_concentration": False},
                    )
            checks["event_concentration"] = True

            # 11. Check sport concentration (if sport provided)
            if sport:
                current_sport_exposure = self._exposure_by_sport.get(sport, 0)
                max_sport_exposure = int(self.bankroll_cents * self.limits.max_exposure_per_sport_pct)
                if current_sport_exposure + risk_cents > max_sport_exposure:
                    return TradeCheckResult(
                        allowed=False,
                        reason=f"Sport exposure limit {self.limits.max_exposure_per_sport_pct*100:.0f}% would be exceeded",
                        block_reason=BlockReason.SPORT_CONCENTRATION,
                        checks_performed={**checks, "sport_concentration": False},
                    )
            checks["sport_concentration"] = True

            # 12. Check balance (if provided)
            if balance_cents is not None:
                if balance_cents < self.limits.min_balance_cents:
                    return TradeCheckResult(
                        allowed=False,
                        reason=f"Balance ${balance_cents/100:.2f} below minimum ${self.limits.min_balance_cents/100:.2f}",
                        block_reason=BlockReason.INSUFFICIENT_BALANCE,
                        checks_performed={**checks, "balance": False},
                    )
                if balance_cents < risk_cents:
                    return TradeCheckResult(
                        allowed=False,
                        reason=f"Insufficient balance for trade risk",
                        block_reason=BlockReason.INSUFFICIENT_BALANCE,
                        checks_performed={**checks, "balance": False},
                    )
            checks["balance"] = True

            # All checks passed
            return TradeCheckResult(
                allowed=True,
                reason="All checks passed",
                checks_performed=checks,
            )

    # -------------------------------------------------------------------------
    # Recording Methods
    # -------------------------------------------------------------------------

    def record_trade(
        self,
        result: "ExecutionResult",
        event_id: str = "",
        sport: str = "",
    ) -> None:
        """
        Record a completed trade execution.

        Call this after ExecutionResult is available.

        Args:
            result: ExecutionResult from executor
            event_id: Event identifier for concentration tracking
            sport: Sport category for concentration tracking
        """
        with self._lock:
            self._check_day_rollover()

            order = result.order

            if result.success or result.partial:
                # Record successful trade
                self._daily_stats.record_trade(
                    contracts=order.filled_contracts,
                    cost_cents=result.total_cost_cents,
                    filled=True,
                )

                # Update position tracking
                if event_id:
                    current = self._positions_by_event.get(event_id, 0)
                    self._positions_by_event[event_id] = current + order.filled_contracts

                if sport:
                    current_contracts = self._positions_by_sport.get(sport, 0)
                    current_exposure = self._exposure_by_sport.get(sport, 0)
                    self._positions_by_sport[sport] = current_contracts + order.filled_contracts
                    self._exposure_by_sport[sport] = current_exposure + result.total_cost_cents

                # Record order timestamp for rate limiting
                self._last_order_time = datetime.now(timezone.utc)

            elif result.failed:
                self._daily_stats.record_rejection()

    def record_settlement(self, pnl_cents: int, event_id: str = "", sport: str = "") -> None:
        """
        Record P&L from a settled position.

        Call this when a position settles (event completes).

        Args:
            pnl_cents: Profit (positive) or loss (negative) in cents
            event_id: Event that settled
            sport: Sport category
        """
        with self._lock:
            self._check_day_rollover()

            # Record P&L
            self._daily_stats.record_settlement(pnl_cents)

            # Clear position tracking for settled event
            if event_id and event_id in self._positions_by_event:
                del self._positions_by_event[event_id]

            # Check if we should trip the breaker
            if self._daily_stats.consecutive_losses >= self.limits.max_consecutive_losses:
                self._trip_breaker(f"Consecutive loss limit ({self.limits.max_consecutive_losses}) reached")

            if -self._daily_stats.realized_pnl_cents >= self.limits.max_daily_loss_cents:
                self._trip_breaker(f"Daily loss limit (${self.limits.max_daily_loss_cents/100:.2f}) reached")

    def record_order_attempt(self) -> None:
        """
        Record an order submission attempt for rate limiting.

        Call this when submitting an order (before knowing if it fills).
        """
        with self._lock:
            now = datetime.now(timezone.utc)
            self._order_timestamps.append(now)
            self._last_order_time = now

    # -------------------------------------------------------------------------
    # Manual Controls
    # -------------------------------------------------------------------------

    def halt(self, reason: str = "Manual halt requested") -> None:
        """
        Manual halt - immediately stop all trading.

        Use for emergency situations or end of trading day.

        Args:
            reason: Description of why trading was halted
        """
        with self._lock:
            self._manual_halt = True
            self._halt_reason = reason
            self._halt_time = datetime.now(timezone.utc)
            self._state = BreakerState.OPEN

    def resume(self) -> bool:
        """
        Resume trading after manual halt.

        Returns:
            True if resumed, False if automatic trip still active
        """
        with self._lock:
            self._manual_halt = False
            self._halt_reason = ""

            # Check if there's still an automatic trip condition
            if self._daily_stats.consecutive_losses >= self.limits.max_consecutive_losses:
                return False
            if -self._daily_stats.realized_pnl_cents >= self.limits.max_daily_loss_cents:
                return False

            self._state = BreakerState.CLOSED
            return True

    def reset_daily_stats(self) -> None:
        """
        Force reset daily statistics.

        Normally happens automatically at midnight UTC.
        """
        with self._lock:
            self._daily_stats = DailyStats()
            self._positions_by_event.clear()
            self._order_timestamps.clear()

            # Don't clear manual halt
            if not self._manual_halt:
                self._state = BreakerState.CLOSED

    def reset_consecutive_losses(self) -> None:
        """Reset consecutive loss counter (e.g., after manual review)."""
        with self._lock:
            self._daily_stats.consecutive_losses = 0

            # Resume if that was the only trip reason
            if not self._manual_halt and self._state == BreakerState.OPEN:
                if -self._daily_stats.realized_pnl_cents < self.limits.max_daily_loss_cents:
                    self._state = BreakerState.CLOSED

    # -------------------------------------------------------------------------
    # Private Methods
    # -------------------------------------------------------------------------

    def _trip_breaker(self, reason: str) -> None:
        """Trip the circuit breaker."""
        self._state = BreakerState.OPEN
        self._halt_reason = reason
        self._halt_time = datetime.now(timezone.utc)

    def _check_day_rollover(self) -> None:
        """Reset stats if new trading day (UTC)."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._daily_stats.date != today:
            # New day - reset stats
            self._daily_stats = DailyStats(date=today)
            self._positions_by_event.clear()
            self._positions_by_sport.clear()
            self._exposure_by_sport.clear()
            self._order_timestamps.clear()

            # Auto-resume on new day (unless manual halt)
            if not self._manual_halt:
                self._state = BreakerState.CLOSED
                self._halt_reason = ""

    def _check_rate_limit(self) -> tuple[bool, str]:
        """
        Check if we're within rate limits.

        Returns:
            (allowed, reason)
        """
        now = datetime.now(timezone.utc)

        # Check minimum time between orders
        if self._last_order_time:
            elapsed_ms = (now - self._last_order_time).total_seconds() * 1000
            if elapsed_ms < self.limits.min_time_between_orders_ms:
                return False, f"Minimum {self.limits.min_time_between_orders_ms}ms between orders"

        # Check orders per minute
        one_minute_ago = now - timedelta(minutes=1)
        self._order_timestamps = [
            ts for ts in self._order_timestamps
            if ts > one_minute_ago
        ]

        if len(self._order_timestamps) >= self.limits.max_orders_per_minute:
            return False, f"Rate limit: {self.limits.max_orders_per_minute} orders per minute"

        return True, ""

    # -------------------------------------------------------------------------
    # Status Methods
    # -------------------------------------------------------------------------

    def get_status(self) -> dict:
        """
        Get complete circuit breaker status.

        Useful for monitoring and dashboards.
        """
        with self._lock:
            self._check_day_rollover()

            return {
                "state": self._state.value,
                "is_halted": self.is_halted,
                "manual_halt": self._manual_halt,
                "halt_reason": self._halt_reason,
                "halt_time": self._halt_time.isoformat() if self._halt_time else None,
                "limits": self.limits.to_dict(),
                "daily_stats": self._daily_stats.to_dict(),
                "positions_by_event_count": len(self._positions_by_event),
                "positions_by_sport": dict(self._positions_by_sport),
                "exposure_by_sport_dollars": {
                    k: v / 100 for k, v in self._exposure_by_sport.items()
                },
                "bankroll_dollars": self.bankroll_cents / 100,
                "orders_in_last_minute": len(self._order_timestamps),
            }

    def get_remaining_capacity(self) -> dict:
        """
        Get remaining trading capacity.

        Useful for position sizing decisions.
        """
        with self._lock:
            self._check_day_rollover()

            remaining_trades = max(0, self.limits.max_daily_trades - self._daily_stats.trades_count)
            remaining_volume = max(0, self.limits.max_daily_volume - self._daily_stats.contracts_traded)
            remaining_loss = max(0, self.limits.max_daily_loss_cents + self._daily_stats.realized_pnl_cents)
            remaining_losses = max(0, self.limits.max_consecutive_losses - self._daily_stats.consecutive_losses)

            return {
                "remaining_trades": remaining_trades,
                "remaining_volume_contracts": remaining_volume,
                "remaining_loss_dollars": remaining_loss / 100,
                "remaining_consecutive_losses": remaining_losses,
                "can_trade": self._state == BreakerState.CLOSED and not self._manual_halt,
            }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CircuitBreaker(state={self._state.value}, "
            f"trades={self._daily_stats.trades_count}, "
            f"pnl=${self._daily_stats.realized_pnl_cents/100:.2f})"
        )
