"""
Unit Tests for Circuit Breaker

Tests all risk control checks:
1. Per-trade limits (position size, risk amount)
2. Daily limits (loss, trades, volume)
3. Consecutive loss protection
4. Rate limiting
5. Position concentration
6. Manual halt/resume

Run with:
    pytest tests/test_circuit_breaker.py -v
"""

import pytest
import time
from datetime import datetime, timezone, timedelta

from app.execution.circuit_breaker import (
    CircuitBreaker,
    RiskLimits,
    DailyStats,
    TradeCheckResult,
    BreakerState,
    BlockReason,
)
from app.execution.models import (
    ExecutionOrder,
    ExecutionResult,
    Fill,
    OrderState,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def default_limits() -> RiskLimits:
    """Default risk limits for testing."""
    return RiskLimits(
        max_position_size=100,
        max_risk_per_trade_cents=500_00,      # $5.00
        max_daily_loss_cents=1000_00,         # $10.00
        max_daily_trades=50,
        max_daily_volume=5000,
        max_position_per_event=500,
        max_exposure_per_sport_pct=0.30,
        max_orders_per_minute=10,
        min_time_between_orders_ms=100,
        max_consecutive_losses=5,
    )


@pytest.fixture
def breaker(default_limits) -> CircuitBreaker:
    """Circuit breaker with default limits."""
    return CircuitBreaker(
        limits=default_limits,
        bankroll_cents=1_000_000,  # $10,000
    )


@pytest.fixture
def tight_limits() -> RiskLimits:
    """Very restrictive limits for testing blocks."""
    return RiskLimits(
        max_position_size=10,
        max_risk_per_trade_cents=100_00,      # $1.00
        max_daily_loss_cents=200_00,          # $2.00
        max_daily_trades=5,
        max_daily_volume=50,
        max_position_per_event=20,
        max_exposure_per_sport_pct=0.10,
        max_orders_per_minute=2,
        min_time_between_orders_ms=500,
        max_consecutive_losses=2,
    )


@pytest.fixture
def tight_breaker(tight_limits) -> CircuitBreaker:
    """Circuit breaker with tight limits."""
    return CircuitBreaker(
        limits=tight_limits,
        bankroll_cents=100_000,  # $1,000
    )


@pytest.fixture
def sample_order() -> ExecutionOrder:
    """Sample execution order."""
    return ExecutionOrder(
        ticker="KXNFL-26JAN11-BUF",
        side="yes",
        action="buy",
        limit_price=48,
        contracts=25,
    )


@pytest.fixture
def sample_result(sample_order) -> ExecutionResult:
    """Sample successful execution result."""
    sample_order.mark_submitted(kalshi_order_id="test_123")
    fill = sample_order.record_fill(contracts=25, price=48)
    return ExecutionResult.from_order(sample_order, fills=[fill])


# =============================================================================
# RISK LIMITS TESTS
# =============================================================================

class TestRiskLimits:
    """Test RiskLimits configuration."""

    def test_default_initialization(self):
        """RiskLimits should have sensible defaults."""
        limits = RiskLimits()

        assert limits.max_position_size == 100
        assert limits.max_daily_trades == 50
        assert limits.max_consecutive_losses == 5

    def test_from_env_uses_defaults(self):
        """from_env should use defaults when env vars not set."""
        limits = RiskLimits.from_env()

        # Should have default values
        assert limits.max_position_size > 0
        assert limits.max_daily_loss_cents > 0

    def test_conservative_preset(self):
        """Conservative preset should be more restrictive."""
        conservative = RiskLimits.conservative()
        default = RiskLimits()

        assert conservative.max_position_size < default.max_position_size
        assert conservative.max_daily_loss_cents < default.max_daily_loss_cents

    def test_aggressive_preset(self):
        """Aggressive preset should be more permissive."""
        aggressive = RiskLimits.aggressive()
        default = RiskLimits()

        assert aggressive.max_position_size > default.max_position_size
        assert aggressive.max_daily_loss_cents > default.max_daily_loss_cents

    def test_to_dict(self):
        """to_dict should include all fields."""
        limits = RiskLimits()
        d = limits.to_dict()

        assert "max_position_size" in d
        assert "max_daily_loss_dollars" in d
        assert "max_consecutive_losses" in d


# =============================================================================
# DAILY STATS TESTS
# =============================================================================

class TestDailyStats:
    """Test DailyStats tracking."""

    def test_default_initialization(self):
        """DailyStats should initialize with today's date."""
        stats = DailyStats()

        assert stats.date == datetime.now(timezone.utc).strftime("%Y-%m-%d")
        assert stats.trades_count == 0
        assert stats.realized_pnl_cents == 0

    def test_record_trade(self):
        """record_trade should update counts."""
        stats = DailyStats()

        stats.record_trade(contracts=25, cost_cents=1200)

        assert stats.trades_count == 1
        assert stats.contracts_traded == 25
        assert stats.volume_cents == 1200
        assert stats.orders_submitted == 1
        assert stats.orders_filled == 1

    def test_record_rejection(self):
        """record_rejection should track rejected orders."""
        stats = DailyStats()

        stats.record_rejection()

        assert stats.orders_submitted == 1
        assert stats.orders_rejected == 1
        assert stats.orders_filled == 0

    def test_record_settlement_profit(self):
        """record_settlement should track wins."""
        stats = DailyStats()

        stats.record_settlement(pnl_cents=500)

        assert stats.realized_pnl_cents == 500
        assert stats.consecutive_wins == 1
        assert stats.consecutive_losses == 0

    def test_record_settlement_loss(self):
        """record_settlement should track losses."""
        stats = DailyStats()

        stats.record_settlement(pnl_cents=-300)

        assert stats.realized_pnl_cents == -300
        assert stats.consecutive_losses == 1
        assert stats.consecutive_wins == 0

    def test_consecutive_loss_tracking(self):
        """Consecutive losses should accumulate."""
        stats = DailyStats()

        stats.record_settlement(pnl_cents=-100)
        stats.record_settlement(pnl_cents=-100)
        stats.record_settlement(pnl_cents=-100)

        assert stats.consecutive_losses == 3

        # Win resets streak
        stats.record_settlement(pnl_cents=100)

        assert stats.consecutive_losses == 0
        assert stats.consecutive_wins == 1

    def test_drawdown_tracking(self):
        """Drawdown should track from peak."""
        stats = DailyStats()

        stats.record_settlement(pnl_cents=500)  # Peak: 500
        stats.record_settlement(pnl_cents=-200) # PnL: 300, DD: 200
        stats.record_settlement(pnl_cents=-200) # PnL: 100, DD: 400

        assert stats.peak_pnl_cents == 500
        assert stats.max_drawdown_cents == 400

    def test_fill_rate(self):
        """fill_rate should calculate correctly."""
        stats = DailyStats()

        stats.record_trade(contracts=10, cost_cents=500)
        stats.record_trade(contracts=10, cost_cents=500)
        stats.record_rejection()

        assert stats.fill_rate == 2 / 3


# =============================================================================
# TRADE CHECK RESULT TESTS
# =============================================================================

class TestTradeCheckResult:
    """Test TradeCheckResult dataclass."""

    def test_allowed_result(self):
        """Allowed result should have correct flags."""
        result = TradeCheckResult(allowed=True, reason="All checks passed")

        assert result.allowed
        assert not result.blocked
        assert result.block_reason is None

    def test_blocked_result(self):
        """Blocked result should have correct flags."""
        result = TradeCheckResult(
            allowed=False,
            reason="Position too large",
            block_reason=BlockReason.POSITION_SIZE,
        )

        assert not result.allowed
        assert result.blocked
        assert result.block_reason == BlockReason.POSITION_SIZE

    def test_to_dict(self):
        """to_dict should include all fields."""
        result = TradeCheckResult(
            allowed=False,
            reason="Test",
            block_reason=BlockReason.DAILY_LOSS,
        )
        d = result.to_dict()

        assert d["allowed"] is False
        assert d["block_reason"] == "daily_loss_limit"


# =============================================================================
# CIRCUIT BREAKER BASIC TESTS
# =============================================================================

class TestCircuitBreakerBasic:
    """Test basic circuit breaker functionality."""

    def test_default_initialization(self, breaker):
        """Breaker should start in closed state."""
        assert breaker.state == BreakerState.CLOSED
        assert breaker.is_closed
        assert not breaker.is_open
        assert not breaker.is_halted

    def test_from_env(self):
        """from_env should create working breaker."""
        breaker = CircuitBreaker.from_env()

        assert breaker.state == BreakerState.CLOSED
        assert breaker.limits is not None

    def test_check_trade_allowed(self, breaker):
        """Normal trade should be allowed."""
        result = breaker.check_trade(
            ticker="KXNFL-26JAN11-BUF",
            contracts=25,
            risk_cents=1200,
        )

        assert result.allowed
        assert result.reason == "All checks passed"
        assert len(result.checks_performed) > 0

    def test_repr(self, breaker):
        """__repr__ should work."""
        s = repr(breaker)
        assert "CircuitBreaker" in s
        assert "closed" in s


# =============================================================================
# PER-TRADE LIMIT TESTS
# =============================================================================

class TestPerTradeLimits:
    """Test per-trade risk limits."""

    def test_position_size_exceeded(self, tight_breaker):
        """Should block when position size exceeds limit."""
        result = tight_breaker.check_trade(
            ticker="TEST",
            contracts=50,  # Exceeds limit of 10
            risk_cents=2500,
        )

        assert not result.allowed
        assert result.block_reason == BlockReason.POSITION_SIZE
        assert "10" in result.reason  # Should mention limit

    def test_position_size_at_limit(self, tight_breaker):
        """Should allow when at exactly the limit."""
        result = tight_breaker.check_trade(
            ticker="TEST",
            contracts=10,  # Exactly at limit
            risk_cents=500,
        )

        assert result.allowed

    def test_risk_per_trade_exceeded(self, tight_breaker):
        """Should block when risk exceeds limit."""
        result = tight_breaker.check_trade(
            ticker="TEST",
            contracts=5,
            risk_cents=200_00,  # Exceeds $1.00 limit
        )

        assert not result.allowed
        assert result.block_reason == BlockReason.RISK_PER_TRADE


# =============================================================================
# DAILY LIMIT TESTS
# =============================================================================

class TestDailyLimits:
    """Test daily trading limits."""

    def test_daily_trade_limit(self, tight_breaker):
        """Should block when daily trades exceeded."""
        # Simulate 5 trades (the limit)
        for i in range(5):
            tight_breaker._daily_stats.record_trade(contracts=5, cost_cents=250)

        result = tight_breaker.check_trade(
            ticker="TEST",
            contracts=5,
            risk_cents=250,
        )

        assert not result.allowed
        assert result.block_reason == BlockReason.DAILY_TRADES

    def test_daily_volume_limit(self, tight_breaker):
        """Should block when daily volume exceeded."""
        # Simulate volume near limit (50 contracts)
        tight_breaker._daily_stats.contracts_traded = 48

        result = tight_breaker.check_trade(
            ticker="TEST",
            contracts=5,  # Would exceed 50
            risk_cents=250,
        )

        assert not result.allowed
        assert result.block_reason == BlockReason.DAILY_VOLUME

    def test_daily_loss_limit(self, tight_breaker):
        """Should block when daily loss exceeded."""
        # Simulate loss at limit ($2.00)
        tight_breaker._daily_stats.realized_pnl_cents = -200_00

        result = tight_breaker.check_trade(
            ticker="TEST",
            contracts=5,
            risk_cents=250,
        )

        assert not result.allowed
        assert result.block_reason == BlockReason.DAILY_LOSS


# =============================================================================
# CONSECUTIVE LOSS TESTS
# =============================================================================

class TestConsecutiveLosses:
    """Test consecutive loss protection."""

    def test_consecutive_loss_limit(self, tight_breaker):
        """Should block after consecutive losses."""
        # Simulate 2 consecutive losses (the limit)
        tight_breaker._daily_stats.consecutive_losses = 2

        result = tight_breaker.check_trade(
            ticker="TEST",
            contracts=5,
            risk_cents=250,
        )

        assert not result.allowed
        assert result.block_reason == BlockReason.CONSECUTIVE_LOSSES

    def test_consecutive_losses_via_settlement(self, tight_breaker):
        """Consecutive losses via record_settlement should trip breaker."""
        # Record 2 losses
        tight_breaker.record_settlement(pnl_cents=-100)
        tight_breaker.record_settlement(pnl_cents=-100)

        # Breaker should now be open
        assert tight_breaker.is_open

    def test_win_resets_consecutive_losses(self, breaker):
        """Win should reset consecutive loss counter."""
        breaker._daily_stats.consecutive_losses = 3

        breaker.record_settlement(pnl_cents=100)  # Win

        assert breaker._daily_stats.consecutive_losses == 0

    def test_reset_consecutive_losses(self, breaker):
        """Manual reset should clear counter."""
        breaker._daily_stats.consecutive_losses = 4
        breaker._state = BreakerState.OPEN

        breaker.reset_consecutive_losses()

        assert breaker._daily_stats.consecutive_losses == 0
        assert breaker.is_closed  # Should auto-resume


# =============================================================================
# RATE LIMIT TESTS
# =============================================================================

class TestRateLimits:
    """Test rate limiting."""

    def test_orders_per_minute_limit(self, tight_breaker):
        """Should block when rate limit exceeded."""
        # Simulate max orders already submitted
        now = datetime.now(timezone.utc)
        tight_breaker._order_timestamps = [now - timedelta(seconds=i) for i in range(2)]

        result = tight_breaker.check_trade(
            ticker="TEST",
            contracts=5,
            risk_cents=250,
        )

        assert not result.allowed
        assert result.block_reason == BlockReason.RATE_LIMIT

    def test_min_time_between_orders(self, tight_breaker):
        """Should block if too fast."""
        # Set last order to just now
        tight_breaker._last_order_time = datetime.now(timezone.utc)

        result = tight_breaker.check_trade(
            ticker="TEST",
            contracts=5,
            risk_cents=250,
        )

        assert not result.allowed
        assert result.block_reason == BlockReason.RATE_LIMIT

    def test_rate_limit_clears_old_timestamps(self, breaker):
        """Old timestamps should be cleared."""
        # Add old timestamps
        old_time = datetime.now(timezone.utc) - timedelta(minutes=5)
        breaker._order_timestamps = [old_time] * 50

        result = breaker.check_trade(
            ticker="TEST",
            contracts=25,
            risk_cents=1200,
        )

        # Should pass because old timestamps are cleared
        assert result.allowed


# =============================================================================
# CONCENTRATION LIMIT TESTS
# =============================================================================

class TestConcentrationLimits:
    """Test position concentration limits."""

    def test_event_concentration_limit(self, tight_breaker):
        """Should block when event position exceeded."""
        # Set existing position
        tight_breaker._positions_by_event["event_123"] = 18

        result = tight_breaker.check_trade(
            ticker="TEST",
            contracts=5,  # Would exceed 20 limit
            risk_cents=250,
            event_id="event_123",
        )

        assert not result.allowed
        assert result.block_reason == BlockReason.EVENT_CONCENTRATION

    def test_sport_concentration_limit(self, tight_breaker):
        """Should block when sport exposure exceeded."""
        # Set existing exposure (10% of $1000 = $100)
        tight_breaker._exposure_by_sport["nfl"] = 9500  # $95

        result = tight_breaker.check_trade(
            ticker="TEST",
            contracts=5,
            risk_cents=1000,  # Would push to $105, exceeding $100
            sport="nfl",
        )

        assert not result.allowed
        assert result.block_reason == BlockReason.SPORT_CONCENTRATION


# =============================================================================
# BALANCE CHECK TESTS
# =============================================================================

class TestBalanceChecks:
    """Test balance validation."""

    def test_insufficient_balance(self, breaker):
        """Should block when balance too low."""
        result = breaker.check_trade(
            ticker="TEST",
            contracts=25,
            risk_cents=1200,
            balance_cents=50,  # Below minimum
        )

        assert not result.allowed
        assert result.block_reason == BlockReason.INSUFFICIENT_BALANCE

    def test_balance_less_than_risk(self, breaker):
        """Should block when balance less than risk."""
        result = breaker.check_trade(
            ticker="TEST",
            contracts=25,
            risk_cents=1200,
            balance_cents=1000,  # Less than risk
        )

        assert not result.allowed
        assert result.block_reason == BlockReason.INSUFFICIENT_BALANCE


# =============================================================================
# MANUAL CONTROL TESTS
# =============================================================================

class TestManualControls:
    """Test manual halt/resume."""

    def test_halt(self, breaker):
        """halt() should stop all trading."""
        breaker.halt("Emergency stop")

        assert breaker.is_halted
        assert breaker.is_open
        assert breaker._manual_halt
        assert breaker._halt_reason == "Emergency stop"

    def test_halt_blocks_trades(self, breaker):
        """Halted breaker should block all trades."""
        breaker.halt("Test")

        result = breaker.check_trade(
            ticker="TEST",
            contracts=5,
            risk_cents=250,
        )

        assert not result.allowed
        assert result.block_reason == BlockReason.MANUAL_HALT

    def test_resume(self, breaker):
        """resume() should allow trading again."""
        breaker.halt("Test")
        success = breaker.resume()

        assert success
        assert breaker.is_closed
        assert not breaker._manual_halt

    def test_resume_fails_if_automatic_trip(self, tight_breaker):
        """resume() should fail if auto-trip still active."""
        # Trip via consecutive losses
        tight_breaker._daily_stats.consecutive_losses = 5

        # Manual halt
        tight_breaker.halt("Test")

        # Try to resume - should fail due to consecutive losses
        success = tight_breaker.resume()

        # Still open due to consecutive losses
        assert tight_breaker._state == BreakerState.OPEN

    def test_reset_daily_stats(self, breaker):
        """reset_daily_stats should clear counters."""
        breaker._daily_stats.trades_count = 10
        breaker._daily_stats.realized_pnl_cents = -500
        breaker._positions_by_event["test"] = 100

        breaker.reset_daily_stats()

        assert breaker._daily_stats.trades_count == 0
        assert breaker._daily_stats.realized_pnl_cents == 0
        assert len(breaker._positions_by_event) == 0


# =============================================================================
# RECORDING TESTS
# =============================================================================

class TestRecording:
    """Test trade and settlement recording."""

    def test_record_trade_success(self, breaker, sample_result):
        """record_trade should update stats for success."""
        breaker.record_trade(sample_result, event_id="event_123", sport="nfl")

        assert breaker._daily_stats.trades_count == 1
        assert breaker._daily_stats.contracts_traded == 25
        assert breaker._positions_by_event.get("event_123") == 25
        assert breaker._positions_by_sport.get("nfl") == 25

    def test_record_trade_failed(self, breaker, sample_order):
        """record_trade should track rejections."""
        sample_order.mark_rejected(message="Test rejection")
        result = ExecutionResult.from_order(sample_order)

        breaker.record_trade(result)

        assert breaker._daily_stats.orders_rejected == 1
        assert breaker._daily_stats.trades_count == 0

    def test_record_settlement_clears_position(self, breaker):
        """record_settlement should clear event position."""
        breaker._positions_by_event["event_123"] = 25

        breaker.record_settlement(pnl_cents=500, event_id="event_123")

        assert "event_123" not in breaker._positions_by_event

    def test_record_order_attempt(self, breaker):
        """record_order_attempt should track for rate limiting."""
        initial_count = len(breaker._order_timestamps)

        breaker.record_order_attempt()

        assert len(breaker._order_timestamps) == initial_count + 1
        assert breaker._last_order_time is not None


# =============================================================================
# STATUS AND CAPACITY TESTS
# =============================================================================

class TestStatusAndCapacity:
    """Test status and capacity reporting."""

    def test_get_status(self, breaker):
        """get_status should return complete status."""
        status = breaker.get_status()

        assert "state" in status
        assert "is_halted" in status
        assert "limits" in status
        assert "daily_stats" in status
        assert "bankroll_dollars" in status

    def test_get_remaining_capacity(self, breaker):
        """get_remaining_capacity should calculate correctly."""
        breaker._daily_stats.trades_count = 10
        breaker._daily_stats.contracts_traded = 1000

        capacity = breaker.get_remaining_capacity()

        assert capacity["remaining_trades"] == 40  # 50 - 10
        assert capacity["remaining_volume_contracts"] == 4000  # 5000 - 1000
        assert capacity["can_trade"] is True

    def test_get_remaining_capacity_halted(self, breaker):
        """Halted breaker should show can_trade=False."""
        breaker.halt("Test")

        capacity = breaker.get_remaining_capacity()

        assert capacity["can_trade"] is False


# =============================================================================
# DAY ROLLOVER TESTS
# =============================================================================

class TestDayRollover:
    """Test day rollover behavior."""

    def test_day_rollover_resets_stats(self, breaker):
        """New day should reset stats."""
        # Set yesterday's date
        breaker._daily_stats.date = "2020-01-01"
        breaker._daily_stats.trades_count = 50
        breaker._daily_stats.realized_pnl_cents = -500

        # Check trade triggers rollover
        breaker.check_trade(ticker="TEST", contracts=5, risk_cents=250)

        # Stats should be reset
        assert breaker._daily_stats.date == datetime.now(timezone.utc).strftime("%Y-%m-%d")
        assert breaker._daily_stats.trades_count == 0

    def test_day_rollover_resumes_breaker(self, breaker):
        """New day should auto-resume (unless manual halt)."""
        breaker._daily_stats.date = "2020-01-01"
        breaker._state = BreakerState.OPEN

        breaker.check_trade(ticker="TEST", contracts=5, risk_cents=250)

        assert breaker.is_closed

    def test_day_rollover_preserves_manual_halt(self, breaker):
        """Manual halt should persist across day rollover."""
        breaker.halt("End of day")
        breaker._daily_stats.date = "2020-01-01"

        breaker.check_trade(ticker="TEST", contracts=5, risk_cents=250)

        # Still halted
        assert breaker.is_halted
        assert breaker._manual_halt


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests with execution models."""

    def test_full_trade_flow(self, breaker):
        """Test complete flow: check -> trade -> record -> settle."""
        # 1. Check trade
        check = breaker.check_trade(
            ticker="KXNFL-26JAN11-BUF",
            contracts=25,
            risk_cents=1200,
            sport="nfl",
            event_id="event_123",
        )
        assert check.allowed

        # 2. Create and execute order
        order = ExecutionOrder(
            ticker="KXNFL-26JAN11-BUF",
            contracts=25,
            limit_price=48,
        )
        order.mark_submitted()
        fill = order.record_fill(contracts=25, price=48)
        result = ExecutionResult.from_order(order, fills=[fill])

        # 3. Record trade
        breaker.record_trade(result, event_id="event_123", sport="nfl")

        assert breaker._daily_stats.trades_count == 1
        assert breaker._positions_by_event.get("event_123") == 25

        # 4. Settle position (win)
        breaker.record_settlement(pnl_cents=1300, event_id="event_123")

        assert breaker._daily_stats.realized_pnl_cents == 1300
        assert "event_123" not in breaker._positions_by_event

    def test_multiple_trades_accumulate(self, breaker):
        """Multiple trades should accumulate in stats."""
        for i in range(5):
            order = ExecutionOrder(contracts=10, limit_price=50)
            order.mark_submitted()
            fill = order.record_fill(contracts=10, price=50)
            result = ExecutionResult.from_order(order, fills=[fill])

            breaker.record_trade(result)

        assert breaker._daily_stats.trades_count == 5
        assert breaker._daily_stats.contracts_traded == 50

    def test_breaker_trips_on_losses(self, tight_breaker):
        """Breaker should trip after consecutive losses."""
        # Lose twice (the limit)
        tight_breaker.record_settlement(pnl_cents=-100)
        tight_breaker.record_settlement(pnl_cents=-100)

        # Breaker should be open
        assert tight_breaker.is_open

        # Trades should be blocked
        result = tight_breaker.check_trade(
            ticker="TEST",
            contracts=5,
            risk_cents=250,
        )
        assert not result.allowed


# =============================================================================
# THREAD SAFETY TESTS
# =============================================================================

class TestThreadSafety:
    """Test thread safety."""

    def test_concurrent_checks(self, breaker):
        """Concurrent checks should be thread-safe."""
        import threading

        results = []

        def check_trade():
            result = breaker.check_trade(
                ticker="TEST",
                contracts=25,
                risk_cents=1200,
            )
            results.append(result.allowed)

        threads = [threading.Thread(target=check_trade) for _ in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should have completed
        assert len(results) == 10

    def test_concurrent_recording(self, breaker):
        """Concurrent recording should be thread-safe."""
        import threading

        def record_settlement():
            breaker.record_settlement(pnl_cents=100)

        threads = [threading.Thread(target=record_settlement) for _ in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Total should be correct
        assert breaker._daily_stats.realized_pnl_cents == 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
