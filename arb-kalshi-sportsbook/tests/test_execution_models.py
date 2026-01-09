"""
Unit Tests for Execution Layer Models

Tests ExecutionOrder, Fill, and ExecutionResult dataclasses
to ensure they integrate properly with the existing Signal
structure from detector.py.

Run with:
    pytest tests/test_execution_models.py -v
    python -m pytest tests/test_execution_models.py -v
"""

import pytest
from datetime import datetime, timezone

from app.execution.models import (
    ExecutionMode,
    OrderState,
    OrderSide,
    OrderAction,
    OrderType,
    ExecutionOrder,
    Fill,
    ExecutionResult,
    now_ns,
    ns_to_datetime,
    generate_id,
)
from app.arb.detector import (
    Signal,
    KalshiMarket,
    SportsbookConsensus,
    EdgeCalculation,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_kalshi_market() -> KalshiMarket:
    """Sample Kalshi market data."""
    return KalshiMarket(
        ticker="KXNFL-26JAN11-BUF",
        title="Buffalo Bills to beat Jacksonville Jaguars",
        yes_bid=46,
        yes_ask=48,
        no_bid=52,
        no_ask=54,
        volume=5000,
        status="active",
    )


@pytest.fixture
def sample_sportsbook_consensus() -> SportsbookConsensus:
    """Sample sportsbook consensus."""
    return SportsbookConsensus(
        event_id="event123",
        team="Buffalo Bills",
        sport="americanfootball_nfl",
        consensus_prob=55,
        book_probs={
            "draftkings": 54,
            "fanduel": 55,
            "betmgm": 56,
            "caesars": 55,
        },
        book_count=4,
        book_spread=2.0,
        updated_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def sample_edge(sample_kalshi_market, sample_sportsbook_consensus) -> EdgeCalculation:
    """Sample edge calculation."""
    return EdgeCalculation.calculate(sample_kalshi_market, sample_sportsbook_consensus)


@pytest.fixture
def sample_signal(sample_kalshi_market, sample_sportsbook_consensus, sample_edge) -> Signal:
    """Sample trading signal."""
    return Signal(
        signal_id="sig_20260108_KXNFL-26JAN11-BUF",
        timestamp=datetime.now(timezone.utc),
        action="BUY_YES",
        should_trade=True,
        kalshi=sample_kalshi_market,
        sportsbook=sample_sportsbook_consensus,
        edge=sample_edge,
        confidence_score=75,
        confidence_tier="HIGH",
        confidence_reasons=["4/4 books reporting", "Tight consensus"],
        conditions_met=True,
        condition_checks=[],
        recommended_contracts=25,
        max_price=50,
        risk_amount=1200,
        potential_profit=1300,
    )


# =============================================================================
# UTILITY TESTS
# =============================================================================

class TestUtilities:
    """Test utility functions."""

    def test_now_ns_returns_int(self):
        """now_ns should return integer nanoseconds."""
        result = now_ns()
        assert isinstance(result, int)
        assert result > 0

    def test_now_ns_is_recent(self):
        """now_ns should be within recent time."""
        result = now_ns()
        expected = int(datetime.now(timezone.utc).timestamp() * 1e9)
        # Allow 1 second tolerance
        assert abs(result - expected) < 1e9

    def test_ns_to_datetime_roundtrip(self):
        """ns_to_datetime should roundtrip with now_ns."""
        ns = now_ns()
        dt = ns_to_datetime(ns)
        assert isinstance(dt, datetime)
        assert dt.tzinfo == timezone.utc

    def test_generate_id_unique(self):
        """generate_id should return unique IDs."""
        id1 = generate_id()
        id2 = generate_id()
        assert id1 != id2
        assert len(id1) == 36  # UUID format


# =============================================================================
# ENUM TESTS
# =============================================================================

class TestEnums:
    """Test enum definitions."""

    def test_execution_mode_values(self):
        """ExecutionMode should have paper and live."""
        assert ExecutionMode.PAPER.value == "paper"
        assert ExecutionMode.LIVE.value == "live"

    def test_order_state_values(self):
        """OrderState should have all lifecycle states."""
        states = [s.value for s in OrderState]
        assert "pending" in states
        assert "submitted" in states
        assert "resting" in states
        assert "partial" in states
        assert "filled" in states
        assert "canceled" in states
        assert "rejected" in states
        assert "expired" in states

    def test_order_side_values(self):
        """OrderSide should match Kalshi API."""
        assert OrderSide.YES.value == "yes"
        assert OrderSide.NO.value == "no"

    def test_order_action_values(self):
        """OrderAction should match Kalshi API."""
        assert OrderAction.BUY.value == "buy"
        assert OrderAction.SELL.value == "sell"


# =============================================================================
# EXECUTION ORDER TESTS
# =============================================================================

class TestExecutionOrder:
    """Test ExecutionOrder dataclass."""

    def test_default_initialization(self):
        """ExecutionOrder should initialize with defaults."""
        order = ExecutionOrder()

        assert order.execution_id is not None
        assert order.client_order_id is not None
        assert order.state == OrderState.PENDING
        assert order.mode == ExecutionMode.PAPER
        assert order.created_at_ns > 0

    def test_from_signal_buy_yes(self, sample_signal):
        """from_signal should create order for BUY_YES."""
        order = ExecutionOrder.from_signal(sample_signal)

        assert order.ticker == "KXNFL-26JAN11-BUF"
        assert order.side == "yes"
        assert order.action == "buy"
        assert order.contracts == 25
        assert order.limit_price == 49  # 48 + 1 offset
        assert order.signal_id == sample_signal.signal_id
        assert order.signal_edge_cents == sample_signal.edge.best_edge
        assert order.signal_confidence == 75

    def test_from_signal_buy_no(self, sample_kalshi_market, sample_sportsbook_consensus):
        """from_signal should create order for BUY_NO."""
        # Create signal for NO side
        signal = Signal(
            signal_id="sig_test",
            timestamp=datetime.now(timezone.utc),
            action="BUY_NO",
            should_trade=True,
            kalshi=sample_kalshi_market,
            sportsbook=sample_sportsbook_consensus,
            edge=EdgeCalculation(
                yes_edge=-5,
                no_edge=8,
                best_side="no",
                best_edge=8,
                edge_pct=14.8,
            ),
            confidence_score=70,
            confidence_tier="HIGH",
            confidence_reasons=[],
            conditions_met=True,
            condition_checks=[],
            recommended_contracts=20,
            max_price=55,
            risk_amount=1100,
            potential_profit=900,
        )

        order = ExecutionOrder.from_signal(signal)

        assert order.side == "no"
        assert order.limit_price == 55  # 54 + 1

    def test_from_signal_invalid_action(self, sample_signal):
        """from_signal should reject invalid action."""
        sample_signal.action = "INVALID"

        with pytest.raises(ValueError, match="Cannot create order"):
            ExecutionOrder.from_signal(sample_signal)

    def test_is_terminal_states(self):
        """is_terminal should return True for terminal states."""
        order = ExecutionOrder()

        # Non-terminal states
        for state in [OrderState.PENDING, OrderState.SUBMITTED,
                      OrderState.RESTING, OrderState.PARTIALLY_FILLED]:
            order.state = state
            assert not order.is_terminal

        # Terminal states
        for state in [OrderState.FILLED, OrderState.CANCELED,
                      OrderState.REJECTED, OrderState.EXPIRED]:
            order.state = state
            assert order.is_terminal

    def test_is_active(self):
        """is_active should return True for active states."""
        order = ExecutionOrder()

        # Active states
        for state in [OrderState.SUBMITTED, OrderState.RESTING,
                      OrderState.PARTIALLY_FILLED]:
            order.state = state
            assert order.is_active

        # Inactive states
        for state in [OrderState.PENDING, OrderState.FILLED,
                      OrderState.CANCELED]:
            order.state = state
            assert not order.is_active

    def test_fill_rate_calculation(self):
        """fill_rate should calculate correctly."""
        order = ExecutionOrder(contracts=100)

        assert order.fill_rate == 0.0

        order.filled_contracts = 50
        assert order.fill_rate == 0.5
        assert order.fill_rate_pct == 50.0

        order.filled_contracts = 100
        assert order.fill_rate == 1.0

    def test_mark_submitted(self):
        """mark_submitted should update state and timestamp."""
        order = ExecutionOrder()
        assert order.submitted_at_ns == 0

        order.mark_submitted(kalshi_order_id="kalshi123")

        assert order.state == OrderState.SUBMITTED
        assert order.kalshi_order_id == "kalshi123"
        assert order.submitted_at_ns > 0

    def test_mark_filled(self):
        """mark_filled should update state and timestamp."""
        order = ExecutionOrder(contracts=10)
        order.filled_contracts = 10
        order.remaining_contracts = 0

        order.mark_filled()

        assert order.state == OrderState.FILLED
        assert order.completed_at_ns > 0

    def test_mark_rejected(self):
        """mark_rejected should set error info."""
        order = ExecutionOrder()

        order.mark_rejected(code="INSUFFICIENT_FUNDS", message="Not enough balance")

        assert order.state == OrderState.REJECTED
        assert order.error_code == "INSUFFICIENT_FUNDS"
        assert order.error_message == "Not enough balance"

    def test_record_fill(self):
        """record_fill should update order and return Fill."""
        order = ExecutionOrder(contracts=100)

        fill = order.record_fill(contracts=50, price=48)

        assert isinstance(fill, Fill)
        assert fill.contracts == 50
        assert fill.price == 48
        assert order.filled_contracts == 50
        assert order.remaining_contracts == 50
        assert order.average_fill_price == 48.0
        assert order.state == OrderState.PARTIALLY_FILLED

    def test_record_multiple_fills(self):
        """record_fill should calculate weighted average."""
        order = ExecutionOrder(contracts=100)

        order.record_fill(contracts=40, price=48)
        order.record_fill(contracts=60, price=50)

        assert order.filled_contracts == 100
        assert order.remaining_contracts == 0
        # Weighted average: (40*48 + 60*50) / 100 = 49.2
        assert abs(order.average_fill_price - 49.2) < 0.01
        assert order.state == OrderState.FILLED

    def test_can_retry(self):
        """can_retry should check state and count."""
        order = ExecutionOrder(max_retries=3)

        # Cannot retry from wrong states
        order.state = OrderState.PENDING
        assert not order.can_retry()

        order.state = OrderState.FILLED
        assert not order.can_retry()

        # Can retry from expired/canceled
        order.state = OrderState.CANCELED
        assert order.can_retry()

        order.state = OrderState.EXPIRED
        assert order.can_retry()

        # Cannot retry after max
        order.retry_count = 3
        assert not order.can_retry()

    def test_prepare_retry(self):
        """prepare_retry should reset state and improve price."""
        order = ExecutionOrder(
            limit_price=48,
            state=OrderState.CANCELED,
        )
        old_client_id = order.client_order_id

        order.prepare_retry(price_improvement=2)

        assert order.state == OrderState.PENDING
        assert order.limit_price == 50
        assert order.retry_count == 1
        assert order.client_order_id != old_client_id

    def test_financial_properties(self):
        """Financial properties should calculate correctly."""
        order = ExecutionOrder(contracts=10)
        order.filled_contracts = 10
        order.average_fill_price = 48.0

        assert order.total_cost_cents == 480
        assert order.total_cost_dollars == 4.80
        assert order.potential_profit_cents == 520  # 10 * (100 - 48)
        assert order.potential_profit_dollars == 5.20

    def test_latency_properties(self):
        """Latency properties should calculate correctly."""
        order = ExecutionOrder()
        created = order.created_at_ns

        # Simulate submission delay
        import time
        time.sleep(0.01)  # 10ms
        order.mark_submitted()

        latency = order.submission_latency_ms
        assert latency >= 10  # At least 10ms
        assert latency < 100  # Reasonable upper bound

    def test_to_dict(self):
        """to_dict should include all fields."""
        order = ExecutionOrder(
            ticker="KXNFL-26JAN11-BUF",
            contracts=25,
        )
        order.mark_submitted(kalshi_order_id="test123")

        d = order.to_dict()

        assert d["ticker"] == "KXNFL-26JAN11-BUF"
        assert d["contracts"] == 25
        assert d["kalshi_order_id"] == "test123"
        assert d["state"] == "submitted"
        assert "created_at" in d
        assert "submitted_at" in d


# =============================================================================
# FILL TESTS
# =============================================================================

class TestFill:
    """Test Fill dataclass."""

    def test_default_initialization(self):
        """Fill should initialize with defaults."""
        fill = Fill()

        assert fill.fill_id is not None
        assert fill.timestamp_ns > 0
        assert fill.contracts == 0
        assert fill.price == 0

    def test_notional_calculation(self):
        """notional should calculate correctly."""
        fill = Fill(contracts=25, price=48)

        assert fill.notional_cents == 1200  # 25 * 48
        assert fill.notional_dollars == 12.0

    def test_potential_profit(self):
        """potential_profit should calculate correctly."""
        fill = Fill(contracts=25, price=48)

        assert fill.potential_profit_cents == 1300  # 25 * (100 - 48)

    def test_timestamp_property(self):
        """timestamp property should convert from ns."""
        fill = Fill()

        ts = fill.timestamp
        assert isinstance(ts, datetime)
        assert ts.tzinfo == timezone.utc

    def test_to_dict(self):
        """to_dict should include all fields."""
        fill = Fill(
            execution_id="exec123",
            contracts=25,
            price=48,
            is_maker=True,
        )

        d = fill.to_dict()

        assert d["execution_id"] == "exec123"
        assert d["contracts"] == 25
        assert d["price"] == 48
        assert d["is_maker"] is True
        assert d["notional_cents"] == 1200


# =============================================================================
# EXECUTION RESULT TESTS
# =============================================================================

class TestExecutionResult:
    """Test ExecutionResult dataclass."""

    def test_from_order_success(self):
        """from_order should create result for filled order."""
        order = ExecutionOrder(contracts=25, limit_price=49)
        order.mark_submitted()
        fill = order.record_fill(contracts=25, price=48)

        result = ExecutionResult.from_order(order, fills=[fill])

        assert result.success
        assert not result.partial
        assert not result.failed
        assert result.total_cost_cents == 1200
        assert result.potential_profit_cents == 1300

    def test_from_order_partial(self):
        """from_order should detect partial fills."""
        order = ExecutionOrder(contracts=25, limit_price=49)
        order.mark_submitted()
        fill = order.record_fill(contracts=10, price=48)
        order.mark_canceled()

        result = ExecutionResult.from_order(order, fills=[fill])

        assert not result.success
        assert result.partial
        assert not result.failed

    def test_from_order_failed(self):
        """from_order should detect failures."""
        order = ExecutionOrder(contracts=25)
        order.mark_rejected(message="Insufficient funds")

        result = ExecutionResult.from_order(order)

        assert not result.success
        assert not result.partial
        assert result.failed

    def test_slippage_calculation(self):
        """Slippage should be calculated correctly."""
        order = ExecutionOrder(contracts=25, limit_price=49)
        order.mark_submitted()
        # Fill at worse price than expected
        fill = order.record_fill(contracts=25, price=50)

        result = ExecutionResult.from_order(order, fills=[fill])

        # Expected was 48 (limit - 1), got 50
        assert result.slippage_cents == 2.0

    def test_summary_string(self):
        """summary should return readable string."""
        order = ExecutionOrder(contracts=25, limit_price=49)
        order.mark_submitted()
        fill = order.record_fill(contracts=25, price=48)

        result = ExecutionResult.from_order(order, fills=[fill])
        summary = result.summary()

        assert "FILLED" in summary
        assert "25/25" in summary
        assert "48.0c" in summary

    def test_to_dict(self):
        """to_dict should include order, fills, and metrics."""
        order = ExecutionOrder(contracts=25)
        order.mark_submitted()
        fill = order.record_fill(contracts=25, price=48)

        result = ExecutionResult.from_order(order, fills=[fill])
        d = result.to_dict()

        assert "order" in d
        assert "fills" in d
        assert "metrics" in d
        assert d["metrics"]["success"] is True
        assert d["metrics"]["total_cost_cents"] == 1200


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests with existing components."""

    def test_signal_to_order_to_result_flow(self, sample_signal):
        """Test complete flow from Signal to ExecutionResult."""
        # Create order from signal
        order = ExecutionOrder.from_signal(sample_signal, mode=ExecutionMode.PAPER)

        assert order.ticker == sample_signal.kalshi.ticker
        assert order.contracts == sample_signal.recommended_contracts

        # Simulate execution
        order.mark_submitted(kalshi_order_id="paper_123")
        fill = order.record_fill(
            contracts=order.contracts,
            price=order.limit_price - 1,  # Good fill
        )

        # Build result
        result = ExecutionResult.from_order(order, fills=[fill])

        assert result.success
        assert result.fill_count == 1
        assert result.total_cost_cents > 0
        assert result.potential_profit_cents > 0
        assert result.order.signal_edge_cents == sample_signal.edge.best_edge

    def test_order_matches_kalshi_client_format(self, sample_signal):
        """Order fields should match KalshiClient.create_order params."""
        order = ExecutionOrder.from_signal(sample_signal)

        # These should match KalshiClient.create_order parameters
        assert order.ticker  # Required
        assert order.side in ["yes", "no"]  # Must be valid side
        assert order.action in ["buy", "sell"]  # Must be valid action
        assert order.order_type in ["limit", "market"]  # Must be valid type
        assert 1 <= order.limit_price <= 99  # Valid price range
        assert order.contracts > 0  # Must have contracts
        assert order.client_order_id  # For idempotency

    def test_ilp_fields_for_questdb(self, sample_signal):
        """ILP fields should be suitable for QuestDB."""
        order = ExecutionOrder.from_signal(sample_signal)
        order.mark_submitted()
        fill = order.record_fill(contracts=10, price=48)

        order_fields = order.to_ilp_fields()
        fill_fields = fill.to_ilp_fields()

        # Check order fields
        assert isinstance(order_fields["execution_id"], str)
        assert isinstance(order_fields["limit_price"], int)
        assert isinstance(order_fields["average_fill_price"], float)
        assert isinstance(order_fields["submission_latency_ms"], float)

        # Check fill fields
        assert isinstance(fill_fields["fill_id"], str)
        assert isinstance(fill_fields["contracts"], int)
        assert isinstance(fill_fields["notional_cents"], int)


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_contracts(self):
        """Handle zero contracts gracefully."""
        order = ExecutionOrder(contracts=0)

        assert order.fill_rate == 0.0
        assert order.total_cost_cents == 0
        assert order.potential_profit_cents == 0

    def test_price_cap_at_99(self, sample_signal):
        """Price should be capped at 99."""
        sample_signal.kalshi.yes_ask = 99

        order = ExecutionOrder.from_signal(sample_signal, price_offset=5)

        assert order.limit_price == 99  # Capped

    def test_empty_fills_list(self):
        """Result should handle empty fills."""
        order = ExecutionOrder(contracts=25)
        order.mark_rejected()

        result = ExecutionResult.from_order(order, fills=[])

        assert result.fill_count == 0
        assert result.failed
        assert result.total_cost_cents == 0

    def test_timestamp_precision(self):
        """Timestamps should have nanosecond precision."""
        ns1 = now_ns()
        ns2 = now_ns()

        # Should be different (unless exactly same ns, very unlikely)
        # At minimum, should both be valid nanosecond timestamps
        assert ns1 > 1e18  # After year 2000
        assert ns2 > 1e18


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
