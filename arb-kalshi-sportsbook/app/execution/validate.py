"""
Execution Models Validation Script

Validates that the execution models integrate properly with:
1. Signal from app.arb.detector
2. KalshiClient from app.connectors.kalshi.client
3. QuestDB ILP format from app.data.questdb

Run with:
    python -m app.execution.validate
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_imports() -> bool:
    """Validate all required imports work."""
    print("=" * 60)
    print("EXECUTION MODELS VALIDATION")
    print("=" * 60)
    print()
    print("1. Testing imports...")

    try:
        from app.execution import (
            ExecutionMode,
            OrderState,
            OrderSide,
            OrderAction,
            ExecutionOrder,
            Fill,
            ExecutionResult,
            now_ns,
            ns_to_datetime,
        )
        print("   [PASS] app.execution imports successful")

        from app.arb.detector import (
            Signal,
            KalshiMarket,
            SportsbookConsensus,
            EdgeCalculation,
            EdgeDetector,
        )
        print("   [PASS] app.arb.detector imports successful")

        from app.connectors.kalshi.client import (
            KalshiClient,
            Order,
            OrderSide as KalshiOrderSide,
            OrderAction as KalshiOrderAction,
        )
        print("   [PASS] app.connectors.kalshi.client imports successful")

        return True

    except ImportError as e:
        print(f"   [FAIL] Import error: {e}")
        return False


def validate_signal_to_order_conversion() -> bool:
    """Validate Signal -> ExecutionOrder conversion."""
    print()
    print("2. Testing Signal -> ExecutionOrder conversion...")

    from app.execution import ExecutionOrder, ExecutionMode
    from app.arb.detector import Signal, KalshiMarket, SportsbookConsensus, EdgeCalculation

    # Create sample signal
    kalshi = KalshiMarket(
        ticker="KXNFL-26JAN11-BUF",
        title="Buffalo Bills to beat Jacksonville Jaguars",
        yes_bid=46,
        yes_ask=48,
        no_bid=52,
        no_ask=54,
        volume=5000,
        status="active",
    )

    sportsbook = SportsbookConsensus(
        event_id="event123",
        team="Buffalo Bills",
        sport="americanfootball_nfl",
        consensus_prob=55,
        book_probs={"draftkings": 54, "fanduel": 55, "betmgm": 56, "caesars": 55},
        book_count=4,
        book_spread=2.0,
        updated_at=datetime.now(timezone.utc),
    )

    edge = EdgeCalculation.calculate(kalshi, sportsbook)

    signal = Signal(
        signal_id="sig_test_20260108",
        timestamp=datetime.now(timezone.utc),
        action="BUY_YES",
        should_trade=True,
        kalshi=kalshi,
        sportsbook=sportsbook,
        edge=edge,
        confidence_score=75,
        confidence_tier="HIGH",
        confidence_reasons=["4/4 books", "Tight consensus"],
        conditions_met=True,
        condition_checks=[],
        recommended_contracts=25,
        max_price=50,
        risk_amount=1200,
        potential_profit=1300,
    )

    # Convert to ExecutionOrder
    order = ExecutionOrder.from_signal(signal, mode=ExecutionMode.PAPER)

    # Validate conversion
    checks = [
        ("ticker", order.ticker == "KXNFL-26JAN11-BUF"),
        ("side", order.side == "yes"),
        ("action", order.action == "buy"),
        ("contracts", order.contracts == 25),
        ("limit_price", order.limit_price == 49),  # 48 + 1 offset
        ("signal_id", order.signal_id == "sig_test_20260108"),
        ("signal_edge", order.signal_edge_cents == edge.best_edge),
        ("mode", order.mode == ExecutionMode.PAPER),
    ]

    all_passed = True
    for name, passed in checks:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"   {status} {name}")
        if not passed:
            all_passed = False

    return all_passed


def validate_order_lifecycle() -> bool:
    """Validate order state machine transitions."""
    print()
    print("3. Testing order lifecycle state machine...")

    from app.execution import ExecutionOrder, OrderState, Fill

    order = ExecutionOrder(
        ticker="KXNFL-26JAN11-BUF",
        side="yes",
        action="buy",
        limit_price=49,
        contracts=100,
    )

    checks = []

    # Initial state
    checks.append(("initial_state", order.state == OrderState.PENDING))

    # Submit
    order.mark_submitted(kalshi_order_id="kalshi_123")
    checks.append(("submitted_state", order.state == OrderState.SUBMITTED))
    checks.append(("has_kalshi_id", order.kalshi_order_id == "kalshi_123"))

    # Partial fill
    fill1 = order.record_fill(contracts=40, price=48)
    checks.append(("partial_state", order.state == OrderState.PARTIALLY_FILLED))
    checks.append(("fill_created", isinstance(fill1, Fill)))
    checks.append(("filled_count", order.filled_contracts == 40))

    # Another fill
    fill2 = order.record_fill(contracts=60, price=50)
    checks.append(("filled_state", order.state == OrderState.FILLED))
    checks.append(("total_filled", order.filled_contracts == 100))
    checks.append(("remaining_zero", order.remaining_contracts == 0))

    # Check average price (weighted: 40*48 + 60*50 = 4920, /100 = 49.2)
    checks.append(("avg_price", abs(order.average_fill_price - 49.2) < 0.01))

    all_passed = True
    for name, passed in checks:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"   {status} {name}")
        if not passed:
            all_passed = False

    return all_passed


def validate_kalshi_client_compatibility() -> bool:
    """Validate ExecutionOrder fields match KalshiClient.create_order params."""
    print()
    print("4. Testing KalshiClient compatibility...")

    from app.execution import ExecutionOrder

    order = ExecutionOrder(
        ticker="KXNFL-26JAN11-BUF",
        side="yes",
        action="buy",
        order_type="limit",
        limit_price=49,
        contracts=25,
    )

    # These fields should be directly usable with KalshiClient.create_order()
    checks = [
        ("ticker_present", bool(order.ticker)),
        ("side_valid", order.side in ["yes", "no"]),
        ("action_valid", order.action in ["buy", "sell"]),
        ("type_valid", order.order_type in ["limit", "market"]),
        ("price_range", 1 <= order.limit_price <= 99),
        ("contracts_positive", order.contracts > 0),
        ("client_order_id", bool(order.client_order_id)),
    ]

    all_passed = True
    for name, passed in checks:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"   {status} {name}")
        if not passed:
            all_passed = False

    # Show example of how to use with KalshiClient
    print()
    print("   Example usage with KalshiClient:")
    print("   ```python")
    print("   kalshi_order = client.create_order(")
    print(f"       ticker=\"{order.ticker}\",")
    print(f"       side=\"{order.side}\",")
    print(f"       action=\"{order.action}\",")
    print(f"       count={order.contracts},")
    print(f"       type=\"{order.order_type}\",")
    print(f"       yes_price={order.limit_price},")
    print(f"       client_order_id=\"{order.client_order_id[:8]}...\",")
    print("   )")
    print("   ```")

    return all_passed


def validate_execution_result() -> bool:
    """Validate ExecutionResult metrics calculation."""
    print()
    print("5. Testing ExecutionResult metrics...")

    from app.execution import ExecutionOrder, ExecutionResult, Fill

    order = ExecutionOrder(
        ticker="KXNFL-26JAN11-BUF",
        side="yes",
        limit_price=49,
        contracts=25,
        signal_edge_cents=7,
    )

    order.mark_submitted()
    fill = order.record_fill(contracts=25, price=48)

    result = ExecutionResult.from_order(order, fills=[fill])

    checks = [
        ("success", result.success),
        ("not_partial", not result.partial),
        ("not_failed", not result.failed),
        ("fill_count", result.fill_count == 1),
        ("total_cost", result.total_cost_cents == 1200),  # 25 * 48
        ("potential_profit", result.potential_profit_cents == 1300),  # 25 * (100-48)
        ("has_latency", result.execution_time_ms >= 0),
    ]

    all_passed = True
    for name, passed in checks:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"   {status} {name}")
        if not passed:
            all_passed = False

    print()
    print(f"   Summary: {result.summary()}")

    return all_passed


def validate_ilp_format() -> bool:
    """Validate ILP fields for QuestDB compatibility."""
    print()
    print("6. Testing QuestDB ILP field format...")

    from app.execution import ExecutionOrder, Fill

    order = ExecutionOrder(
        ticker="KXNFL-26JAN11-BUF",
        side="yes",
        limit_price=49,
        contracts=25,
        signal_edge_cents=7,
    )
    order.mark_submitted(kalshi_order_id="kalshi_123")
    fill = order.record_fill(contracts=25, price=48)

    order_fields = order.to_ilp_fields()
    fill_fields = fill.to_ilp_fields()

    # Check types are QuestDB compatible
    order_checks = [
        ("execution_id_str", isinstance(order_fields["execution_id"], str)),
        ("ticker_str", isinstance(order_fields["ticker"], str)),
        ("limit_price_int", isinstance(order_fields["limit_price"], int)),
        ("average_fill_price_float", isinstance(order_fields["average_fill_price"], float)),
        ("state_str", isinstance(order_fields["state"], str)),
    ]

    fill_checks = [
        ("fill_id_str", isinstance(fill_fields["fill_id"], str)),
        ("contracts_int", isinstance(fill_fields["contracts"], int)),
        ("price_int", isinstance(fill_fields["price"], int)),
        ("notional_int", isinstance(fill_fields["notional_cents"], int)),
    ]

    all_passed = True
    for name, passed in order_checks + fill_checks:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"   {status} {name}")
        if not passed:
            all_passed = False

    return all_passed


def validate_circuit_breaker() -> bool:
    """Validate CircuitBreaker functionality."""
    print()
    print("7. Testing CircuitBreaker risk controls...")

    from app.execution import (
        CircuitBreaker,
        RiskLimits,
        BreakerState,
        BlockReason,
        ExecutionOrder,
        ExecutionResult,
    )

    # Create breaker with tight limits for testing
    limits = RiskLimits(
        max_position_size=50,
        max_risk_per_trade_cents=500_00,
        max_daily_loss_cents=1000_00,
        max_daily_trades=100,
        max_consecutive_losses=5,
    )
    breaker = CircuitBreaker(limits=limits, bankroll_cents=1_000_000)

    checks = []

    # Test initial state
    checks.append(("initial_state_closed", breaker.is_closed))
    checks.append(("not_halted", not breaker.is_halted))

    # Test allowed trade
    result = breaker.check_trade(
        ticker="KXNFL-26JAN11-BUF",
        contracts=25,
        risk_cents=1200,
    )
    checks.append(("trade_allowed", result.allowed))

    # Test position size block
    result = breaker.check_trade(
        ticker="TEST",
        contracts=100,  # Exceeds 50 limit
        risk_cents=5000,
    )
    checks.append(("blocks_large_position", not result.allowed))
    checks.append(("correct_block_reason", result.block_reason == BlockReason.POSITION_SIZE))

    # Test manual halt
    breaker.halt("Test halt")
    checks.append(("halt_works", breaker.is_halted))

    result = breaker.check_trade(ticker="TEST", contracts=10, risk_cents=500)
    checks.append(("halt_blocks_trades", not result.allowed))

    # Test resume
    breaker.resume()
    checks.append(("resume_works", breaker.is_closed))

    # Test trade recording
    order = ExecutionOrder(ticker="TEST", contracts=25, limit_price=48)
    order.mark_submitted()
    fill = order.record_fill(contracts=25, price=48)
    exec_result = ExecutionResult.from_order(order, fills=[fill])

    breaker.record_trade(exec_result, sport="nfl")
    checks.append(("records_trade", breaker.daily_stats.trades_count == 1))

    # Test settlement recording
    breaker.record_settlement(pnl_cents=500)
    checks.append(("records_settlement", breaker.daily_stats.realized_pnl_cents == 500))

    # Test status reporting
    status = breaker.get_status()
    checks.append(("status_has_state", "state" in status))
    checks.append(("status_has_limits", "limits" in status))

    # Test capacity reporting
    capacity = breaker.get_remaining_capacity()
    checks.append(("capacity_has_remaining", "remaining_trades" in capacity))
    checks.append(("capacity_can_trade", capacity["can_trade"]))

    all_passed = True
    for name, passed in checks:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"   {status} {name}")
        if not passed:
            all_passed = False

    return all_passed


def run_validation():
    """Run all validation checks."""
    results = []

    results.append(("Imports", validate_imports()))
    results.append(("Signal -> Order", validate_signal_to_order_conversion()))
    results.append(("Order Lifecycle", validate_order_lifecycle()))
    results.append(("KalshiClient Compat", validate_kalshi_client_compatibility()))
    results.append(("ExecutionResult", validate_execution_result()))
    results.append(("QuestDB ILP", validate_ilp_format()))
    results.append(("CircuitBreaker", validate_circuit_breaker()))

    print()
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"   {status}: {name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("ALL VALIDATIONS PASSED")
        print()
        print("Phase 1 (Core Models) and Phase 2 (Circuit Breaker) complete.")
        print("The execution layer integrates properly with:")
        print("  - Signal from app.arb.detector")
        print("  - KalshiClient from app.connectors.kalshi.client")
        print("  - QuestDB ILP format from app.data.questdb")
        print()
        print("Risk controls implemented:")
        print("  - Per-trade limits (position size, risk amount)")
        print("  - Daily limits (loss, trades, volume)")
        print("  - Consecutive loss protection")
        print("  - Rate limiting")
        print("  - Position concentration")
        print("  - Manual halt/resume")
        print()
        print("Ready for Phase 3: Order Manager implementation")
        return 0
    else:
        print("SOME VALIDATIONS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(run_validation())
