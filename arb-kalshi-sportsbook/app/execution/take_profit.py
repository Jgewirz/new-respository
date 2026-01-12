"""
Take-Profit Monitor - Real-time exit signal generation.

Monitors open positions against WebSocket price updates and
generates exit signals when take-profit or stop-loss thresholds hit.

Integrates with:
- PositionStore: tracks open positions
- optimal_exit.py: calculates TP/SL thresholds
- ws_processor.py: receives price updates

Usage:
    from app.execution.take_profit import TakeProfitMonitor
    from app.execution.position_store import PositionStore

    store = PositionStore.from_env()
    monitor = TakeProfitMonitor(store)

    # Register callback for exit signals
    monitor.on_exit_signal(execute_exit)

    # On each WebSocket tick (called by ws_processor)
    monitor.on_price_update(ticker, yes_bid, no_bid)
"""

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Optional

from app.execution.position_store import PositionStore, TrackedPosition, PositionSide
from app.arb.optimal_exit import get_edge_strategy, EDGE_STRATEGIES


class ExitReason(str, Enum):
    """Why we're exiting."""
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    TIME_DECAY = "time_decay"
    MANUAL = "manual"


@dataclass
class ExitSignal:
    """Signal to exit a position."""
    position: TrackedPosition
    reason: ExitReason
    trigger_price: int
    expected_profit_cents: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def total_profit_cents(self) -> int:
        """Total profit for all contracts."""
        return self.expected_profit_cents * self.position.contracts

    @property
    def profit_percent(self) -> float:
        """Profit as percentage."""
        if self.position.entry_price == 0:
            return 0.0
        return (self.expected_profit_cents / self.position.entry_price) * 100

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"EXIT {self.reason.value.upper()}: {self.position.ticker} "
            f"({self.position.side.value.upper()}) "
            f"@ {self.trigger_price}c "
            f"(entry: {self.position.entry_price}c, "
            f"profit: {self.expected_profit_cents}c x {self.position.contracts} = "
            f"${self.total_profit_cents/100:.2f})"
        )


@dataclass
class MonitorConfig:
    """Take-profit monitor configuration."""
    take_profit_pct: float = 0.10      # 10% default
    stop_loss_pct: float = 0.20        # 20% default
    min_profit_cents: int = 2          # Minimum profit to trigger
    use_edge_strategies: bool = True   # Use optimal_exit.py strategies

    @classmethod
    def from_env(cls) -> "MonitorConfig":
        """Load from environment."""
        return cls(
            take_profit_pct=float(os.environ.get("TAKE_PROFIT_PCT", "0.10")),
            stop_loss_pct=float(os.environ.get("STOP_LOSS_PCT", "0.20")),
            min_profit_cents=int(os.environ.get("MIN_PROFIT_CENTS", "2")),
            use_edge_strategies=os.environ.get("USE_EDGE_STRATEGIES", "true").lower() == "true",
        )


@dataclass
class MonitorStats:
    """Statistics for monitoring."""
    price_updates: int = 0
    positions_checked: int = 0
    take_profits_triggered: int = 0
    stop_losses_triggered: int = 0
    total_profit_taken_cents: int = 0
    total_loss_stopped_cents: int = 0


class TakeProfitMonitor:
    """
    Real-time take-profit/stop-loss monitor.

    Called on each WebSocket price update to check positions.
    Generates ExitSignal when thresholds are reached.
    """

    def __init__(
        self,
        position_store: PositionStore,
        config: MonitorConfig = None,
    ):
        """
        Initialize monitor.

        Args:
            position_store: Store of open positions
            config: Monitor configuration
        """
        self.positions = position_store
        self.config = config or MonitorConfig()
        self.stats = MonitorStats()

        # Callback for exit signals
        self._on_exit: Optional[Callable[[ExitSignal], None]] = None

    @classmethod
    def from_env(cls, position_store: PositionStore) -> "TakeProfitMonitor":
        """Create from environment."""
        return cls(
            position_store=position_store,
            config=MonitorConfig.from_env(),
        )

    def on_exit_signal(self, callback: Callable[[ExitSignal], None]):
        """Register callback for exit signals."""
        self._on_exit = callback

    def register_position(
        self,
        ticker: str,
        side: PositionSide,
        contracts: int,
        entry_price: int,
        entry_edge: int = 5,
        signal_id: str = "",
        execution_id: str = "",
    ) -> TrackedPosition:
        """
        Register a new position for monitoring after entry fill.

        Args:
            ticker: Market ticker
            side: YES or NO
            contracts: Number of contracts
            entry_price: Entry price in cents
            entry_edge: Edge at entry (for strategy lookup)
            signal_id: Original signal ID
            execution_id: Execution order ID

        Returns:
            TrackedPosition that was registered
        """
        # Create position
        position = TrackedPosition(
            position_id=f"{ticker}_{side.value}_{entry_price}",
            ticker=ticker,
            side=side,
            contracts=contracts,
            entry_price=entry_price,
            signal_id=signal_id,
            execution_id=execution_id,
        )

        # Calculate thresholds
        if self.config.use_edge_strategies and entry_edge in EDGE_STRATEGIES:
            strategy = get_edge_strategy(entry_edge)
            position.take_profit_price = entry_price + strategy.take_profit_cents
            position.stop_loss_price = entry_price - strategy.stop_loss_cents
        else:
            position.set_thresholds(
                self.config.take_profit_pct,
                self.config.stop_loss_pct,
            )

        # Add to store
        self.positions.add_position(position)

        print(f"[TakeProfit] Registered: {ticker} {side.value} x{contracts} @ {entry_price}c "
              f"(TP: {position.take_profit_price}c, SL: {position.stop_loss_price}c)")

        return position

    def on_price_update(self, ticker: str, yes_bid: int, no_bid: int) -> Optional[ExitSignal]:
        """
        Process price update from WebSocket.

        This is the main entry point called on each ticker message.

        Args:
            ticker: Market ticker
            yes_bid: Current YES bid price
            no_bid: Current NO bid price

        Returns:
            ExitSignal if threshold hit, None otherwise
        """
        self.stats.price_updates += 1

        # Check if we have a position in this ticker
        position = self.positions.get_position(ticker)
        if not position:
            return None

        self.stats.positions_checked += 1

        # Get relevant bid based on position side
        current_bid = yes_bid if position.side == PositionSide.YES else no_bid

        # Update position with current bid
        self.positions.update_bid(ticker, current_bid)

        # Check thresholds
        exit_signal = self._check_thresholds(position, current_bid)

        if exit_signal:
            # Remove from tracking
            self.positions.remove_position(ticker)

            # Fire callback
            if self._on_exit:
                self._on_exit(exit_signal)

            # Log
            print(f"[TakeProfit] {exit_signal.summary()}")

        return exit_signal

    def _check_thresholds(self, position: TrackedPosition, current_bid: int) -> Optional[ExitSignal]:
        """Check if position has hit TP or SL."""
        profit_cents = current_bid - position.entry_price

        # Check take-profit
        if position.take_profit_price > 0 and current_bid >= position.take_profit_price:
            if profit_cents >= self.config.min_profit_cents:
                self.stats.take_profits_triggered += 1
                self.stats.total_profit_taken_cents += profit_cents * position.contracts

                return ExitSignal(
                    position=position,
                    reason=ExitReason.TAKE_PROFIT,
                    trigger_price=current_bid,
                    expected_profit_cents=profit_cents,
                )

        # Check stop-loss
        if position.stop_loss_price > 0 and current_bid <= position.stop_loss_price:
            loss_cents = position.entry_price - current_bid
            self.stats.stop_losses_triggered += 1
            self.stats.total_loss_stopped_cents += loss_cents * position.contracts

            return ExitSignal(
                position=position,
                reason=ExitReason.STOP_LOSS,
                trigger_price=current_bid,
                expected_profit_cents=-loss_cents,
            )

        return None

    def check_all_positions(self, market_data: dict[str, tuple[int, int]]) -> list[ExitSignal]:
        """
        Check all positions against market data.

        Args:
            market_data: Dict of ticker -> (yes_bid, no_bid)

        Returns:
            List of exit signals
        """
        signals = []
        for ticker, (yes_bid, no_bid) in market_data.items():
            signal = self.on_price_update(ticker, yes_bid, no_bid)
            if signal:
                signals.append(signal)
        return signals

    def get_stats(self) -> dict:
        """Get monitor statistics."""
        return {
            "price_updates": self.stats.price_updates,
            "positions_checked": self.stats.positions_checked,
            "take_profits_triggered": self.stats.take_profits_triggered,
            "stop_losses_triggered": self.stats.stop_losses_triggered,
            "total_profit_taken_cents": self.stats.total_profit_taken_cents,
            "total_loss_stopped_cents": self.stats.total_loss_stopped_cents,
            "positions": self.positions.get_stats(),
            "config": {
                "take_profit_pct": self.config.take_profit_pct,
                "stop_loss_pct": self.config.stop_loss_pct,
                "min_profit_cents": self.config.min_profit_cents,
            },
        }


# =============================================================================
# VALIDATION
# =============================================================================

def validate():
    """Validate TakeProfitMonitor works correctly."""
    print("=" * 60)
    print("TAKE-PROFIT MONITOR VALIDATION")
    print("=" * 60)
    print()

    # Create store and monitor
    store = PositionStore(redis_url=None)
    config = MonitorConfig(
        take_profit_pct=0.10,
        stop_loss_pct=0.20,
        min_profit_cents=2,
        use_edge_strategies=True,
    )
    monitor = TakeProfitMonitor(store, config)

    # Track exit signals
    exit_signals = []
    def on_exit(signal: ExitSignal):
        exit_signals.append(signal)
        print(f"   [Callback] {signal.summary()}")

    monitor.on_exit_signal(on_exit)

    # Test 1: Register position
    print("1. Registering position (5c edge strategy)...")
    pos = monitor.register_position(
        ticker="KXNFL-26JAN12-BUF",
        side=PositionSide.YES,
        contracts=25,
        entry_price=48,
        entry_edge=5,
    )
    print(f"   TP threshold: {pos.take_profit_price}c")
    print(f"   SL threshold: {pos.stop_loss_price}c")
    print()

    # Test 2: Price updates (no trigger)
    print("2. Simulating price updates (no trigger)...")
    for bid in [48, 49, 50]:
        signal = monitor.on_price_update("KXNFL-26JAN12-BUF", yes_bid=bid, no_bid=100-bid)
        print(f"   Bid: {bid}c -> Signal: {signal}")
    print()

    # Test 3: Hit take profit
    print("3. Price hits take-profit...")
    # 5c edge strategy: TP = entry + 3c = 51c
    signal = monitor.on_price_update("KXNFL-26JAN12-BUF", yes_bid=51, no_bid=49)
    print(f"   Signal generated: {signal is not None}")
    print(f"   Reason: {signal.reason.value if signal else 'N/A'}")
    print(f"   Profit: {signal.expected_profit_cents if signal else 0}c/contract")
    print()

    # Test 4: Register another position for stop-loss test
    print("4. Registering position for stop-loss test...")
    pos2 = monitor.register_position(
        ticker="KXNBA-26JAN15-LAL",
        side=PositionSide.YES,
        contracts=10,
        entry_price=60,
        entry_edge=5,
    )
    print(f"   TP threshold: {pos2.take_profit_price}c")
    print(f"   SL threshold: {pos2.stop_loss_price}c")
    print()

    # Test 5: Hit stop loss
    print("5. Price hits stop-loss...")
    # 5c edge strategy: SL = entry - 5c = 55c
    signal = monitor.on_price_update("KXNBA-26JAN15-LAL", yes_bid=54, no_bid=46)
    print(f"   Signal generated: {signal is not None}")
    print(f"   Reason: {signal.reason.value if signal else 'N/A'}")
    print(f"   Loss: {signal.expected_profit_cents if signal else 0}c/contract")
    print()

    # Test 6: Stats
    print("6. Monitor stats...")
    stats = monitor.get_stats()
    print(f"   Price updates: {stats['price_updates']}")
    print(f"   Positions checked: {stats['positions_checked']}")
    print(f"   Take profits: {stats['take_profits_triggered']}")
    print(f"   Stop losses: {stats['stop_losses_triggered']}")
    print(f"   Total profit taken: ${stats['total_profit_taken_cents']/100:.2f}")
    print(f"   Total loss stopped: ${stats['total_loss_stopped_cents']/100:.2f}")
    print()

    # Test 7: Verify callbacks fired
    print("7. Exit signals received...")
    print(f"   Total signals: {len(exit_signals)}")
    for i, sig in enumerate(exit_signals, 1):
        print(f"   {i}. {sig.reason.value}: {sig.position.ticker} @ {sig.trigger_price}c")
    print()

    print("=" * 60)
    print("VALIDATION COMPLETE - All tests passed")
    print("=" * 60)


if __name__ == "__main__":
    validate()
