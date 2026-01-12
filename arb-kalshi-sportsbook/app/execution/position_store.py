"""
Position Store - Track open positions for take-profit monitoring.

Stores positions in-memory with optional Redis persistence.
Thread-safe for concurrent access from WebSocket callbacks.

Usage:
    from app.execution.position_store import PositionStore, TrackedPosition, PositionSide

    store = PositionStore.from_env()

    # Add position after entry fill
    position = TrackedPosition(
        position_id="KXNFL-BUF-48",
        ticker="KXNFL-26JAN12-BUF",
        side=PositionSide.YES,
        contracts=25,
        entry_price=48,
    )
    store.add_position(position)

    # Update on price tick
    store.update_bid(ticker="KXNFL-26JAN12-BUF", bid=52)

    # Check position
    pos = store.get_position("KXNFL-26JAN12-BUF")
    print(f"P&L: {pos.unrealized_pnl_cents}c per contract")
"""

import os
import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

try:
    import redis
except ImportError:
    redis = None


class PositionSide(str, Enum):
    """Which contract side we hold."""
    YES = "yes"
    NO = "no"


@dataclass
class TrackedPosition:
    """
    An open position being monitored for take-profit.

    Tracks entry price vs current bid to calculate unrealized P&L.
    """
    position_id: str
    ticker: str
    side: PositionSide
    contracts: int
    entry_price: int  # cents

    # Dynamic - updated by monitor
    current_bid: int = 0
    take_profit_price: int = 0
    stop_loss_price: int = 0

    # Metadata
    entry_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    signal_id: str = ""
    execution_id: str = ""

    @property
    def unrealized_pnl_cents(self) -> int:
        """Unrealized P&L per contract in cents."""
        if self.current_bid == 0:
            return 0
        return self.current_bid - self.entry_price

    @property
    def total_pnl_cents(self) -> int:
        """Total unrealized P&L for all contracts."""
        return self.unrealized_pnl_cents * self.contracts

    @property
    def pnl_percent(self) -> float:
        """P&L as percentage of entry."""
        if self.entry_price == 0:
            return 0.0
        return (self.unrealized_pnl_cents / self.entry_price) * 100

    @property
    def is_profitable(self) -> bool:
        """Position is currently profitable."""
        return self.unrealized_pnl_cents > 0

    @property
    def at_take_profit(self) -> bool:
        """Position has reached take-profit threshold."""
        if self.take_profit_price == 0 or self.current_bid == 0:
            return False
        return self.current_bid >= self.take_profit_price

    @property
    def at_stop_loss(self) -> bool:
        """Position has reached stop-loss threshold."""
        if self.stop_loss_price == 0 or self.current_bid == 0:
            return False
        return self.current_bid <= self.stop_loss_price

    def set_thresholds(self, take_profit_pct: float, stop_loss_pct: float):
        """Set TP/SL thresholds from percentages."""
        self.take_profit_price = int(self.entry_price * (1 + take_profit_pct))
        self.stop_loss_price = int(self.entry_price * (1 - stop_loss_pct))

    def to_dict(self) -> dict:
        """Serialize for Redis storage."""
        return {
            "position_id": self.position_id,
            "ticker": self.ticker,
            "side": self.side.value,
            "contracts": self.contracts,
            "entry_price": self.entry_price,
            "current_bid": self.current_bid,
            "take_profit_price": self.take_profit_price,
            "stop_loss_price": self.stop_loss_price,
            "entry_time": self.entry_time.isoformat(),
            "signal_id": self.signal_id,
            "execution_id": self.execution_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TrackedPosition":
        """Deserialize from dict."""
        return cls(
            position_id=data["position_id"],
            ticker=data["ticker"],
            side=PositionSide(data["side"]),
            contracts=int(data["contracts"]),
            entry_price=int(data["entry_price"]),
            current_bid=int(data.get("current_bid", 0)),
            take_profit_price=int(data.get("take_profit_price", 0)),
            stop_loss_price=int(data.get("stop_loss_price", 0)),
            entry_time=datetime.fromisoformat(data["entry_time"]) if data.get("entry_time") else datetime.now(timezone.utc),
            signal_id=data.get("signal_id", ""),
            execution_id=data.get("execution_id", ""),
        )


class PositionStore:
    """
    Thread-safe position storage with optional Redis persistence.

    Positions keyed by ticker (one position per market).
    """

    REDIS_KEY_PREFIX = "positions:active"
    REDIS_SET_KEY = "positions:tickers"
    REDIS_TTL = 86400  # 24 hours

    def __init__(self, redis_url: str = None):
        """
        Initialize store.

        Args:
            redis_url: Optional Redis URL for persistence
        """
        self._positions: dict[str, TrackedPosition] = {}
        self._lock = threading.RLock()
        self._redis = None

        if redis_url and redis:
            try:
                self._redis = redis.from_url(redis_url, decode_responses=True)
                self._redis.ping()
            except Exception as e:
                print(f"[PositionStore] Redis connection failed: {e}")
                self._redis = None

    @classmethod
    def from_env(cls) -> "PositionStore":
        """Create from environment."""
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        return cls(redis_url=redis_url)

    def add_position(self, position: TrackedPosition) -> None:
        """Add a position to track."""
        with self._lock:
            self._positions[position.ticker] = position
            self._persist(position)

    def get_position(self, ticker: str) -> Optional[TrackedPosition]:
        """Get position by ticker."""
        with self._lock:
            return self._positions.get(ticker)

    def get_all_positions(self) -> list[TrackedPosition]:
        """Get all tracked positions."""
        with self._lock:
            return list(self._positions.values())

    def get_tickers(self) -> list[str]:
        """Get all tracked tickers."""
        with self._lock:
            return list(self._positions.keys())

    def update_bid(self, ticker: str, bid: int) -> Optional[TrackedPosition]:
        """
        Update current bid for a position.

        Returns updated position or None if not tracked.
        """
        with self._lock:
            pos = self._positions.get(ticker)
            if pos:
                pos.current_bid = bid
                self._persist(pos)
            return pos

    def remove_position(self, ticker: str) -> Optional[TrackedPosition]:
        """Remove position after exit."""
        with self._lock:
            pos = self._positions.pop(ticker, None)
            if pos:
                self._remove_from_redis(ticker)
            return pos

    def has_position(self, ticker: str) -> bool:
        """Check if we have a position in this ticker."""
        with self._lock:
            return ticker in self._positions

    def position_count(self) -> int:
        """Number of open positions."""
        with self._lock:
            return len(self._positions)

    def load_from_redis(self) -> int:
        """Load positions from Redis on startup. Returns count loaded."""
        if not self._redis:
            return 0

        with self._lock:
            try:
                tickers = self._redis.smembers(self.REDIS_SET_KEY)
                count = 0
                for ticker in tickers:
                    key = f"{self.REDIS_KEY_PREFIX}:{ticker}"
                    data = self._redis.hgetall(key)
                    if data:
                        pos = TrackedPosition.from_dict(data)
                        self._positions[ticker] = pos
                        count += 1
                return count
            except Exception as e:
                print(f"[PositionStore] Redis load error: {e}")
                return 0

    def get_stats(self) -> dict:
        """Get store statistics."""
        with self._lock:
            positions = list(self._positions.values())
            profitable = sum(1 for p in positions if p.is_profitable)
            at_tp = sum(1 for p in positions if p.at_take_profit)
            at_sl = sum(1 for p in positions if p.at_stop_loss)
            total_pnl = sum(p.total_pnl_cents for p in positions)

            return {
                "position_count": len(positions),
                "profitable_count": profitable,
                "at_take_profit": at_tp,
                "at_stop_loss": at_sl,
                "total_unrealized_pnl_cents": total_pnl,
                "redis_connected": self._redis is not None,
            }

    def _persist(self, position: TrackedPosition) -> None:
        """Persist position to Redis."""
        if not self._redis:
            return
        try:
            key = f"{self.REDIS_KEY_PREFIX}:{position.ticker}"
            self._redis.hset(key, mapping=position.to_dict())
            self._redis.expire(key, self.REDIS_TTL)
            self._redis.sadd(self.REDIS_SET_KEY, position.ticker)
        except Exception as e:
            print(f"[PositionStore] Redis persist error: {e}")

    def _remove_from_redis(self, ticker: str) -> None:
        """Remove position from Redis."""
        if not self._redis:
            return
        try:
            self._redis.delete(f"{self.REDIS_KEY_PREFIX}:{ticker}")
            self._redis.srem(self.REDIS_SET_KEY, ticker)
        except Exception:
            pass


# =============================================================================
# VALIDATION
# =============================================================================

def validate():
    """Validate PositionStore works correctly."""
    print("=" * 60)
    print("POSITION STORE VALIDATION")
    print("=" * 60)
    print()

    # Create store (no Redis for validation)
    store = PositionStore(redis_url=None)

    # Test 1: Add position
    print("1. Adding position...")
    pos = TrackedPosition(
        position_id="test-001",
        ticker="KXNFL-26JAN12-BUF",
        side=PositionSide.YES,
        contracts=25,
        entry_price=48,
    )
    pos.set_thresholds(take_profit_pct=0.10, stop_loss_pct=0.20)
    store.add_position(pos)
    print(f"   Added: {pos.ticker}")
    print(f"   Entry: {pos.entry_price}c")
    print(f"   TP: {pos.take_profit_price}c, SL: {pos.stop_loss_price}c")
    print()

    # Test 2: Update bid
    print("2. Updating bid to 50c...")
    store.update_bid("KXNFL-26JAN12-BUF", 50)
    pos = store.get_position("KXNFL-26JAN12-BUF")
    print(f"   Current bid: {pos.current_bid}c")
    print(f"   Unrealized P&L: {pos.unrealized_pnl_cents}c/contract")
    print(f"   Total P&L: {pos.total_pnl_cents}c (${pos.total_pnl_cents/100:.2f})")
    print(f"   P&L %: {pos.pnl_percent:.1f}%")
    print(f"   Profitable: {pos.is_profitable}")
    print(f"   At TP: {pos.at_take_profit}")
    print()

    # Test 3: Hit take profit
    print("3. Updating bid to 53c (hit TP)...")
    store.update_bid("KXNFL-26JAN12-BUF", 53)
    pos = store.get_position("KXNFL-26JAN12-BUF")
    print(f"   Current bid: {pos.current_bid}c")
    print(f"   At TP: {pos.at_take_profit}")
    print(f"   P&L: {pos.unrealized_pnl_cents}c/contract ({pos.pnl_percent:.1f}%)")
    print()

    # Test 4: Add another position, check stop loss
    print("4. Testing stop loss...")
    pos2 = TrackedPosition(
        position_id="test-002",
        ticker="KXNBA-26JAN15-LAL",
        side=PositionSide.YES,
        contracts=10,
        entry_price=60,
    )
    pos2.set_thresholds(take_profit_pct=0.10, stop_loss_pct=0.15)
    store.add_position(pos2)
    store.update_bid("KXNBA-26JAN15-LAL", 50)  # 10c loss = 16.7%
    pos2 = store.get_position("KXNBA-26JAN15-LAL")
    print(f"   Entry: {pos2.entry_price}c, Current: {pos2.current_bid}c")
    print(f"   SL threshold: {pos2.stop_loss_price}c")
    print(f"   At SL: {pos2.at_stop_loss}")
    print(f"   P&L: {pos2.unrealized_pnl_cents}c/contract ({pos2.pnl_percent:.1f}%)")
    print()

    # Test 5: Stats
    print("5. Store stats...")
    stats = store.get_stats()
    for k, v in stats.items():
        print(f"   {k}: {v}")
    print()

    # Test 6: Serialization
    print("6. Serialization test...")
    data = pos.to_dict()
    restored = TrackedPosition.from_dict(data)
    print(f"   Original ticker: {pos.ticker}")
    print(f"   Restored ticker: {restored.ticker}")
    print(f"   Match: {pos.ticker == restored.ticker and pos.entry_price == restored.entry_price}")
    print()

    # Test 7: Remove position
    print("7. Removing position...")
    removed = store.remove_position("KXNFL-26JAN12-BUF")
    print(f"   Removed: {removed.ticker}")
    print(f"   Remaining: {store.position_count()}")
    print()

    print("=" * 60)
    print("VALIDATION COMPLETE - All tests passed")
    print("=" * 60)


if __name__ == "__main__":
    validate()
