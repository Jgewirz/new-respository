"""
JSONL file reader for Kalshi sports markets data.

Provides streaming iteration over large JSONL files with:
- Memory-efficient line-by-line processing
- Automatic JSON parsing and validation
- Progress tracking for large files
- Batch yielding for efficient ingestion
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional, Any
from dataclasses import dataclass


@dataclass
class KalshiMarket:
    """Parsed Kalshi market data from JSONL."""

    # Identifiers
    market_ticker: str
    series_ticker: str
    event_ticker: str

    # Market info
    title: str
    subtitle: str
    status: str

    # Pricing (cents, 0-100)
    yes_bid: int
    yes_ask: int
    no_bid: int
    no_ask: int
    last_price: int

    # Volume/liquidity
    volume: int
    volume_24h: int
    liquidity: int

    # Timestamps (ISO format strings)
    open_time: Optional[str]
    close_time: Optional[str]
    expiration_time: Optional[str]
    fetched_at: Optional[str]

    @classmethod
    def from_dict(cls, data: dict) -> "KalshiMarket":
        """Parse a JSONL line into a KalshiMarket object."""
        pricing = data.get("pricing", {})
        meta = data.get("_meta", {})

        return cls(
            market_ticker=data.get("market_ticker", ""),
            series_ticker=data.get("series_ticker", ""),
            event_ticker=data.get("event_ticker", ""),
            title=data.get("title", ""),
            subtitle=data.get("subtitle", ""),
            status=data.get("status", ""),
            yes_bid=pricing.get("yes_bid", 0) or 0,
            yes_ask=pricing.get("yes_ask", 0) or 0,
            no_bid=pricing.get("no_bid", 0) or 0,
            no_ask=pricing.get("no_ask", 0) or 0,
            last_price=pricing.get("last_price", 0) or 0,
            volume=data.get("volume", 0) or 0,
            volume_24h=data.get("volume_24h", 0) or 0,
            liquidity=data.get("liquidity", 0) or 0,
            open_time=data.get("open_time"),
            close_time=data.get("close_time"),
            expiration_time=data.get("expiration_time"),
            fetched_at=meta.get("fetched_at"),
        )

    def to_ilp_kwargs(self) -> dict:
        """Convert to kwargs for QuestDBILPClient.write_market_snapshot()."""
        return {
            "market_ticker": self.market_ticker,
            "series_ticker": self.series_ticker,
            "event_ticker": self.event_ticker,
            "title": self.title,
            "subtitle": self.subtitle,
            "status": self.status,
            "yes_bid": self.yes_bid,
            "yes_ask": self.yes_ask,
            "no_bid": self.no_bid,
            "no_ask": self.no_ask,
            "last_price": self.last_price,
            "volume": self.volume,
            "volume_24h": self.volume_24h,
            "liquidity": self.liquidity,
            "open_time_ns": self._parse_timestamp_ns(self.open_time),
            "close_time_ns": self._parse_timestamp_ns(self.close_time),
            "expiration_time_ns": self._parse_timestamp_ns(self.expiration_time),
            "fetched_at_ns": self._parse_timestamp_ns(self.fetched_at),
        }

    @staticmethod
    def _parse_timestamp_ns(iso_str: Optional[str]) -> Optional[int]:
        """Parse ISO timestamp string to nanoseconds."""
        if not iso_str:
            return None
        try:
            # Handle various ISO formats
            dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
            return int(dt.timestamp() * 1e9)
        except (ValueError, AttributeError):
            return None


class JSONLReader:
    """
    Memory-efficient JSONL file reader with streaming.

    Usage:
        reader = JSONLReader("kalshi_sports_markets.jsonl")
        for market in reader.iter_markets():
            process(market)

        # Or with batching:
        for batch in reader.iter_batches(batch_size=1000):
            process_batch(batch)
    """

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {file_path}")

        self._file_size = self.file_path.stat().st_size
        self._line_count: Optional[int] = None

    @property
    def file_size_mb(self) -> float:
        """File size in megabytes."""
        return self._file_size / (1024 * 1024)

    def count_lines(self) -> int:
        """Count total lines in the file (caches result)."""
        if self._line_count is None:
            with open(self.file_path, "r", encoding="utf-8") as f:
                self._line_count = sum(1 for _ in f)
        return self._line_count

    def iter_raw(self) -> Iterator[dict]:
        """Iterate over raw JSON objects from each line."""
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON at line {line_num}: {e}")
                    continue

    def iter_markets(self) -> Iterator[KalshiMarket]:
        """Iterate over parsed KalshiMarket objects."""
        for data in self.iter_raw():
            try:
                yield KalshiMarket.from_dict(data)
            except Exception as e:
                ticker = data.get("market_ticker", "unknown")
                print(f"Warning: Failed to parse market {ticker}: {e}")
                continue

    def iter_batches(
        self,
        batch_size: int = 1000
    ) -> Iterator[list[KalshiMarket]]:
        """Iterate over batches of KalshiMarket objects."""
        batch = []
        for market in self.iter_markets():
            batch.append(market)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def iter_with_progress(
        self,
        report_every: int = 10000
    ) -> Iterator[tuple[int, KalshiMarket]]:
        """Iterate with progress reporting."""
        for idx, market in enumerate(self.iter_markets()):
            if idx > 0 and idx % report_every == 0:
                print(f"  Processed {idx:,} markets...")
            yield idx, market


# =============================================================================
# Convenience Functions
# =============================================================================

def read_jsonl_sample(file_path: str, n: int = 10) -> list[KalshiMarket]:
    """Read first N markets from a JSONL file."""
    reader = JSONLReader(file_path)
    markets = []
    for market in reader.iter_markets():
        markets.append(market)
        if len(markets) >= n:
            break
    return markets


def get_file_stats(file_path: str) -> dict:
    """Get statistics about a JSONL file."""
    reader = JSONLReader(file_path)

    # Sample first 100 lines to estimate
    sample_tickers = set()
    sample_series = set()
    sample_statuses = set()

    for idx, market in enumerate(reader.iter_markets()):
        sample_tickers.add(market.market_ticker)
        sample_series.add(market.series_ticker)
        sample_statuses.add(market.status)
        if idx >= 100:
            break

    return {
        "file_path": str(reader.file_path),
        "file_size_mb": round(reader.file_size_mb, 2),
        "sample_tickers": len(sample_tickers),
        "sample_series": list(sample_series)[:10],
        "sample_statuses": list(sample_statuses),
    }


# =============================================================================
# CLI Test
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python jsonl_reader.py <jsonl_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    print(f"Reading: {file_path}")

    stats = get_file_stats(file_path)
    print(f"\nFile Stats:")
    print(f"  Size: {stats['file_size_mb']} MB")
    print(f"  Sample tickers: {stats['sample_tickers']}")
    print(f"  Series: {stats['sample_series']}")
    print(f"  Statuses: {stats['sample_statuses']}")

    print(f"\nFirst 5 markets:")
    for market in read_jsonl_sample(file_path, 5):
        print(f"  {market.market_ticker}: {market.title}")
        print(f"    Yes: {market.yes_bid}/{market.yes_ask}, No: {market.no_bid}/{market.no_ask}")
