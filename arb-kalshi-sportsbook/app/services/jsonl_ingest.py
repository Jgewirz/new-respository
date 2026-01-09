"""
JSONL to QuestDB ingestion pipeline.

Consumes Kalshi sports markets from JSONL files and writes to QuestDB
using the high-performance ILP (InfluxDB Line Protocol) interface.

Architecture:
    JSONL File → JSONLReader → QuestDBILPClient → QuestDB (kalshi_markets table)

Usage:
    python -m app.services.jsonl_ingest path/to/kalshi_sports_markets.jsonl

    # With options:
    python -m app.services.jsonl_ingest data.jsonl --batch-size 5000 --dry-run
"""

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.data.jsonl_reader import JSONLReader, KalshiMarket
from app.data.questdb import QuestDBILPClient, QuestDBClient, create_tables


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_BATCH_SIZE = 1000
PROGRESS_REPORT_INTERVAL = 10000


# =============================================================================
# Ingestion Pipeline
# =============================================================================

class JSONLIngestionPipeline:
    """
    Pipeline for ingesting JSONL market data into QuestDB.

    Features:
    - Batched ILP writes for high throughput
    - Progress tracking and reporting
    - Error handling with skip-on-failure
    - Dry-run mode for testing
    """

    def __init__(
        self,
        jsonl_path: str,
        batch_size: int = DEFAULT_BATCH_SIZE,
        dry_run: bool = False,
        questdb_host: str = "localhost",
        questdb_port: int = 9009,
    ):
        self.jsonl_path = jsonl_path
        self.batch_size = batch_size
        self.dry_run = dry_run
        self.questdb_host = questdb_host
        self.questdb_port = questdb_port

        # Stats
        self.stats = {
            "total_read": 0,
            "total_written": 0,
            "total_errors": 0,
            "start_time": None,
            "end_time": None,
        }

    def ensure_tables(self):
        """Ensure QuestDB tables exist."""
        if self.dry_run:
            print("[DRY-RUN] Would create tables")
            return

        print("Ensuring QuestDB tables exist...")
        try:
            create_tables()
            print("Tables ready.")
        except Exception as e:
            print(f"Warning: Could not create tables (may already exist): {e}")

    def run(self) -> dict:
        """Execute the ingestion pipeline."""
        print(f"\n{'='*60}")
        print("JSONL to QuestDB Ingestion Pipeline")
        print(f"{'='*60}")
        print(f"  Source:     {self.jsonl_path}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Dry run:    {self.dry_run}")
        print(f"  QuestDB:    {self.questdb_host}:{self.questdb_port}")
        print()

        # Ensure tables exist
        self.ensure_tables()

        # Initialize reader
        reader = JSONLReader(self.jsonl_path)
        print(f"File size: {reader.file_size_mb:.2f} MB")

        self.stats["start_time"] = datetime.now(timezone.utc)

        if self.dry_run:
            self._run_dry(reader)
        else:
            self._run_live(reader)

        self.stats["end_time"] = datetime.now(timezone.utc)
        self._print_summary()

        return self.stats

    def _run_dry(self, reader: JSONLReader):
        """Dry run - read and parse but don't write."""
        print("\n[DRY-RUN] Validating data without writing...\n")

        for idx, market in reader.iter_with_progress(PROGRESS_REPORT_INTERVAL):
            self.stats["total_read"] += 1

            # Validate by converting to ILP kwargs
            try:
                kwargs = market.to_ilp_kwargs()
                self.stats["total_written"] += 1
            except Exception as e:
                self.stats["total_errors"] += 1
                print(f"  Error parsing {market.market_ticker}: {e}")

        print(f"\n[DRY-RUN] Completed. {self.stats['total_read']:,} records validated.")

    def _run_live(self, reader: JSONLReader):
        """Live run - write to QuestDB."""
        print("\nConnecting to QuestDB ILP endpoint...")

        with QuestDBILPClient(self.questdb_host, self.questdb_port) as ilp:
            print("Connected. Starting ingestion...\n")

            batch_count = 0
            for batch in reader.iter_batches(self.batch_size):
                batch_count += 1

                for market in batch:
                    self.stats["total_read"] += 1

                    try:
                        kwargs = market.to_ilp_kwargs()
                        ilp.write_market_snapshot(**kwargs)
                        self.stats["total_written"] += 1
                    except Exception as e:
                        self.stats["total_errors"] += 1
                        if self.stats["total_errors"] <= 10:
                            print(f"  Error writing {market.market_ticker}: {e}")
                        elif self.stats["total_errors"] == 11:
                            print("  (suppressing further error messages...)")

                # Progress report
                if self.stats["total_read"] % PROGRESS_REPORT_INTERVAL == 0:
                    self._print_progress()

            ilp.flush()

        print("\nIngestion complete.")

    def _print_progress(self):
        """Print progress update."""
        elapsed = (datetime.now(timezone.utc) - self.stats["start_time"]).total_seconds()
        rate = self.stats["total_written"] / elapsed if elapsed > 0 else 0
        print(
            f"  Progress: {self.stats['total_read']:,} read, "
            f"{self.stats['total_written']:,} written, "
            f"{self.stats['total_errors']:,} errors, "
            f"{rate:,.0f} records/sec"
        )

    def _print_summary(self):
        """Print final summary."""
        elapsed = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()
        rate = self.stats["total_written"] / elapsed if elapsed > 0 else 0

        print(f"\n{'='*60}")
        print("INGESTION SUMMARY")
        print(f"{'='*60}")
        print(f"  Records read:    {self.stats['total_read']:,}")
        print(f"  Records written: {self.stats['total_written']:,}")
        print(f"  Errors:          {self.stats['total_errors']:,}")
        print(f"  Duration:        {elapsed:.1f} seconds")
        print(f"  Throughput:      {rate:,.0f} records/second")
        print(f"{'='*60}\n")


# =============================================================================
# Query Utilities
# =============================================================================

def query_ingested_data(limit: int = 10):
    """Query recently ingested data from QuestDB."""
    print(f"\nQuerying kalshi_markets table (last {limit} records)...")

    with QuestDBClient() as client:
        results = client.execute(f"""
            SELECT
                market_ticker,
                series_ticker,
                title,
                status,
                yes_bid,
                yes_ask,
                volume,
                timestamp
            FROM kalshi_markets
            ORDER BY timestamp DESC
            LIMIT {limit}
        """)

        if not results:
            print("  No data found.")
            return

        print(f"\n  Found {len(results)} records:")
        for row in results:
            print(f"    {row['market_ticker']}: {row['title'][:40]}...")
            print(f"      Yes: {row['yes_bid']}/{row['yes_ask']}, Volume: {row['volume']}")


def query_market_stats():
    """Query aggregate statistics from ingested data."""
    print("\nQuerying market statistics...")

    with QuestDBClient() as client:
        results = client.execute("""
            SELECT
                count() as total_records,
                count_distinct(market_ticker) as unique_markets,
                count_distinct(series_ticker) as unique_series,
                sum(volume) as total_volume,
                avg(liquidity) as avg_liquidity,
                min(timestamp) as first_record,
                max(timestamp) as last_record
            FROM kalshi_markets
        """)

        if results:
            stats = results[0]
            print(f"\n  Total records:   {stats['total_records']:,}")
            print(f"  Unique markets:  {stats['unique_markets']:,}")
            print(f"  Unique series:   {stats['unique_series']:,}")
            print(f"  Total volume:    {stats['total_volume']:,}")
            print(f"  Avg liquidity:   {stats['avg_liquidity']:,.0f}")
            print(f"  Time range:      {stats['first_record']} to {stats['last_record']}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ingest Kalshi sports markets JSONL into QuestDB"
    )
    parser.add_argument(
        "jsonl_file",
        help="Path to JSONL file containing Kalshi market data"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for ILP writes (default: {DEFAULT_BATCH_SIZE})"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and validate without writing to QuestDB"
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="QuestDB host (default: localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9009,
        help="QuestDB ILP port (default: 9009)"
    )
    parser.add_argument(
        "--query",
        action="store_true",
        help="Query ingested data after completion"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show aggregate statistics after completion"
    )

    args = parser.parse_args()

    # Validate file exists
    if not Path(args.jsonl_file).exists():
        print(f"Error: File not found: {args.jsonl_file}")
        sys.exit(1)

    # Run pipeline
    pipeline = JSONLIngestionPipeline(
        jsonl_path=args.jsonl_file,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        questdb_host=args.host,
        questdb_port=args.port,
    )

    stats = pipeline.run()

    # Optional post-ingestion queries
    if args.query and not args.dry_run:
        query_ingested_data()

    if args.stats and not args.dry_run:
        query_market_stats()

    # Exit with error code if there were failures
    if stats["total_errors"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
