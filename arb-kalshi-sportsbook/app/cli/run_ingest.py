"""
CLI command to run data ingestion.

Supports multiple ingestion modes:
- jsonl: Ingest from JSONL file (batch/historical)
- redis: Sync from Redis hot cache
- websocket: Real-time Kalshi WebSocket streaming

Usage:
    python -m app.cli.run_ingest jsonl path/to/data.jsonl
    python -m app.cli.run_ingest jsonl data.jsonl --batch-size 5000 --dry-run
    python -m app.cli.run_ingest schema  # Create tables only
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services.jsonl_ingest import JSONLIngestionPipeline, query_ingested_data, query_market_stats
from app.data.questdb import create_tables


def cmd_jsonl(args):
    """Run JSONL ingestion."""
    if not Path(args.file).exists():
        print(f"Error: File not found: {args.file}")
        return 1

    pipeline = JSONLIngestionPipeline(
        jsonl_path=args.file,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        questdb_host=args.host,
        questdb_port=args.port,
    )

    stats = pipeline.run()

    if args.query:
        query_ingested_data()

    if args.stats:
        query_market_stats()

    return 1 if stats["total_errors"] > 0 else 0


def cmd_schema(args):
    """Create QuestDB tables."""
    print("Creating QuestDB tables...")
    try:
        create_tables()
        print("Done.")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


def cmd_query(args):
    """Query ingested data."""
    if args.stats:
        query_market_stats()
    else:
        query_ingested_data(limit=args.limit)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Kalshi data ingestion CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Ingestion commands")

    # JSONL ingestion
    jsonl_parser = subparsers.add_parser(
        "jsonl",
        help="Ingest from JSONL file"
    )
    jsonl_parser.add_argument(
        "file",
        help="Path to JSONL file"
    )
    jsonl_parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for ILP writes (default: 1000)"
    )
    jsonl_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate without writing to QuestDB"
    )
    jsonl_parser.add_argument(
        "--host",
        default="localhost",
        help="QuestDB host (default: localhost)"
    )
    jsonl_parser.add_argument(
        "--port",
        type=int,
        default=9009,
        help="QuestDB ILP port (default: 9009)"
    )
    jsonl_parser.add_argument(
        "--query",
        action="store_true",
        help="Query data after ingestion"
    )
    jsonl_parser.add_argument(
        "--stats",
        action="store_true",
        help="Show statistics after ingestion"
    )
    jsonl_parser.set_defaults(func=cmd_jsonl)

    # Schema creation
    schema_parser = subparsers.add_parser(
        "schema",
        help="Create QuestDB tables"
    )
    schema_parser.set_defaults(func=cmd_schema)

    # Query data
    query_parser = subparsers.add_parser(
        "query",
        help="Query ingested data"
    )
    query_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of records to show (default: 10)"
    )
    query_parser.add_argument(
        "--stats",
        action="store_true",
        help="Show aggregate statistics"
    )
    query_parser.set_defaults(func=cmd_query)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
