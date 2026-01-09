"""
Data ingestion service.

Provides unified ingestion from multiple sources:
- JSONL files (historical/batch)
- Redis hot cache
- Kalshi WebSocket (real-time)
- Sportsbook APIs

All data flows to QuestDB for cold storage and analytics.
"""

from pathlib import Path
from typing import Optional
from datetime import datetime, timezone

from app.data.questdb import QuestDBILPClient, QuestDBClient, create_tables
from app.data.jsonl_reader import JSONLReader, KalshiMarket


class IngestService:
    """
    Unified data ingestion service.

    Coordinates ingestion from multiple sources into QuestDB.
    """

    def __init__(
        self,
        questdb_host: str = "localhost",
        questdb_ilp_port: int = 9009,
        questdb_pg_port: int = 8812,
    ):
        self.questdb_host = questdb_host
        self.questdb_ilp_port = questdb_ilp_port
        self.questdb_pg_port = questdb_pg_port

    def ensure_schema(self):
        """Ensure QuestDB tables exist."""
        create_tables()

    def ingest_jsonl(
        self,
        file_path: str,
        batch_size: int = 1000,
        progress_callback: Optional[callable] = None,
    ) -> dict:
        """
        Ingest markets from a JSONL file into QuestDB.

        Args:
            file_path: Path to JSONL file
            batch_size: Number of records per batch
            progress_callback: Optional callback(count, market) for progress

        Returns:
            dict with ingestion stats
        """
        reader = JSONLReader(file_path)
        stats = {
            "total_read": 0,
            "total_written": 0,
            "total_errors": 0,
            "start_time": datetime.now(timezone.utc),
        }

        with QuestDBILPClient(self.questdb_host, self.questdb_ilp_port) as ilp:
            for batch in reader.iter_batches(batch_size):
                for market in batch:
                    stats["total_read"] += 1

                    try:
                        kwargs = market.to_ilp_kwargs()
                        ilp.write_market_snapshot(**kwargs)
                        stats["total_written"] += 1

                        if progress_callback:
                            progress_callback(stats["total_read"], market)
                    except Exception as e:
                        stats["total_errors"] += 1

                ilp.flush()

        stats["end_time"] = datetime.now(timezone.utc)
        stats["duration_seconds"] = (
            stats["end_time"] - stats["start_time"]
        ).total_seconds()

        return stats

    def ingest_market(self, market: KalshiMarket) -> bool:
        """
        Ingest a single market into QuestDB.

        Args:
            market: KalshiMarket object

        Returns:
            True if successful
        """
        try:
            with QuestDBILPClient(self.questdb_host, self.questdb_ilp_port) as ilp:
                kwargs = market.to_ilp_kwargs()
                ilp.write_market_snapshot(**kwargs)
                ilp.flush()
            return True
        except Exception:
            return False

    def query_latest_markets(self, limit: int = 100) -> list[dict]:
        """Query latest markets from QuestDB."""
        with QuestDBClient(
            host=self.questdb_host,
            port=self.questdb_pg_port,
        ) as client:
            return client.execute(f"""
                SELECT *
                FROM kalshi_markets
                LATEST ON timestamp PARTITION BY market_ticker
                ORDER BY timestamp DESC
                LIMIT {limit}
            """)

    def query_market_by_ticker(self, ticker: str) -> list[dict]:
        """Query market history by ticker."""
        with QuestDBClient(
            host=self.questdb_host,
            port=self.questdb_pg_port,
        ) as client:
            return client.execute("""
                SELECT *
                FROM kalshi_markets
                WHERE market_ticker = %s
                ORDER BY timestamp DESC
                LIMIT 100
            """, (ticker,))
