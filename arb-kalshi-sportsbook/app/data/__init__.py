"""
Data persistence module.

Provides:
- QuestDB clients (ILP for writes, PostgreSQL for queries)
- JSONL file readers for batch ingestion
- Data models for Kalshi markets
"""

from app.data.questdb import (
    QuestDBILPClient,
    QuestDBClient,
    create_tables,
)
from app.data.jsonl_reader import (
    JSONLReader,
    KalshiMarket,
    read_jsonl_sample,
    get_file_stats,
)

__all__ = [
    "QuestDBILPClient",
    "QuestDBClient",
    "create_tables",
    "JSONLReader",
    "KalshiMarket",
    "read_jsonl_sample",
    "get_file_stats",
]
