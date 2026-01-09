"""
Business logic services module.

Provides:
- IngestService: Unified data ingestion from multiple sources
- JSONLIngestionPipeline: Batch JSONL to QuestDB ingestion
"""

from app.services.ingest_service import IngestService
from app.services.jsonl_ingest import (
    JSONLIngestionPipeline,
    query_ingested_data,
    query_market_stats,
)

__all__ = [
    "IngestService",
    "JSONLIngestionPipeline",
    "query_ingested_data",
    "query_market_stats",
]
