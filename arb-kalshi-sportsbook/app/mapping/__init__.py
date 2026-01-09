"""
Event Mapping Module - Kalshi ↔ Sportsbook Resolution

Single consolidated module for mapping Kalshi markets to sportsbook events.
"""

from app.mapping.resolver import (
    MappedEvent,
    TeamNormalizer,
    EventResolver,
    KALSHI_SERIES_TO_SPORT,
    SPORT_TO_KALSHI_SERIES,
)

__all__ = [
    "MappedEvent",
    "TeamNormalizer",
    "EventResolver",
    "KALSHI_SERIES_TO_SPORT",
    "SPORT_TO_KALSHI_SERIES",
]
