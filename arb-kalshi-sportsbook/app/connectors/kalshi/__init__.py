"""
Kalshi API Connector

Production-grade client for the Kalshi Trading API.

Components:
    - auth: RSA-PSS authentication for API requests
    - client: HTTP client for market data and trading
    - models: Pydantic models for API responses (in client.py)
    - rate_limit: Rate limiting utilities (in client.py)

Usage:
    from app.connectors.kalshi import KalshiClient, KalshiAuth

    # Simple usage with context manager
    with KalshiClient.from_env() as client:
        balance = client.get_balance()
        market = client.get_market("KXNFL-26JAN11-BUF")
        order = client.buy_yes("KXNFL-26JAN11-BUF", count=10, price=52)

    # Async usage
    async with KalshiAsyncClient.from_env() as client:
        balance = await client.get_balance()

    # Manual auth if needed
    auth = KalshiAuth.from_env()
    headers = auth.get_headers("GET", "/portfolio/balance")
"""

from app.connectors.kalshi.auth import (
    KalshiAuth,
    AuthHeaders,
    KalshiAuthError,
    KeyLoadError,
    SignatureError,
    ConfigurationError,
    create_auth_from_env,
)

from app.connectors.kalshi.client import (
    # Clients
    KalshiClient,
    KalshiAsyncClient,
    # Response Models
    Balance,
    Position,
    Order,
    Market,
    Event,
    # Enums
    OrderSide,
    OrderAction,
    OrderType,
    OrderStatus,
    MarketStatus,
    # Errors
    KalshiAPIError,
    AuthenticationError,
    RateLimitError,
    OrderRejectedError,
    InsufficientFundsError,
    MarketClosedError,
    NotFoundError,
    # Rate Limiter
    RateLimiter,
)

__all__ = [
    # Auth
    "KalshiAuth",
    "AuthHeaders",
    "KalshiAuthError",
    "KeyLoadError",
    "SignatureError",
    "ConfigurationError",
    "create_auth_from_env",
    # Clients
    "KalshiClient",
    "KalshiAsyncClient",
    # Response Models
    "Balance",
    "Position",
    "Order",
    "Market",
    "Event",
    # Enums
    "OrderSide",
    "OrderAction",
    "OrderType",
    "OrderStatus",
    "MarketStatus",
    # Errors
    "KalshiAPIError",
    "AuthenticationError",
    "RateLimitError",
    "OrderRejectedError",
    "InsufficientFundsError",
    "MarketClosedError",
    "NotFoundError",
    # Rate Limiter
    "RateLimiter",
]
