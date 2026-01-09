"""
Kalshi API Client - Production-Grade Low-Latency Trading Interface

High-performance HTTP client for Kalshi Trading API with:
- RSA-PSS authenticated requests via KalshiAuth
- Connection pooling for sub-10ms latency
- Automatic retries with exponential backoff
- Rate limiting (10 req/s burst, 100 req/min sustained)
- Full support for portfolio, order, and market endpoints

Integration Points (per EXECUTION_ENGINE_PLAN.md):
- Used by: kalshi_executor.py for live order submission
- Depends on: auth.py for RSA-PSS signing
- Feeds: circuit_breaker.py with balance/position data

Endpoints Implemented:
    GET  /portfolio/balance     - Check available funds
    GET  /portfolio/positions   - Current holdings
    POST /portfolio/orders      - Submit new order
    GET  /portfolio/orders/{id} - Check order status
    DELETE /portfolio/orders/{id} - Cancel order
    GET  /markets/{ticker}      - Current market price
    GET  /markets               - List markets (with filters)
    GET  /events/{event_id}     - Event details

Usage:
    from app.connectors.kalshi import KalshiClient

    # Sync client
    with KalshiClient.from_env() as client:
        balance = client.get_balance()
        market = client.get_market("KXNFL-26JAN11-BUF")
        order = client.create_order(
            ticker="KXNFL-26JAN11-BUF",
            side="yes",
            action="buy",
            count=10,
            type="limit",
            yes_price=52,
        )

    # Async client
    async with KalshiAsyncClient.from_env() as client:
        balance = await client.get_balance()
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import httpx
from pydantic import BaseModel, Field

from app.connectors.kalshi.auth import ConfigurationError, KalshiAuth

# =============================================================================
# CONFIGURATION
# =============================================================================

# API URLs
KALSHI_PROD_URL = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_DEMO_URL = "https://demo-api.kalshi.co/trade-api/v2"

# Performance tuning
DEFAULT_TIMEOUT = 5.0  # 5 second timeout for low latency
DEFAULT_RETRIES = 3
RETRY_BACKOFF_BASE = 0.1  # 100ms initial backoff
MAX_RETRY_BACKOFF = 2.0  # 2 second max backoff

# Rate limiting (Kalshi limits: ~10 req/s burst)
RATE_LIMIT_REQUESTS = 10
RATE_LIMIT_WINDOW = 1.0  # seconds


# =============================================================================
# ENUMS
# =============================================================================

class OrderSide(str, Enum):
    """Order side - yes or no contract."""
    YES = "yes"
    NO = "no"


class OrderAction(str, Enum):
    """Order action - buy or sell."""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type."""
    LIMIT = "limit"
    MARKET = "market"


class OrderStatus(str, Enum):
    """Order status from Kalshi API."""
    RESTING = "resting"
    FILLED = "filled"
    CANCELED = "canceled"
    PENDING = "pending"


class MarketStatus(str, Enum):
    """Market status."""
    OPEN = "open"
    CLOSED = "closed"
    SETTLED = "settled"


# =============================================================================
# RESPONSE MODELS (Pydantic for type safety)
# =============================================================================

class Balance(BaseModel):
    """Portfolio balance response."""
    balance: int  # Available balance in cents
    payout: int = 0  # Pending payout

    @property
    def balance_dollars(self) -> float:
        """Balance in dollars."""
        return self.balance / 100

    @property
    def payout_dollars(self) -> float:
        """Payout in dollars."""
        return self.payout / 100


class Position(BaseModel):
    """Single position in portfolio."""
    ticker: str
    event_ticker: str = ""
    market_title: str = ""
    yes_count: int = 0
    no_count: int = 0
    average_yes_price: int = 0  # cents
    average_no_price: int = 0  # cents
    settlement_value: int | None = None

    @property
    def net_contracts(self) -> int:
        """Net position (positive = long yes, negative = long no)."""
        return self.yes_count - self.no_count

    @property
    def is_yes_position(self) -> bool:
        """Has YES contracts."""
        return self.yes_count > 0

    @property
    def is_no_position(self) -> bool:
        """Has NO contracts."""
        return self.no_count > 0


class Order(BaseModel):
    """Order response from Kalshi API."""
    order_id: str
    client_order_id: str | None = None
    ticker: str
    side: str  # "yes" or "no"
    action: str  # "buy" or "sell"
    type: str  # "limit" or "market"
    status: str
    yes_price: int | None = None
    no_price: int | None = None
    count: int = 0  # Original count
    remaining_count: int = 0
    filled_count: int = 0
    created_time: datetime | None = None
    updated_time: datetime | None = None
    expiration_time: datetime | None = None

    @property
    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED.value

    @property
    def is_resting(self) -> bool:
        return self.status == OrderStatus.RESTING.value

    @property
    def is_canceled(self) -> bool:
        return self.status == OrderStatus.CANCELED.value

    @property
    def price_cents(self) -> int:
        """Get price in cents."""
        return self.yes_price or self.no_price or 0


class Market(BaseModel):
    """Market data from Kalshi API."""
    ticker: str
    event_ticker: str = ""
    title: str = ""
    subtitle: str = ""
    status: str = ""
    yes_bid: int = 0  # Best bid in cents
    yes_ask: int = 0  # Best ask in cents
    no_bid: int = 0
    no_ask: int = 0
    last_price: int = 0
    volume: int = 0
    volume_24h: int = 0
    open_interest: int = 0
    previous_price: int = 0
    previous_yes_bid: int = 0
    previous_yes_ask: int = 0
    result: str | None = None  # "yes", "no", or None if not settled
    close_time: datetime | None = None
    expiration_time: datetime | None = None

    @property
    def spread(self) -> int:
        """Bid-ask spread in cents."""
        return self.yes_ask - self.yes_bid if self.yes_ask and self.yes_bid else 0

    @property
    def midpoint(self) -> float:
        """Midpoint price."""
        if self.yes_bid and self.yes_ask:
            return (self.yes_bid + self.yes_ask) / 2
        return self.last_price

    @property
    def is_active(self) -> bool:
        """Market is actively trading."""
        return self.status == MarketStatus.OPEN.value


class Event(BaseModel):
    """Event details."""
    event_ticker: str
    title: str = ""
    subtitle: str = ""
    category: str = ""
    mutually_exclusive: bool = True
    markets: list[str] = Field(default_factory=list)


# =============================================================================
# ERROR CLASSES
# =============================================================================

class KalshiAPIError(Exception):
    """Base exception for Kalshi API errors."""
    def __init__(self, message: str, status_code: int = 0, response: dict | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response or {}


class AuthenticationError(KalshiAPIError):
    """Authentication failed (401/403)."""
    pass


class RateLimitError(KalshiAPIError):
    """Rate limit exceeded (429)."""
    pass


class OrderRejectedError(KalshiAPIError):
    """Order was rejected."""
    pass


class InsufficientFundsError(KalshiAPIError):
    """Insufficient balance for order."""
    pass


class MarketClosedError(KalshiAPIError):
    """Market is closed for trading."""
    pass


class NotFoundError(KalshiAPIError):
    """Resource not found (404)."""
    pass


# =============================================================================
# RATE LIMITER
# =============================================================================

@dataclass
class RateLimiter:
    """
    Simple token bucket rate limiter.

    Thread-safe for sync operations, async-safe for async operations.
    """
    max_tokens: int = RATE_LIMIT_REQUESTS
    refill_rate: float = RATE_LIMIT_REQUESTS  # tokens per second
    _tokens: float = field(default=0, init=False)
    _last_refill: float = field(default=0, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    def __post_init__(self):
        self._tokens = float(self.max_tokens)
        self._last_refill = time.monotonic()

    def _refill(self) -> None:
        """Refill tokens based on time elapsed."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self.max_tokens, self._tokens + elapsed * self.refill_rate)
        self._last_refill = now

    def acquire_sync(self) -> float:
        """
        Acquire a token synchronously.

        Returns:
            Time to wait before proceeding (0 if immediate)
        """
        self._refill()

        if self._tokens >= 1:
            self._tokens -= 1
            return 0.0

        # Calculate wait time
        wait_time = (1 - self._tokens) / self.refill_rate
        return wait_time

    async def acquire(self) -> None:
        """Acquire a token asynchronously, waiting if necessary."""
        async with self._lock:
            self._refill()

            if self._tokens >= 1:
                self._tokens -= 1
                return

            # Wait for token to be available
            wait_time = (1 - self._tokens) / self.refill_rate
            await asyncio.sleep(wait_time)
            self._tokens = 0


# =============================================================================
# SYNC CLIENT
# =============================================================================

class KalshiClient:
    """
    Synchronous Kalshi API client.

    Designed for low-latency trading with:
    - Connection pooling via httpx
    - Automatic retries with exponential backoff
    - Rate limiting to prevent 429 errors
    - Typed responses for safety

    Usage:
        with KalshiClient.from_env() as client:
            balance = client.get_balance()
            order = client.create_order(...)
    """

    def __init__(
        self,
        auth: KalshiAuth,
        base_url: str = KALSHI_PROD_URL,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_RETRIES,
    ):
        """
        Initialize Kalshi client.

        Args:
            auth: KalshiAuth instance for request signing
            base_url: API base URL (prod or demo)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts for transient errors
        """
        self.auth = auth
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._rate_limiter = RateLimiter()

        # HTTP client with connection pooling
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(
                max_keepalive_connections=5,
                max_connections=10,
                keepalive_expiry=30,
            ),
        )

    @classmethod
    def from_env(cls, demo: bool | None = None) -> KalshiClient:
        """
        Create client from environment variables.

        Environment variables:
            - KALSHI_KEY_ID: API Key ID
            - KALSHI_PRIVATE_KEY_PATH: Path to PEM file
            - KALSHI_DEMO: "true" for demo, "false" for prod
            - KALSHI_BASE_URL: Override base URL
            - PAPER_TRADING: If "true", use demo URL

        Args:
            demo: Override demo mode (None = use env vars)
        """
        auth = KalshiAuth.from_env()

        # Determine if demo mode
        if demo is None:
            # Check multiple env vars for demo mode
            demo_env = os.environ.get("KALSHI_DEMO", "").lower()
            paper_env = os.environ.get("PAPER_TRADING", "").lower()
            demo = demo_env == "true" or paper_env == "true"

        # Get base URL
        if custom_url := os.environ.get("KALSHI_BASE_URL"):
            base_url = custom_url
        elif demo:
            base_url = KALSHI_DEMO_URL
        else:
            base_url = KALSHI_PROD_URL

        return cls(auth=auth, base_url=base_url)

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self) -> KalshiClient:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def _make_request(
        self,
        method: str,
        endpoint: str,
        json: dict | None = None,
        params: dict | None = None,
    ) -> dict:
        """
        Make an authenticated API request with retries.

        Args:
            method: HTTP method
            endpoint: API endpoint (e.g., "/portfolio/balance")
            json: Request body for POST/PUT
            params: Query parameters

        Returns:
            Response JSON as dict

        Raises:
            KalshiAPIError: On API error
        """
        url = f"{self.base_url}{endpoint}"

        for attempt in range(self.max_retries):
            # Rate limiting
            wait_time = self._rate_limiter.acquire_sync()
            if wait_time > 0:
                time.sleep(wait_time)

            # Get auth headers
            headers = self.auth.get_headers(method, endpoint)

            try:
                response = self._client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=json,
                    params=params,
                )

                # Handle response
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 201:
                    return response.json()
                elif response.status_code == 204:
                    return {}
                elif response.status_code == 401 or response.status_code == 403:
                    raise AuthenticationError(
                        f"Authentication failed: {response.text}",
                        status_code=response.status_code,
                    )
                elif response.status_code == 404:
                    raise NotFoundError(
                        f"Resource not found: {endpoint}",
                        status_code=404,
                    )
                elif response.status_code == 429:
                    if attempt < self.max_retries - 1:
                        # Exponential backoff on rate limit
                        backoff = min(
                            RETRY_BACKOFF_BASE * (2 ** attempt),
                            MAX_RETRY_BACKOFF,
                        )
                        time.sleep(backoff)
                        continue
                    raise RateLimitError(
                        "Rate limit exceeded",
                        status_code=429,
                    )
                elif response.status_code >= 500:
                    if attempt < self.max_retries - 1:
                        backoff = RETRY_BACKOFF_BASE * (2 ** attempt)
                        time.sleep(backoff)
                        continue
                    raise KalshiAPIError(
                        f"Server error: {response.text}",
                        status_code=response.status_code,
                    )
                else:
                    # Parse error response
                    try:
                        error_data = response.json()
                        error_msg = error_data.get("error", {}).get("message", response.text)
                        error_code = error_data.get("error", {}).get("code", "")
                    except Exception:
                        error_msg = response.text
                        error_code = ""

                    # Map specific error codes
                    if "insufficient" in error_msg.lower() or error_code == "insufficient_balance":
                        raise InsufficientFundsError(
                            error_msg,
                            status_code=response.status_code,
                            response=error_data if 'error_data' in dir() else {},
                        )
                    elif "closed" in error_msg.lower() or error_code == "market_closed":
                        raise MarketClosedError(
                            error_msg,
                            status_code=response.status_code,
                        )
                    elif "rejected" in error_msg.lower():
                        raise OrderRejectedError(
                            error_msg,
                            status_code=response.status_code,
                        )
                    else:
                        raise KalshiAPIError(
                            error_msg,
                            status_code=response.status_code,
                        )

            except httpx.RequestError as e:
                if attempt < self.max_retries - 1:
                    backoff = RETRY_BACKOFF_BASE * (2 ** attempt)
                    time.sleep(backoff)
                    continue
                raise KalshiAPIError(f"Request failed: {e}")

        raise KalshiAPIError("Max retries exceeded")

    # =========================================================================
    # PORTFOLIO ENDPOINTS
    # =========================================================================

    def get_balance(self) -> Balance:
        """
        Get portfolio balance.

        Returns:
            Balance with available funds in cents
        """
        data = self._make_request("GET", "/portfolio/balance")
        return Balance(**data)

    def get_positions(
        self,
        ticker: str | None = None,
        event_ticker: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> list[Position]:
        """
        Get portfolio positions.

        Args:
            ticker: Filter by market ticker
            event_ticker: Filter by event
            limit: Max results (default 100)
            cursor: Pagination cursor

        Returns:
            List of Position objects
        """
        params: dict[str, Any] = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if event_ticker:
            params["event_ticker"] = event_ticker
        if cursor:
            params["cursor"] = cursor

        data = self._make_request("GET", "/portfolio/positions", params=params)
        positions = data.get("market_positions", [])
        return [Position(**p) for p in positions]

    def get_position(self, ticker: str) -> Position | None:
        """
        Get position for a specific market.

        Args:
            ticker: Market ticker

        Returns:
            Position or None if not found
        """
        positions = self.get_positions(ticker=ticker, limit=1)
        return positions[0] if positions else None

    # =========================================================================
    # ORDER ENDPOINTS
    # =========================================================================

    def create_order(
        self,
        ticker: str,
        side: str | OrderSide,
        action: str | OrderAction,
        count: int,
        type: str | OrderType = OrderType.LIMIT,
        yes_price: int | None = None,
        no_price: int | None = None,
        client_order_id: str | None = None,
        expiration_ts: int | None = None,
        buy_max_cost: int | None = None,
        sell_position_floor: int | None = None,
    ) -> Order:
        """
        Create a new order.

        Args:
            ticker: Market ticker (e.g., "KXNFL-26JAN11-BUF")
            side: "yes" or "no"
            action: "buy" or "sell"
            count: Number of contracts
            type: "limit" or "market"
            yes_price: Limit price for YES side (1-99 cents)
            no_price: Limit price for NO side (1-99 cents)
            client_order_id: Optional client-generated UUID for deduplication
            expiration_ts: Order expiration timestamp (seconds)
            buy_max_cost: Max total cost for buy orders
            sell_position_floor: Min position after sell

        Returns:
            Order object with order_id

        Raises:
            InsufficientFundsError: Not enough balance
            MarketClosedError: Market not accepting orders
            OrderRejectedError: Order validation failed
        """
        # Normalize enums
        side_str = side.value if isinstance(side, OrderSide) else side
        action_str = action.value if isinstance(action, OrderAction) else action
        type_str = type.value if isinstance(type, OrderType) else type

        # Build request body
        body: dict[str, Any] = {
            "ticker": ticker,
            "side": side_str,
            "action": action_str,
            "count": count,
            "type": type_str,
        }

        # Add price based on side
        if yes_price is not None:
            body["yes_price"] = yes_price
        if no_price is not None:
            body["no_price"] = no_price

        # Optional fields
        if client_order_id:
            body["client_order_id"] = client_order_id
        else:
            # Generate UUID for idempotency
            body["client_order_id"] = str(uuid.uuid4())

        if expiration_ts:
            body["expiration_ts"] = expiration_ts
        if buy_max_cost is not None:
            body["buy_max_cost"] = buy_max_cost
        if sell_position_floor is not None:
            body["sell_position_floor"] = sell_position_floor

        data = self._make_request("POST", "/portfolio/orders", json=body)
        order_data = data.get("order", data)
        return Order(**order_data)

    def get_order(self, order_id: str) -> Order:
        """
        Get order by ID.

        Args:
            order_id: Server-generated order ID

        Returns:
            Order object
        """
        data = self._make_request("GET", f"/portfolio/orders/{order_id}")
        order_data = data.get("order", data)
        return Order(**order_data)

    def cancel_order(self, order_id: str) -> Order:
        """
        Cancel an order.

        Args:
            order_id: Order to cancel

        Returns:
            Updated Order object
        """
        data = self._make_request("DELETE", f"/portfolio/orders/{order_id}")
        order_data = data.get("order", data)
        return Order(**order_data)

    def get_orders(
        self,
        ticker: str | None = None,
        event_ticker: str | None = None,
        status: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> list[Order]:
        """
        List orders with filters.

        Args:
            ticker: Filter by market
            event_ticker: Filter by event
            status: Filter by status ("resting", "filled", etc.)
            limit: Max results
            cursor: Pagination cursor

        Returns:
            List of Order objects
        """
        params: dict[str, Any] = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if event_ticker:
            params["event_ticker"] = event_ticker
        if status:
            params["status"] = status
        if cursor:
            params["cursor"] = cursor

        data = self._make_request("GET", "/portfolio/orders", params=params)
        orders = data.get("orders", [])
        return [Order(**o) for o in orders]

    def get_open_orders(self, ticker: str | None = None) -> list[Order]:
        """
        Get all resting (open) orders.

        Args:
            ticker: Optional market filter

        Returns:
            List of open orders
        """
        return self.get_orders(ticker=ticker, status="resting")

    def cancel_all_orders(self, ticker: str | None = None) -> int:
        """
        Cancel all open orders.

        Args:
            ticker: Optional market filter

        Returns:
            Number of orders canceled
        """
        open_orders = self.get_open_orders(ticker=ticker)
        canceled = 0

        for order in open_orders:
            try:
                self.cancel_order(order.order_id)
                canceled += 1
            except KalshiAPIError:
                pass  # Order may have filled

        return canceled

    # =========================================================================
    # MARKET ENDPOINTS
    # =========================================================================

    def get_market(self, ticker: str) -> Market:
        """
        Get market data by ticker.

        Args:
            ticker: Market ticker (e.g., "KXNFL-26JAN11-BUF")

        Returns:
            Market object with current prices
        """
        data = self._make_request("GET", f"/markets/{ticker}")
        market_data = data.get("market", data)
        return Market(**market_data)

    def get_markets(
        self,
        event_ticker: str | None = None,
        series_ticker: str | None = None,
        status: str | None = None,
        tickers: list[str] | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> list[Market]:
        """
        List markets with filters.

        Args:
            event_ticker: Filter by event
            series_ticker: Filter by series
            status: Filter by status ("open", "closed", etc.)
            tickers: List of specific tickers
            limit: Max results
            cursor: Pagination cursor

        Returns:
            List of Market objects
        """
        params: dict[str, Any] = {"limit": limit}
        if event_ticker:
            params["event_ticker"] = event_ticker
        if series_ticker:
            params["series_ticker"] = series_ticker
        if status:
            params["status"] = status
        if tickers:
            params["tickers"] = ",".join(tickers)
        if cursor:
            params["cursor"] = cursor

        data = self._make_request("GET", "/markets", params=params)
        markets = data.get("markets", [])
        return [Market(**m) for m in markets]

    def get_event(self, event_ticker: str) -> Event:
        """
        Get event details.

        Args:
            event_ticker: Event ticker

        Returns:
            Event object
        """
        data = self._make_request("GET", f"/events/{event_ticker}")
        event_data = data.get("event", data)
        return Event(**event_data)

    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================

    def buy_yes(
        self,
        ticker: str,
        count: int,
        price: int,
        client_order_id: str | None = None,
    ) -> Order:
        """
        Convenience: Buy YES contracts at limit price.

        Args:
            ticker: Market ticker
            count: Number of contracts
            price: Max price in cents (1-99)
            client_order_id: Optional for idempotency

        Returns:
            Order object
        """
        return self.create_order(
            ticker=ticker,
            side=OrderSide.YES,
            action=OrderAction.BUY,
            count=count,
            type=OrderType.LIMIT,
            yes_price=price,
            client_order_id=client_order_id,
        )

    def buy_no(
        self,
        ticker: str,
        count: int,
        price: int,
        client_order_id: str | None = None,
    ) -> Order:
        """
        Convenience: Buy NO contracts at limit price.

        Args:
            ticker: Market ticker
            count: Number of contracts
            price: Max price in cents (1-99)
            client_order_id: Optional for idempotency

        Returns:
            Order object
        """
        return self.create_order(
            ticker=ticker,
            side=OrderSide.NO,
            action=OrderAction.BUY,
            count=count,
            type=OrderType.LIMIT,
            no_price=price,
            client_order_id=client_order_id,
        )

    def sell_yes(
        self,
        ticker: str,
        count: int,
        price: int,
        client_order_id: str | None = None,
    ) -> Order:
        """
        Convenience: Sell YES contracts at limit price.
        """
        return self.create_order(
            ticker=ticker,
            side=OrderSide.YES,
            action=OrderAction.SELL,
            count=count,
            type=OrderType.LIMIT,
            yes_price=price,
            client_order_id=client_order_id,
        )

    def sell_no(
        self,
        ticker: str,
        count: int,
        price: int,
        client_order_id: str | None = None,
    ) -> Order:
        """
        Convenience: Sell NO contracts at limit price.
        """
        return self.create_order(
            ticker=ticker,
            side=OrderSide.NO,
            action=OrderAction.SELL,
            count=count,
            type=OrderType.LIMIT,
            no_price=price,
            client_order_id=client_order_id,
        )

    def wait_for_fill(
        self,
        order_id: str,
        timeout: float = 30.0,
        poll_interval: float = 0.5,
    ) -> Order:
        """
        Wait for order to fill.

        Args:
            order_id: Order to monitor
            timeout: Max wait time in seconds
            poll_interval: Time between status checks

        Returns:
            Final order state

        Raises:
            TimeoutError: If order doesn't fill in time
        """
        start = time.monotonic()

        while time.monotonic() - start < timeout:
            order = self.get_order(order_id)

            if order.is_filled or order.is_canceled:
                return order

            time.sleep(poll_interval)

        # Timeout - return final state
        return self.get_order(order_id)

    def execute_and_wait(
        self,
        ticker: str,
        side: str | OrderSide,
        count: int,
        price: int,
        timeout: float = 30.0,
    ) -> Order:
        """
        Execute order and wait for fill.

        Convenience method for atomic execution.

        Args:
            ticker: Market ticker
            side: "yes" or "no"
            count: Number of contracts
            price: Limit price in cents
            timeout: Max wait time

        Returns:
            Filled order
        """
        side_enum = OrderSide(side) if isinstance(side, str) else side

        if side_enum == OrderSide.YES:
            order = self.buy_yes(ticker, count, price)
        else:
            order = self.buy_no(ticker, count, price)

        return self.wait_for_fill(order.order_id, timeout=timeout)

    def get_market_snapshot(self, ticker: str) -> dict:
        """
        Get combined market + position snapshot.

        Useful for execution decisions.

        Returns:
            Dict with market data and current position
        """
        market = self.get_market(ticker)
        position = self.get_position(ticker)

        return {
            "market": market,
            "position": position,
            "ticker": ticker,
            "yes_bid": market.yes_bid,
            "yes_ask": market.yes_ask,
            "no_bid": market.no_bid,
            "no_ask": market.no_ask,
            "spread": market.spread,
            "volume": market.volume,
            "is_active": market.is_active,
            "current_yes_count": position.yes_count if position else 0,
            "current_no_count": position.no_count if position else 0,
        }


# =============================================================================
# ASYNC CLIENT
# =============================================================================

class KalshiAsyncClient:
    """
    Asynchronous Kalshi API client.

    Same functionality as KalshiClient but with async/await support
    for high-concurrency scenarios.
    """

    def __init__(
        self,
        auth: KalshiAuth,
        base_url: str = KALSHI_PROD_URL,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_RETRIES,
    ):
        self.auth = auth
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._rate_limiter = RateLimiter()

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(
                max_keepalive_connections=5,
                max_connections=10,
                keepalive_expiry=30,
            ),
        )

    @classmethod
    def from_env(cls, demo: bool | None = None) -> KalshiAsyncClient:
        """Create client from environment variables."""
        auth = KalshiAuth.from_env()

        if demo is None:
            demo_env = os.environ.get("KALSHI_DEMO", "").lower()
            paper_env = os.environ.get("PAPER_TRADING", "").lower()
            demo = demo_env == "true" or paper_env == "true"

        if custom_url := os.environ.get("KALSHI_BASE_URL"):
            base_url = custom_url
        elif demo:
            base_url = KALSHI_DEMO_URL
        else:
            base_url = KALSHI_PROD_URL

        return cls(auth=auth, base_url=base_url)

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> KalshiAsyncClient:
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        json: dict | None = None,
        params: dict | None = None,
    ) -> dict:
        """Make an authenticated async API request with retries."""
        url = f"{self.base_url}{endpoint}"

        for attempt in range(self.max_retries):
            await self._rate_limiter.acquire()
            headers = self.auth.get_headers(method, endpoint)

            try:
                response = await self._client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=json,
                    params=params,
                )

                if response.status_code in (200, 201):
                    return response.json()
                elif response.status_code == 204:
                    return {}
                elif response.status_code in (401, 403):
                    raise AuthenticationError(
                        f"Authentication failed: {response.text}",
                        status_code=response.status_code,
                    )
                elif response.status_code == 404:
                    raise NotFoundError(
                        f"Resource not found: {endpoint}",
                        status_code=404,
                    )
                elif response.status_code == 429:
                    if attempt < self.max_retries - 1:
                        backoff = min(
                            RETRY_BACKOFF_BASE * (2 ** attempt),
                            MAX_RETRY_BACKOFF,
                        )
                        await asyncio.sleep(backoff)
                        continue
                    raise RateLimitError("Rate limit exceeded", status_code=429)
                elif response.status_code >= 500:
                    if attempt < self.max_retries - 1:
                        backoff = RETRY_BACKOFF_BASE * (2 ** attempt)
                        await asyncio.sleep(backoff)
                        continue
                    raise KalshiAPIError(
                        f"Server error: {response.text}",
                        status_code=response.status_code,
                    )
                else:
                    raise KalshiAPIError(
                        response.text,
                        status_code=response.status_code,
                    )

            except httpx.RequestError as e:
                if attempt < self.max_retries - 1:
                    backoff = RETRY_BACKOFF_BASE * (2 ** attempt)
                    await asyncio.sleep(backoff)
                    continue
                raise KalshiAPIError(f"Request failed: {e}")

        raise KalshiAPIError("Max retries exceeded")

    # Portfolio endpoints
    async def get_balance(self) -> Balance:
        data = await self._make_request("GET", "/portfolio/balance")
        return Balance(**data)

    async def get_positions(
        self,
        ticker: str | None = None,
        limit: int = 100,
    ) -> list[Position]:
        params: dict[str, Any] = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        data = await self._make_request("GET", "/portfolio/positions", params=params)
        return [Position(**p) for p in data.get("market_positions", [])]

    async def get_position(self, ticker: str) -> Position | None:
        positions = await self.get_positions(ticker=ticker, limit=1)
        return positions[0] if positions else None

    # Order endpoints
    async def create_order(
        self,
        ticker: str,
        side: str | OrderSide,
        action: str | OrderAction,
        count: int,
        type: str | OrderType = OrderType.LIMIT,
        yes_price: int | None = None,
        no_price: int | None = None,
        client_order_id: str | None = None,
    ) -> Order:
        side_str = side.value if isinstance(side, OrderSide) else side
        action_str = action.value if isinstance(action, OrderAction) else action
        type_str = type.value if isinstance(type, OrderType) else type

        body: dict[str, Any] = {
            "ticker": ticker,
            "side": side_str,
            "action": action_str,
            "count": count,
            "type": type_str,
            "client_order_id": client_order_id or str(uuid.uuid4()),
        }

        if yes_price is not None:
            body["yes_price"] = yes_price
        if no_price is not None:
            body["no_price"] = no_price

        data = await self._make_request("POST", "/portfolio/orders", json=body)
        return Order(**data.get("order", data))

    async def get_order(self, order_id: str) -> Order:
        data = await self._make_request("GET", f"/portfolio/orders/{order_id}")
        return Order(**data.get("order", data))

    async def cancel_order(self, order_id: str) -> Order:
        data = await self._make_request("DELETE", f"/portfolio/orders/{order_id}")
        return Order(**data.get("order", data))

    async def get_orders(
        self,
        ticker: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[Order]:
        params: dict[str, Any] = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if status:
            params["status"] = status
        data = await self._make_request("GET", "/portfolio/orders", params=params)
        return [Order(**o) for o in data.get("orders", [])]

    async def get_open_orders(self, ticker: str | None = None) -> list[Order]:
        return await self.get_orders(ticker=ticker, status="resting")

    # Market endpoints
    async def get_market(self, ticker: str) -> Market:
        data = await self._make_request("GET", f"/markets/{ticker}")
        return Market(**data.get("market", data))

    async def get_markets(
        self,
        event_ticker: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[Market]:
        params: dict[str, Any] = {"limit": limit}
        if event_ticker:
            params["event_ticker"] = event_ticker
        if status:
            params["status"] = status
        data = await self._make_request("GET", "/markets", params=params)
        return [Market(**m) for m in data.get("markets", [])]

    async def get_event(self, event_ticker: str) -> Event:
        data = await self._make_request("GET", f"/events/{event_ticker}")
        return Event(**data.get("event", data))

    # Convenience methods
    async def buy_yes(
        self,
        ticker: str,
        count: int,
        price: int,
        client_order_id: str | None = None,
    ) -> Order:
        return await self.create_order(
            ticker=ticker,
            side=OrderSide.YES,
            action=OrderAction.BUY,
            count=count,
            type=OrderType.LIMIT,
            yes_price=price,
            client_order_id=client_order_id,
        )

    async def buy_no(
        self,
        ticker: str,
        count: int,
        price: int,
        client_order_id: str | None = None,
    ) -> Order:
        return await self.create_order(
            ticker=ticker,
            side=OrderSide.NO,
            action=OrderAction.BUY,
            count=count,
            type=OrderType.LIMIT,
            no_price=price,
            client_order_id=client_order_id,
        )

    async def wait_for_fill(
        self,
        order_id: str,
        timeout: float = 30.0,
        poll_interval: float = 0.5,
    ) -> Order:
        start = time.monotonic()

        while time.monotonic() - start < timeout:
            order = await self.get_order(order_id)

            if order.is_filled or order.is_canceled:
                return order

            await asyncio.sleep(poll_interval)

        return await self.get_order(order_id)


# =============================================================================
# VALIDATION / TEST
# =============================================================================

def validate_client():
    """
    Validate the Kalshi client configuration and connectivity.

    Run with: python -m app.connectors.kalshi.client
    """
    print("=" * 60)
    print("KALSHI CLIENT VALIDATION")
    print("=" * 60)
    print()

    # Load environment
    from pathlib import Path
    env_file = Path(__file__).parent.parent.parent.parent / ".env"
    if env_file.exists():
        print(f"Loading environment from: {env_file}")
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()
        print()

    # Check environment
    print("Environment Configuration:")
    print(f"  KALSHI_KEY_ID: {'SET' if os.environ.get('KALSHI_KEY_ID') else 'NOT SET'}")
    print(f"  KALSHI_PRIVATE_KEY_PATH: {os.environ.get('KALSHI_PRIVATE_KEY_PATH', 'NOT SET')}")
    print(f"  KALSHI_BASE_URL: {os.environ.get('KALSHI_BASE_URL', 'NOT SET')}")
    print(f"  PAPER_TRADING: {os.environ.get('PAPER_TRADING', 'NOT SET')}")
    print()

    try:
        # Create client
        print("Creating KalshiClient...")
        client = KalshiClient.from_env()
        print(f"  Base URL: {client.base_url}")
        print(f"  Auth Key: {client.auth.key_id[:8]}...")
        print()

        # Test connection - get balance
        print("Testing API connection...")
        print("  GET /portfolio/balance...")

        try:
            balance = client.get_balance()
            print(f"  SUCCESS: Balance = ${balance.balance_dollars:.2f}")
            print()

            # Test market endpoint
            print("Testing market endpoint...")
            print("  GET /markets (limit=1)...")

            markets = client.get_markets(status="open", limit=1)
            if markets:
                market = markets[0]
                print(f"  SUCCESS: Found market {market.ticker}")
                print(f"    Title: {market.title[:50]}...")
                print(f"    Yes Bid/Ask: {market.yes_bid}c / {market.yes_ask}c")
                print(f"    Volume: {market.volume:,}")
            else:
                print("  No open markets found")

            print()
            print("VALIDATION PASSED")

        except AuthenticationError as e:
            print(f"  FAILED: Authentication error - {e}")
            print()
            print("VALIDATION FAILED - Check your API credentials")
            return False

        except RateLimitError as e:
            print(f"  WARNING: Rate limited - {e}")
            print()
            print("VALIDATION INCOMPLETE - Rate limited")
            return True

        except KalshiAPIError as e:
            print(f"  FAILED: API error - {e}")
            print()
            print("VALIDATION FAILED")
            return False

        finally:
            client.close()

        return True

    except ConfigurationError as e:
        print(f"Configuration Error: {e}")
        print()
        print("VALIDATION FAILED - Missing configuration")
        return False

    except Exception as e:
        print(f"Unexpected Error: {e}")
        print()
        print("VALIDATION FAILED")
        return False


if __name__ == "__main__":
    validate_client()
