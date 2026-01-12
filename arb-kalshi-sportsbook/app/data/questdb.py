"""
QuestDB client for high-performance time-series storage.

QuestDB is optimized for:
- Sub-millisecond writes via InfluxDB Line Protocol (ILP)
- Fast time-series queries with SQL
- Columnar storage with automatic partitioning

Connection methods:
- ILP (port 9009): High-speed ingestion (recommended for ticks/trades)
- PostgreSQL wire (port 8812): SQL queries
- REST API (port 9000): HTTP queries
"""

import os
import socket
from datetime import datetime, timezone
from typing import Optional
import psycopg2
from psycopg2.extras import RealDictCursor


# =============================================================================
# Configuration (from environment variables with defaults)
# =============================================================================

QUESTDB_ILP_HOST = os.environ.get("QUESTDB_ILP_HOST", "localhost")
QUESTDB_ILP_PORT = int(os.environ.get("QUESTDB_ILP_PORT", "9009"))
QUESTDB_PG_HOST = os.environ.get("QUESTDB_PG_HOST", "localhost")
QUESTDB_PG_PORT = int(os.environ.get("QUESTDB_PG_PORT", "8812"))
QUESTDB_PG_USER = os.environ.get("QUESTDB_PG_USER", "admin")
QUESTDB_PG_PASSWORD = os.environ.get("QUESTDB_PG_PASSWORD", "quest")
QUESTDB_PG_DATABASE = os.environ.get("QUESTDB_PG_DATABASE", "qdb")


# =============================================================================
# ILP Client (High-Speed Ingestion)
# =============================================================================

class QuestDBILPClient:
    """
    InfluxDB Line Protocol client for high-speed writes.

    Line protocol format:
        table_name,tag1=val1,tag2=val2 field1=value1,field2=value2 timestamp_ns

    Example:
        kalshi_ticks,ticker=KXBTC-25JAN10-97000 yes_bid=45i,yes_ask=46i 1704672000000000000
    """

    def __init__(self, host: str = QUESTDB_ILP_HOST, port: int = QUESTDB_ILP_PORT):
        self.host = host
        self.port = port
        self.sock: Optional[socket.socket] = None

    def connect(self):
        """Establish TCP connection to QuestDB ILP endpoint."""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))

    def close(self):
        """Close the connection."""
        if self.sock:
            self.sock.close()
            self.sock = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _send(self, line: str):
        """Send a line to QuestDB."""
        if not self.sock:
            raise RuntimeError("Not connected. Call connect() first.")
        self.sock.sendall((line + "\n").encode("utf-8"))

    def write_tick(
        self,
        ticker: str,
        yes_bid: int,
        yes_ask: int,
        no_bid: int,
        no_ask: int,
        volume: int,
        series: str = "",
        sport: str = "",
        timestamp_ns: Optional[int] = None
    ):
        """
        Write a market tick to QuestDB.

        Args:
            ticker: Market ticker (e.g., KXNFL-25JAN12-BUF)
            yes_bid/yes_ask: Yes side prices (cents, 0-100)
            no_bid/no_ask: No side prices (cents, 0-100)
            volume: Trade volume
            series: Series ticker
            sport: Sport category
            timestamp_ns: Nanosecond timestamp (default: now)
        """
        if timestamp_ns is None:
            timestamp_ns = int(datetime.now(timezone.utc).timestamp() * 1e9)

        # Build tags
        tags = f"ticker={ticker}"
        if series:
            tags += f",series={series}"
        if sport:
            tags += f",sport={sport}"

        # Build fields (integers need 'i' suffix)
        fields = f"yes_bid={yes_bid}i,yes_ask={yes_ask}i,no_bid={no_bid}i,no_ask={no_ask}i,volume={volume}i"

        line = f"kalshi_ticks,{tags} {fields} {timestamp_ns}"
        self._send(line)

    def write_trade(
        self,
        ticker: str,
        price: int,
        count: int,
        side: str,
        taker_side: str = "",
        timestamp_ns: Optional[int] = None
    ):
        """
        Write a trade to QuestDB.

        Args:
            ticker: Market ticker
            price: Trade price (cents)
            count: Number of contracts
            side: "yes" or "no"
            taker_side: "buy" or "sell"
            timestamp_ns: Nanosecond timestamp
        """
        if timestamp_ns is None:
            timestamp_ns = int(datetime.now(timezone.utc).timestamp() * 1e9)

        tags = f"ticker={ticker},side={side}"
        if taker_side:
            tags += f",taker_side={taker_side}"

        fields = f"price={price}i,count={count}i"

        line = f"kalshi_trades,{tags} {fields} {timestamp_ns}"
        self._send(line)

    def write_orderbook_delta(
        self,
        ticker: str,
        side: str,
        price: int,
        delta: int,
        timestamp_ns: Optional[int] = None
    ):
        """
        Write an orderbook delta to QuestDB.

        Args:
            ticker: Market ticker
            side: "yes" or "no"
            price: Price level (cents)
            delta: Change in quantity (positive = add, negative = remove)
            timestamp_ns: Nanosecond timestamp
        """
        if timestamp_ns is None:
            timestamp_ns = int(datetime.now(timezone.utc).timestamp() * 1e9)

        tags = f"ticker={ticker},side={side}"
        fields = f"price={price}i,delta={delta}i"

        line = f"kalshi_orderbook,{tags} {fields} {timestamp_ns}"
        self._send(line)

    def write_market_snapshot(
        self,
        market_ticker: str,
        series_ticker: str,
        event_ticker: str,
        title: str,
        subtitle: str,
        status: str,
        yes_bid: int,
        yes_ask: int,
        no_bid: int,
        no_ask: int,
        last_price: int,
        volume: int,
        volume_24h: int,
        liquidity: int,
        open_time_ns: Optional[int] = None,
        close_time_ns: Optional[int] = None,
        expiration_time_ns: Optional[int] = None,
        fetched_at_ns: Optional[int] = None,
        timestamp_ns: Optional[int] = None
    ):
        """
        Write a market snapshot to QuestDB (from JSONL ingestion).

        Args:
            market_ticker: Market ticker (e.g., KXNFL-25JAN12-BUF)
            series_ticker: Series ticker
            event_ticker: Event ticker
            title: Market title
            subtitle: Market subtitle
            status: Market status (active, closed, etc.)
            yes_bid/yes_ask: Yes side prices (cents, 0-100)
            no_bid/no_ask: No side prices (cents, 0-100)
            last_price: Last trade price
            volume: Total volume
            volume_24h: 24-hour volume
            liquidity: Market liquidity
            open_time_ns: Market open time (nanoseconds)
            close_time_ns: Market close time (nanoseconds)
            expiration_time_ns: Market expiration time (nanoseconds)
            fetched_at_ns: When data was fetched (nanoseconds)
            timestamp_ns: Record timestamp (nanoseconds, default: now)
        """
        if timestamp_ns is None:
            timestamp_ns = int(datetime.now(timezone.utc).timestamp() * 1e9)

        # Build tags (SYMBOL columns)
        tags = f"market_ticker={self._escape_tag(market_ticker)}"
        tags += f",series_ticker={self._escape_tag(series_ticker)}"
        tags += f",event_ticker={self._escape_tag(event_ticker)}"
        tags += f",status={self._escape_tag(status)}"

        # Build fields
        fields = []
        # String fields need quotes
        fields.append(f'title="{self._escape_string(title)}"')
        fields.append(f'subtitle="{self._escape_string(subtitle)}"')
        # Integer fields
        fields.append(f"yes_bid={yes_bid}i")
        fields.append(f"yes_ask={yes_ask}i")
        fields.append(f"no_bid={no_bid}i")
        fields.append(f"no_ask={no_ask}i")
        fields.append(f"last_price={last_price}i")
        fields.append(f"volume={volume}i")
        fields.append(f"volume_24h={volume_24h}i")
        fields.append(f"liquidity={liquidity}i")
        # Timestamp fields (as integers)
        if open_time_ns:
            fields.append(f"open_time={open_time_ns}t")
        if close_time_ns:
            fields.append(f"close_time={close_time_ns}t")
        if expiration_time_ns:
            fields.append(f"expiration_time={expiration_time_ns}t")
        if fetched_at_ns:
            fields.append(f"fetched_at={fetched_at_ns}t")

        line = f"kalshi_markets,{tags} {','.join(fields)} {timestamp_ns}"
        self._send(line)

    @staticmethod
    def _escape_tag(value: str) -> str:
        """Escape special characters in tag values."""
        if not value:
            return "none"
        # Tags cannot contain spaces, commas, or equals signs
        return value.replace(" ", "\\ ").replace(",", "\\,").replace("=", "\\=")

    @staticmethod
    def _escape_string(value: str) -> str:
        """Escape special characters in string field values."""
        if not value:
            return ""
        # Strings need escaped quotes and backslashes
        return value.replace("\\", "\\\\").replace('"', '\\"')

    def flush(self):
        """Flush is implicit with TCP, but included for API compatibility."""
        pass


# =============================================================================
# PostgreSQL Client (SQL Queries)
# =============================================================================

class QuestDBClient:
    """
    PostgreSQL wire protocol client for SQL queries.

    Use this for:
    - Creating tables
    - Running analytical queries
    - Reading historical data
    """

    def __init__(
        self,
        host: str = QUESTDB_PG_HOST,
        port: int = QUESTDB_PG_PORT,
        user: str = QUESTDB_PG_USER,
        password: str = QUESTDB_PG_PASSWORD,
        database: str = QUESTDB_PG_DATABASE
    ):
        self.conn_params = {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "database": database
        }
        self.conn = None

    def connect(self):
        """Establish PostgreSQL connection."""
        self.conn = psycopg2.connect(**self.conn_params)
        self.conn.autocommit = True

    def close(self):
        """Close the connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def execute(self, sql: str, params: tuple = None) -> list[dict]:
        """Execute SQL and return results as list of dicts."""
        if not self.conn:
            raise RuntimeError("Not connected. Call connect() first.")

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            if cur.description:
                return [dict(row) for row in cur.fetchall()]
            return []

    def execute_ddl(self, sql: str):
        """Execute DDL statement (CREATE, DROP, etc.)."""
        if not self.conn:
            raise RuntimeError("Not connected. Call connect() first.")

        with self.conn.cursor() as cur:
            cur.execute(sql)


# =============================================================================
# Schema Setup
# =============================================================================

def create_tables():
    """Create QuestDB tables for Kalshi market data."""

    with QuestDBClient() as client:
        # Ticks table - partitioned by day
        client.execute_ddl("""
            CREATE TABLE IF NOT EXISTS kalshi_ticks (
                ticker SYMBOL,
                series SYMBOL,
                sport SYMBOL,
                yes_bid INT,
                yes_ask INT,
                no_bid INT,
                no_ask INT,
                volume LONG,
                timestamp TIMESTAMP
            ) TIMESTAMP(timestamp) PARTITION BY DAY WAL
            DEDUP UPSERT KEYS(timestamp, ticker);
        """)

        # Trades table
        client.execute_ddl("""
            CREATE TABLE IF NOT EXISTS kalshi_trades (
                ticker SYMBOL,
                side SYMBOL,
                taker_side SYMBOL,
                price INT,
                count INT,
                timestamp TIMESTAMP
            ) TIMESTAMP(timestamp) PARTITION BY DAY WAL;
        """)

        # Orderbook deltas
        client.execute_ddl("""
            CREATE TABLE IF NOT EXISTS kalshi_orderbook (
                ticker SYMBOL,
                side SYMBOL,
                price INT,
                delta INT,
                timestamp TIMESTAMP
            ) TIMESTAMP(timestamp) PARTITION BY DAY WAL;
        """)

        # Sportsbook odds (for arbitrage comparison)
        client.execute_ddl("""
            CREATE TABLE IF NOT EXISTS sportsbook_odds (
                event_id SYMBOL,
                book SYMBOL,
                market_type SYMBOL,
                outcome SYMBOL,
                odds_decimal DOUBLE,
                implied_prob DOUBLE,
                timestamp TIMESTAMP
            ) TIMESTAMP(timestamp) PARTITION BY DAY WAL;
        """)

        # Arbitrage opportunities detected
        client.execute_ddl("""
            CREATE TABLE IF NOT EXISTS arb_opportunities (
                kalshi_ticker SYMBOL,
                sportsbook_event SYMBOL,
                book SYMBOL,
                arb_type SYMBOL,
                kalshi_prob DOUBLE,
                book_prob DOUBLE,
                edge_pct DOUBLE,
                kalshi_side SYMBOL,
                book_side SYMBOL,
                timestamp TIMESTAMP
            ) TIMESTAMP(timestamp) PARTITION BY DAY WAL;
        """)

        # Kalshi market snapshots (from JSONL ingestion)
        client.execute_ddl("""
            CREATE TABLE IF NOT EXISTS kalshi_markets (
                market_ticker SYMBOL,
                series_ticker SYMBOL,
                event_ticker SYMBOL,
                title STRING,
                subtitle STRING,
                status SYMBOL,
                yes_bid INT,
                yes_ask INT,
                no_bid INT,
                no_ask INT,
                last_price INT,
                volume LONG,
                volume_24h LONG,
                liquidity LONG,
                open_time TIMESTAMP,
                close_time TIMESTAMP,
                expiration_time TIMESTAMP,
                fetched_at TIMESTAMP,
                timestamp TIMESTAMP
            ) TIMESTAMP(timestamp) PARTITION BY DAY WAL
            DEDUP UPSERT KEYS(timestamp, market_ticker);
        """)

        print("QuestDB tables created successfully")


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Create tables
    print("Creating QuestDB tables...")
    create_tables()

    # Test ILP ingestion
    print("\nTesting ILP ingestion...")
    with QuestDBILPClient() as ilp:
        # Write some sample ticks
        ilp.write_tick(
            ticker="KXNFL-25JAN12-BUF",
            yes_bid=45,
            yes_ask=47,
            no_bid=53,
            no_ask=55,
            volume=1000,
            series="KXNFL-25JAN12",
            sport="nfl"
        )
        ilp.write_tick(
            ticker="KXNBA-25JAN12-LAL",
            yes_bid=62,
            yes_ask=64,
            no_bid=36,
            no_ask=38,
            volume=500,
            series="KXNBA-25JAN12",
            sport="nba"
        )
        print("Wrote 2 sample ticks")

    # Query the data
    print("\nQuerying data...")
    with QuestDBClient() as client:
        results = client.execute("""
            SELECT * FROM kalshi_ticks
            ORDER BY timestamp DESC
            LIMIT 10
        """)
        for row in results:
            print(f"  {row}")
