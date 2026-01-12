"""
Full Integration Test - Kalshi Arbitrage System

Tests the complete pipeline with real Kalshi API data:
1. REST API market discovery
2. WebSocket connection and subscription
3. Real-time message processing
4. Processor -> Detector flow
5. Latency measurement

Usage:
    cd arb-kalshi-sportsbook
    python -m scripts.integration_test
"""

import os
import sys
import asyncio
import json
import time
import base64
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())

os.environ["KALSHI_ENV"] = "demo"

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
import aiohttp

from app.connectors.kalshi.ws_processor import WebSocketProcessor
from app.services.realtime_detector import RealtimeDetector


@dataclass
class SystemStats:
    """Track system-wide statistics."""
    messages_received: int = 0
    tickers_processed: int = 0
    trades_processed: int = 0
    latencies: list = field(default_factory=list)
    start_time: float = 0


def create_signer(key_id: str, private_key):
    """Create signing functions for REST and WebSocket auth."""

    def sign_ws():
        ts = str(int(time.time() * 1000))
        msg = f"{ts}GET/trade-api/ws/v2".encode()
        sig = private_key.sign(
            msg,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256()
        )
        return {
            "KALSHI-ACCESS-KEY": key_id,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
            "KALSHI-ACCESS-TIMESTAMP": ts,
        }

    def sign_rest(method: str, path: str):
        ts = str(int(time.time() * 1000))
        full_path = f"/trade-api/v2{path}" if not path.startswith("/trade-api") else path
        msg = f"{ts}{method}{full_path}".encode()
        sig = private_key.sign(
            msg,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256()
        )
        return {
            "KALSHI-ACCESS-KEY": key_id,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "Content-Type": "application/json",
        }

    return sign_ws, sign_rest


async def run_integration_test(duration: int = 30):
    """Run the full integration test."""
    print("=" * 70)
    print("KALSHI ARBITRAGE SYSTEM - FULL INTEGRATION TEST")
    print("=" * 70)
    print()

    # Load credentials
    key_id = os.environ.get("KALSHI_KEY_ID")
    key_path = os.environ.get("KALSHI_PRIVATE_KEY_PATH", "./kalshi_demo_private_key.pem")

    if not key_id:
        print("ERROR: KALSHI_KEY_ID not set")
        return

    with open(key_path, "rb") as f:
        private_key = serialization.load_pem_private_key(
            f.read(), password=None, backend=default_backend()
        )

    sign_ws, sign_rest = create_signer(key_id, private_key)

    base_url = "https://demo-api.kalshi.co/trade-api/v2"
    ws_url = "wss://demo-api.kalshi.co/trade-api/ws/v2"

    # Initialize components
    processor = WebSocketProcessor(write_questdb=False, write_redis=False)
    detector = RealtimeDetector.from_env()
    processor.on_ticker(detector.on_market_update)

    stats = SystemStats()

    async with aiohttp.ClientSession() as session:
        # Phase 1: Market Discovery
        print("Phase 1: Market Discovery")
        print("-" * 70)

        markets = []
        for series in ["KXNBA", "KXMLB", "KXNHL"]:
            path = f"/markets?series_ticker={series}&status=open&limit=20"
            headers = sign_rest("GET", "/markets")

            async with session.get(base_url + path, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    for m in data.get("markets", []):
                        markets.append(m)
                    print(f"  {series}: {len(data.get('markets', []))} markets")

        print(f"  Total: {len(markets)} sports markets discovered")

        if not markets:
            print("ERROR: No markets found")
            return

        # Populate detector with market metadata
        detector.populate_markets(markets)
        detector.start()

        # Phase 2: WebSocket Connection
        print()
        print("Phase 2: WebSocket Connection")
        print("-" * 70)

        headers = sign_ws()
        tickers = [m["ticker"] for m in markets[:30]]

        try:
            async with session.ws_connect(ws_url, headers=headers, heartbeat=30) as ws:
                print("  WebSocket connected successfully")

                # Subscribe
                await ws.send_json({
                    "id": 1,
                    "cmd": "subscribe",
                    "params": {
                        "channels": ["ticker", "trade"],
                        "market_tickers": tickers,
                    }
                })
                print(f"  Subscribed to {len(tickers)} markets")

                # Phase 3: Real-Time Processing
                print()
                print(f"Phase 3: Real-Time Processing ({duration} seconds)")
                print("-" * 70)

                stats.start_time = time.time()

                async for msg in ws:
                    elapsed = time.time() - stats.start_time
                    if elapsed > duration:
                        break

                    if msg.type == aiohttp.WSMsgType.TEXT:
                        recv_time = time.time()
                        stats.messages_received += 1

                        data = json.loads(msg.data)
                        msg_type = data.get("type", "")

                        if msg_type == "ticker":
                            m = data.get("msg", {})
                            ticker = m.get("market_ticker", "")

                            # Time the processing
                            process_start = time.time()
                            processor.process_ticker(m)
                            process_time = (time.time() - process_start) * 1000

                            stats.tickers_processed += 1
                            stats.latencies.append(process_time)

                            # Print periodic updates
                            if stats.tickers_processed % 10 == 1:
                                yes_bid = m.get("yes_bid", 0)
                                yes_ask = m.get("yes_ask", 0)
                                print(f"  [{elapsed:5.1f}s] {ticker}: bid={yes_bid}c ask={yes_ask}c | latency={process_time:.2f}ms")

                        elif msg_type == "trade":
                            m = data.get("msg", {})
                            processor.process_trade(m)
                            stats.trades_processed += 1

                        elif msg_type == "subscribed":
                            print("  Subscription confirmed")

                        elif msg_type == "error":
                            print(f"  ERROR: {data}")

        except Exception as e:
            print(f"  WebSocket error: {e}")
            return

    # Results
    print()
    print("=" * 70)
    print("INTEGRATION TEST RESULTS")
    print("=" * 70)
    print()

    total_time = time.time() - stats.start_time

    print("Message Statistics:")
    print(f"  Total messages received: {stats.messages_received}")
    print(f"  Ticker updates processed: {stats.tickers_processed}")
    print(f"  Trade messages processed: {stats.trades_processed}")
    print(f"  Messages/second: {stats.messages_received / total_time:.1f}")
    print()

    if stats.latencies:
        avg_lat = sum(stats.latencies) / len(stats.latencies)
        max_lat = max(stats.latencies)
        min_lat = min(stats.latencies)
        sorted_lat = sorted(stats.latencies)
        p50_lat = sorted_lat[len(sorted_lat) // 2]
        p95_idx = int(len(sorted_lat) * 0.95)
        p95_lat = sorted_lat[p95_idx] if p95_idx < len(sorted_lat) else max_lat

        print("Latency Statistics (message processing):")
        print(f"  Average: {avg_lat:.3f}ms")
        print(f"  P50:     {p50_lat:.3f}ms")
        print(f"  P95:     {p95_lat:.3f}ms")
        print(f"  Min:     {min_lat:.3f}ms")
        print(f"  Max:     {max_lat:.3f}ms")
        print()

    print("Processor Statistics:")
    print(f"  {processor.stats.summary()}")
    print()

    print("Detector Statistics:")
    print(f"  {detector.stats.summary()}")
    print()

    # Validation
    print("=" * 70)
    print("VALIDATION")
    print("=" * 70)

    issues = []

    if stats.tickers_processed == 0:
        issues.append("No ticker messages processed")

    if stats.latencies:
        avg_lat = sum(stats.latencies) / len(stats.latencies)
        if avg_lat > 10:
            issues.append(f"Average latency {avg_lat:.1f}ms exceeds 10ms target")

    if detector.stats.resolver_misses == detector.stats.detections_run and detector.stats.detections_run > 0:
        issues.append("All resolver lookups failed (expected - no Redis consensus data)")

    if issues:
        print()
        print("Issues Found:")
        for issue in issues:
            print(f"  - {issue}")

    print()
    if stats.tickers_processed > 0 and (not stats.latencies or sum(stats.latencies) / len(stats.latencies) < 10):
        print("SYSTEM STATUS: OPERATIONAL")
        print()
        print("  [OK] Real-time data flowing through pipeline")
        print("  [OK] Sub-10ms processing latency achieved")
        print("  [OK] WebSocket -> Processor -> Detector integration working")
        print("  [--] Signals require Redis consensus data (expected in dry run)")
    else:
        print("SYSTEM STATUS: NEEDS ATTENTION")

    detector.stop()


if __name__ == "__main__":
    import sys
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    asyncio.run(run_integration_test(duration))
