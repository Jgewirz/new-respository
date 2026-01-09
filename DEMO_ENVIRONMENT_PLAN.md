# Kalshi Demo Environment Implementation Plan

## Overview

This plan outlines how to properly implement and test against Kalshi's demo environment. The codebase already has demo URL support (`KALSHI_DEMO_URL`), but needs proper demo credentials and validation tooling.

---

## Kalshi Demo Environment Details

| Property | Value |
|----------|-------|
| **Demo API URL** | `https://demo-api.kalshi.co/trade-api/v2` |
| **Demo Web Portal** | `https://demo.kalshi.com/` |
| **Production API URL** | `https://api.elections.kalshi.com/trade-api/v2` |
| **Credential Isolation** | Demo and prod credentials are completely separate |
| **Mock Funds** | Demo provides simulated trading capital |

---

## Current State Analysis

### What's Already Implemented
- `KALSHI_DEMO_URL` constant in `client.py:69`
- `from_env(demo=True)` parameter support
- Auto-detection via `KALSHI_DEMO=true` or `PAPER_TRADING=true` env vars
- URL switching logic in `from_env()` method (lines 423-436)

### What's Missing
1. **Demo credentials** - Current `.env` only has production credentials
2. **Demo account setup** - No demo account created yet
3. **Validation tooling** - No script to verify demo connectivity
4. **Integration tests** - Tests run against mock, not demo API
5. **Credential switching** - No easy way to switch between demo/prod keys

---

## Implementation Phases

### Phase 1: Demo Account Setup (Manual - 15 min)

**Steps:**
1. Navigate to https://demo.kalshi.com/
2. Create a new account (can use fake information)
3. Generate API credentials in account settings
4. Download the private key PEM file

**Deliverables:**
- Demo account email/password
- Demo API Key ID
- Demo private key file (`kalshi_demo_private_key.pem`)

---

### Phase 2: Environment Configuration

**File: `.env` additions**

```bash
# -----------------------------------------------------------------------------
# KALSHI DEMO CREDENTIALS (SEPARATE FROM PRODUCTION)
# -----------------------------------------------------------------------------
KALSHI_DEMO_KEY_ID=<your-demo-key-id>
KALSHI_DEMO_PRIVATE_KEY_PATH=./kalshi_demo_private_key.pem

# To use demo environment, set one of:
# KALSHI_DEMO=true
# OR
# PAPER_TRADING=true (already set)
```

**File: `.env.example` update**

Add demo credential placeholders for documentation.

---

### Phase 3: Client Enhancement

**File: `app/connectors/kalshi/client.py`**

Enhance `from_env()` to load demo-specific credentials when in demo mode:

```python
@classmethod
def from_env(cls, demo: bool | None = None) -> KalshiClient:
    """Create client from environment variables."""

    # Determine demo mode
    if demo is None:
        demo_env = os.environ.get("KALSHI_DEMO", "").lower()
        paper_env = os.environ.get("PAPER_TRADING", "").lower()
        demo = demo_env == "true" or paper_env == "true"

    # Load appropriate credentials
    if demo:
        # Try demo-specific credentials first
        demo_key_id = os.environ.get("KALSHI_DEMO_KEY_ID")
        demo_key_path = os.environ.get("KALSHI_DEMO_PRIVATE_KEY_PATH")

        if demo_key_id and demo_key_path:
            auth = KalshiAuth(
                key_id=demo_key_id,
                private_key_path=demo_key_path,
            )
            base_url = KALSHI_DEMO_URL
        else:
            # Fall back to production credentials with demo URL (for testing)
            auth = KalshiAuth.from_env()
            base_url = KALSHI_DEMO_URL
            logger.warning(
                "Using production credentials with demo URL. "
                "Set KALSHI_DEMO_KEY_ID and KALSHI_DEMO_PRIVATE_KEY_PATH for proper demo access."
            )
    else:
        auth = KalshiAuth.from_env()
        base_url = os.environ.get("KALSHI_BASE_URL", KALSHI_PROD_URL)

    return cls(auth=auth, base_url=base_url)
```

---

### Phase 4: Demo Validation Script

**File: `scripts/validate_demo.py`**

```python
#!/usr/bin/env python3
"""
Validate Kalshi Demo Environment Connectivity

Usage:
    python -m scripts.validate_demo

    # Or with explicit demo flag
    KALSHI_DEMO=true python -m scripts.validate_demo
"""

import os
import sys

# Force demo mode
os.environ["KALSHI_DEMO"] = "true"

from app.connectors.kalshi import KalshiClient


def main() -> int:
    """Validate demo environment connectivity."""
    print("=" * 60)
    print("KALSHI DEMO ENVIRONMENT VALIDATION")
    print("=" * 60)

    try:
        with KalshiClient.from_env(demo=True) as client:
            print(f"\n✓ Connected to: {client.base_url}")

            # Test 1: Get balance
            print("\n[1/4] Testing Balance Endpoint...")
            balance = client.get_balance()
            print(f"  ✓ Balance: ${balance.balance_dollars:,.2f}")

            # Test 2: List markets
            print("\n[2/4] Testing Markets Endpoint...")
            markets = client.list_markets(limit=5)
            print(f"  ✓ Found {len(markets)} markets")
            for m in markets[:3]:
                print(f"    - {m.ticker}: {m.title[:50]}...")

            # Test 3: Get specific market (if available)
            print("\n[3/4] Testing Single Market Fetch...")
            if markets:
                market = client.get_market(markets[0].ticker)
                print(f"  ✓ Market: {market.ticker}")
                print(f"    Yes Ask: {market.yes_ask}c")
                print(f"    No Ask: {market.no_ask}c")

            # Test 4: Get positions
            print("\n[4/4] Testing Positions Endpoint...")
            positions = client.list_positions()
            print(f"  ✓ Positions: {len(positions)}")

            print("\n" + "=" * 60)
            print("✓ ALL DEMO VALIDATION TESTS PASSED")
            print("=" * 60)
            return 0

    except Exception as e:
        print(f"\n✗ VALIDATION FAILED: {e}")
        print("\nTroubleshooting:")
        print("  1. Create demo account at https://demo.kalshi.com/")
        print("  2. Generate API credentials in demo account settings")
        print("  3. Set KALSHI_DEMO_KEY_ID and KALSHI_DEMO_PRIVATE_KEY_PATH")
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

---

### Phase 5: Integration Tests

**File: `tests/integration/test_kalshi_demo.py`**

```python
"""
Integration tests against Kalshi Demo Environment.

These tests require valid demo credentials and network access.
Skip with: pytest -m "not integration"
"""

import os
import pytest

# Only run if demo credentials are configured
DEMO_CONFIGURED = bool(
    os.environ.get("KALSHI_DEMO_KEY_ID") and
    os.environ.get("KALSHI_DEMO_PRIVATE_KEY_PATH")
)


@pytest.fixture
def demo_client():
    """Create demo client for tests."""
    os.environ["KALSHI_DEMO"] = "true"
    from app.connectors.kalshi import KalshiClient

    with KalshiClient.from_env(demo=True) as client:
        yield client


@pytest.mark.integration
@pytest.mark.skipif(not DEMO_CONFIGURED, reason="Demo credentials not configured")
class TestKalshiDemoIntegration:
    """Integration tests against Kalshi demo API."""

    def test_get_balance(self, demo_client):
        """Test balance retrieval."""
        balance = demo_client.get_balance()
        assert balance.balance >= 0
        assert isinstance(balance.balance_dollars, float)

    def test_list_markets(self, demo_client):
        """Test market listing."""
        markets = demo_client.list_markets(limit=10)
        assert isinstance(markets, list)
        # Demo should have markets
        assert len(markets) > 0

    def test_get_market_details(self, demo_client):
        """Test fetching specific market."""
        markets = demo_client.list_markets(limit=1)
        if markets:
            market = demo_client.get_market(markets[0].ticker)
            assert market.ticker == markets[0].ticker
            assert market.yes_ask is not None or market.yes_bid is not None

    def test_list_positions(self, demo_client):
        """Test position listing (may be empty)."""
        positions = demo_client.list_positions()
        assert isinstance(positions, list)

    @pytest.mark.slow
    def test_paper_order_lifecycle(self, demo_client):
        """Test order creation and cancellation in demo."""
        # Find an open market
        markets = demo_client.list_markets(status="open", limit=10)
        if not markets:
            pytest.skip("No open markets in demo")

        market = markets[0]

        # Create a limit order far from market price (won't fill)
        order = demo_client.create_order(
            ticker=market.ticker,
            side="yes",
            action="buy",
            count=1,
            type="limit",
            yes_price=1,  # 1 cent - won't fill
        )

        assert order.order_id is not None
        assert order.status in ["resting", "pending"]

        # Cancel the order
        cancelled = demo_client.cancel_order(order.order_id)
        assert cancelled
```

---

### Phase 6: CLI Demo Mode

**File: `app/cli/demo.py`**

```python
"""
Demo environment CLI commands.

Usage:
    python -m app.cli.demo status      # Check demo config
    python -m app.cli.demo validate    # Run validation
    python -m app.cli.demo balance     # Get demo balance
    python -m app.cli.demo markets     # List demo markets
"""

import click
import os


@click.group()
def cli():
    """Kalshi Demo Environment Commands."""
    pass


@cli.command()
def status():
    """Show demo configuration status."""
    demo_key = os.environ.get("KALSHI_DEMO_KEY_ID")
    demo_path = os.environ.get("KALSHI_DEMO_PRIVATE_KEY_PATH")

    click.echo("Demo Configuration Status:")
    click.echo(f"  KALSHI_DEMO_KEY_ID: {'✓ Set' if demo_key else '✗ Not set'}")
    click.echo(f"  KALSHI_DEMO_PRIVATE_KEY_PATH: {'✓ Set' if demo_path else '✗ Not set'}")

    if demo_path and os.path.exists(demo_path):
        click.echo(f"  Demo key file: ✓ Exists")
    elif demo_path:
        click.echo(f"  Demo key file: ✗ Not found at {demo_path}")


@cli.command()
def validate():
    """Validate demo environment connectivity."""
    os.environ["KALSHI_DEMO"] = "true"
    from scripts.validate_demo import main
    raise SystemExit(main())


@cli.command()
def balance():
    """Get demo account balance."""
    os.environ["KALSHI_DEMO"] = "true"
    from app.connectors.kalshi import KalshiClient

    with KalshiClient.from_env(demo=True) as client:
        bal = client.get_balance()
        click.echo(f"Demo Balance: ${bal.balance_dollars:,.2f}")


@cli.command()
@click.option("--limit", default=10, help="Number of markets to show")
def markets(limit):
    """List demo markets."""
    os.environ["KALSHI_DEMO"] = "true"
    from app.connectors.kalshi import KalshiClient

    with KalshiClient.from_env(demo=True) as client:
        markets = client.list_markets(limit=limit)
        for m in markets:
            click.echo(f"{m.ticker}: {m.title[:60]}")


if __name__ == "__main__":
    cli()
```

---

## File Summary

| File | Action | Purpose |
|------|--------|---------|
| `.env` | Modify | Add demo credential variables |
| `.env.example` | Modify | Document demo credentials |
| `app/connectors/kalshi/client.py` | Modify | Enhance `from_env()` for demo credentials |
| `scripts/validate_demo.py` | Create | Demo validation script |
| `tests/integration/test_kalshi_demo.py` | Create | Integration tests |
| `app/cli/demo.py` | Create | CLI commands for demo |
| `kalshi_demo_private_key.pem` | Create (manual) | Demo API private key |

---

## Execution Checklist

### Manual Steps (User Required)
- [ ] Create demo account at https://demo.kalshi.com/
- [ ] Generate API credentials in demo settings
- [ ] Download demo private key PEM file
- [ ] Add demo credentials to `.env`

### Automated Implementation
- [ ] Update `.env` and `.env.example` with demo vars
- [ ] Enhance `client.py` with demo credential loading
- [ ] Create `scripts/validate_demo.py`
- [ ] Create `tests/integration/test_kalshi_demo.py`
- [ ] Create `app/cli/demo.py`
- [ ] Run validation: `python -m scripts.validate_demo`

---

## Usage After Implementation

```bash
# Validate demo connectivity
python -m scripts.validate_demo

# Use demo CLI
python -m app.cli.demo status
python -m app.cli.demo balance
python -m app.cli.demo markets

# Run integration tests
pytest tests/integration/test_kalshi_demo.py -v

# Run main pipeline in demo mode
KALSHI_DEMO=true python -m app.services.arb_pipeline
```

---

## Sources

- [Kalshi Demo Environment Docs](https://docs.kalshi.com/getting_started/demo_env)
- [Kalshi Demo Account Help](https://help.kalshi.com/account/demo-account)
- [Kalshi API Guide](https://zuplo.com/learning-center/kalshi-api)
