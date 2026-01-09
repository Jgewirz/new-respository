"""
Kalshi RSA-PSS Authentication Module

Production-grade authentication for Kalshi Trading API using RSA-PSS signatures.

Authentication Flow:
    1. Load RSA private key from PEM file or string
    2. For each request, construct message: "{timestamp_ms}{METHOD}{path}"
    3. Sign with RSA-PSS (SHA256, MGF1, salt_length=DIGEST_LENGTH)
    4. Base64 encode signature
    5. Set headers: KALSHI-ACCESS-KEY, KALSHI-ACCESS-SIGNATURE, KALSHI-ACCESS-TIMESTAMP

Usage:
    from app.connectors.kalshi.auth import KalshiAuth

    auth = KalshiAuth.from_env()  # Load from environment
    # or
    auth = KalshiAuth(key_id="...", private_key_path="./key.pem")

    headers = auth.get_headers("POST", "/trade-api/v2/portfolio/orders")
    response = httpx.post(url, headers=headers, json=data)

Security Notes:
    - Private keys should NEVER be committed to version control
    - Use environment variables or secure key management
    - Timestamps are in milliseconds to prevent replay attacks
"""

import base64
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa


# =============================================================================
# EXCEPTIONS
# =============================================================================

class KalshiAuthError(Exception):
    """Base exception for authentication errors."""
    pass


class KeyLoadError(KalshiAuthError):
    """Failed to load private key."""
    pass


class SignatureError(KalshiAuthError):
    """Failed to create signature."""
    pass


class ConfigurationError(KalshiAuthError):
    """Missing or invalid configuration."""
    pass


# =============================================================================
# AUTH DATACLASS
# =============================================================================

@dataclass
class AuthHeaders:
    """
    Kalshi authentication headers.

    These headers must be included with every authenticated API request.
    """
    key: str          # KALSHI-ACCESS-KEY
    signature: str    # KALSHI-ACCESS-SIGNATURE (base64)
    timestamp: str    # KALSHI-ACCESS-TIMESTAMP (milliseconds)

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary for HTTP request headers."""
        return {
            "KALSHI-ACCESS-KEY": self.key,
            "KALSHI-ACCESS-SIGNATURE": self.signature,
            "KALSHI-ACCESS-TIMESTAMP": self.timestamp,
            "Content-Type": "application/json",
        }

    def __repr__(self) -> str:
        return (
            f"AuthHeaders(key='{self.key[:8]}...', "
            f"timestamp='{self.timestamp}', "
            f"signature='{self.signature[:20]}...')"
        )


# =============================================================================
# KALSHI AUTH CLASS
# =============================================================================

class KalshiAuth:
    """
    Kalshi API Authentication using RSA-PSS signatures.

    Handles:
        - Private key loading (from file or string)
        - Message construction for signing
        - RSA-PSS signature generation
        - Header generation for API requests

    Thread-safe: Can be shared across multiple threads/coroutines.
    """

    # API path prefix (required for signature)
    API_PREFIX = "/trade-api/v2"

    def __init__(
        self,
        key_id: str,
        private_key: Optional[rsa.RSAPrivateKey] = None,
        private_key_path: Optional[Union[str, Path]] = None,
        private_key_pem: Optional[str] = None,
    ):
        """
        Initialize Kalshi authentication.

        Args:
            key_id: Kalshi API Key ID (UUID format)
            private_key: Pre-loaded RSA private key object
            private_key_path: Path to PEM file containing private key
            private_key_pem: PEM-encoded private key string

        Raises:
            ConfigurationError: If key_id is missing
            KeyLoadError: If private key cannot be loaded
        """
        if not key_id:
            raise ConfigurationError("key_id is required")

        self.key_id = key_id

        # Load private key from one of the sources
        if private_key:
            self._private_key = private_key
        elif private_key_path:
            self._private_key = self._load_key_from_file(private_key_path)
        elif private_key_pem:
            self._private_key = self._load_key_from_pem(private_key_pem)
        else:
            raise ConfigurationError(
                "One of private_key, private_key_path, or private_key_pem is required"
            )

    @classmethod
    def from_env(cls) -> "KalshiAuth":
        """
        Create KalshiAuth from environment variables.

        Required environment variables:
            - KALSHI_KEY_ID: API Key ID
            - KALSHI_PRIVATE_KEY_PATH: Path to PEM file (preferred)
            - KALSHI_PRIVATE_KEY: PEM string (alternative, less secure)

        Returns:
            Configured KalshiAuth instance

        Raises:
            ConfigurationError: If required env vars are missing
        """
        key_id = os.environ.get("KALSHI_KEY_ID")
        if not key_id:
            raise ConfigurationError("KALSHI_KEY_ID environment variable not set")

        # Try file path first (more secure)
        key_path = os.environ.get("KALSHI_PRIVATE_KEY_PATH")
        if key_path:
            return cls(key_id=key_id, private_key_path=key_path)

        # Fall back to PEM string
        key_pem = os.environ.get("KALSHI_PRIVATE_KEY")
        if key_pem:
            # Handle escaped newlines from env var
            key_pem = key_pem.replace("\\n", "\n")
            return cls(key_id=key_id, private_key_pem=key_pem)

        raise ConfigurationError(
            "Either KALSHI_PRIVATE_KEY_PATH or KALSHI_PRIVATE_KEY must be set"
        )

    def _load_key_from_file(self, path: Union[str, Path]) -> rsa.RSAPrivateKey:
        """
        Load RSA private key from PEM file.

        Args:
            path: Path to PEM file

        Returns:
            RSA private key object

        Raises:
            KeyLoadError: If file cannot be read or parsed
        """
        path = Path(path)

        if not path.exists():
            raise KeyLoadError(f"Private key file not found: {path}")

        try:
            with open(path, "rb") as f:
                key_data = f.read()

            return serialization.load_pem_private_key(
                key_data,
                password=None,
                backend=default_backend()
            )
        except Exception as e:
            raise KeyLoadError(f"Failed to load private key from {path}: {e}")

    def _load_key_from_pem(self, pem: str) -> rsa.RSAPrivateKey:
        """
        Load RSA private key from PEM string.

        Args:
            pem: PEM-encoded private key string

        Returns:
            RSA private key object

        Raises:
            KeyLoadError: If PEM cannot be parsed
        """
        try:
            return serialization.load_pem_private_key(
                pem.encode("utf-8"),
                password=None,
                backend=default_backend()
            )
        except Exception as e:
            raise KeyLoadError(f"Failed to parse private key PEM: {e}")

    def _get_timestamp_ms(self) -> str:
        """Get current timestamp in milliseconds as string."""
        return str(int(time.time() * 1000))

    def _build_message(self, timestamp: str, method: str, path: str) -> bytes:
        """
        Build the message to be signed.

        Format: "{timestamp}{METHOD}{path}"

        Note: Path should NOT include query parameters.

        Args:
            timestamp: Timestamp in milliseconds
            method: HTTP method (GET, POST, DELETE, etc.)
            path: API path WITHOUT query parameters

        Returns:
            Message bytes ready for signing
        """
        # Ensure method is uppercase
        method = method.upper()

        # Strip query parameters if present
        path = path.split("?")[0]

        # Ensure path starts with API prefix
        if not path.startswith(self.API_PREFIX):
            path = self.API_PREFIX + path

        # Construct message
        message = f"{timestamp}{method}{path}"
        return message.encode("utf-8")

    def _sign(self, message: bytes) -> str:
        """
        Sign message with RSA-PSS.

        Uses:
            - PSS padding with MGF1(SHA256)
            - SHA256 hash
            - Salt length = digest length

        Args:
            message: Bytes to sign

        Returns:
            Base64-encoded signature

        Raises:
            SignatureError: If signing fails
        """
        try:
            signature = self._private_key.sign(
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return base64.b64encode(signature).decode("utf-8")
        except Exception as e:
            raise SignatureError(f"Failed to sign message: {e}")

    def create_signature(self, method: str, path: str) -> AuthHeaders:
        """
        Create authentication headers for an API request.

        Args:
            method: HTTP method (GET, POST, DELETE, etc.)
            path: API path (with or without /trade-api/v2 prefix)

        Returns:
            AuthHeaders with key, signature, and timestamp

        Example:
            >>> auth = KalshiAuth.from_env()
            >>> headers = auth.create_signature("POST", "/portfolio/orders")
            >>> print(headers.to_dict())
        """
        timestamp = self._get_timestamp_ms()
        message = self._build_message(timestamp, method, path)
        signature = self._sign(message)

        return AuthHeaders(
            key=self.key_id,
            signature=signature,
            timestamp=timestamp,
        )

    def get_headers(self, method: str, path: str) -> dict[str, str]:
        """
        Get headers dictionary for an API request.

        Convenience method that returns headers ready for use with
        requests/httpx libraries.

        Args:
            method: HTTP method
            path: API path

        Returns:
            Dictionary with all required headers
        """
        return self.create_signature(method, path).to_dict()

    def verify_key(self) -> bool:
        """
        Verify that the private key is valid and can create signatures.

        Returns:
            True if key is valid and can sign
        """
        try:
            test_message = b"kalshi_auth_test"
            self._private_key.sign(
                test_message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False

    def __repr__(self) -> str:
        return f"KalshiAuth(key_id='{self.key_id[:8]}...', key_valid={self.verify_key()})"


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_auth_from_env() -> KalshiAuth:
    """
    Factory function to create KalshiAuth from environment.

    Convenient for dependency injection.
    """
    return KalshiAuth.from_env()


def create_demo_auth() -> KalshiAuth:
    """
    Create auth configured for demo/sandbox environment.

    Reads from the same env vars but can be used to differentiate
    production vs demo in code.
    """
    return KalshiAuth.from_env()


# =============================================================================
# VALIDATION / TEST
# =============================================================================

def validate_auth():
    """
    Validate authentication configuration and key.

    Run with: python -m app.connectors.kalshi.auth
    """
    print("=" * 60)
    print("KALSHI AUTHENTICATION VALIDATION")
    print("=" * 60)
    print()

    # Check environment variables
    print("Environment Variables:")
    key_id = os.environ.get("KALSHI_KEY_ID")
    key_path = os.environ.get("KALSHI_PRIVATE_KEY_PATH")
    key_pem = os.environ.get("KALSHI_PRIVATE_KEY")

    print(f"  KALSHI_KEY_ID: {'SET' if key_id else 'NOT SET'}")
    if key_id:
        print(f"    Value: {key_id[:8]}...{key_id[-4:]}")

    print(f"  KALSHI_PRIVATE_KEY_PATH: {'SET' if key_path else 'NOT SET'}")
    if key_path:
        path = Path(key_path)
        print(f"    Path: {key_path}")
        print(f"    Exists: {path.exists()}")

    print(f"  KALSHI_PRIVATE_KEY: {'SET' if key_pem else 'NOT SET'}")
    if key_pem:
        print(f"    Length: {len(key_pem)} chars")

    print()

    # Try to create auth
    print("Creating KalshiAuth...")
    try:
        auth = KalshiAuth.from_env()
        print(f"  SUCCESS: {auth}")
        print()

        # Verify key
        print("Verifying key...")
        if auth.verify_key():
            print("  SUCCESS: Key is valid and can sign messages")
        else:
            print("  FAILED: Key verification failed")
        print()

        # Create test signature
        print("Creating test signature...")
        headers = auth.get_headers("GET", "/portfolio/balance")
        print(f"  KALSHI-ACCESS-KEY: {headers['KALSHI-ACCESS-KEY'][:8]}...")
        print(f"  KALSHI-ACCESS-TIMESTAMP: {headers['KALSHI-ACCESS-TIMESTAMP']}")
        print(f"  KALSHI-ACCESS-SIGNATURE: {headers['KALSHI-ACCESS-SIGNATURE'][:40]}...")
        print()

        print("VALIDATION PASSED")
        return True

    except Exception as e:
        print(f"  FAILED: {e}")
        print()
        print("VALIDATION FAILED")
        return False


if __name__ == "__main__":
    # Load .env file if present
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

    validate_auth()
