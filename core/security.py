"""
Federated Security for CEREBRUM.

Provides JWT-based identity and authorization for distributed reasoning clusters.
"""
import os
import time
import jwt
from typing import Dict, List, Any

# Security: In production, rotate this secret or use asymmetric RSA/EdDSA.
SHARED_SECRET = os.getenv("PARALLAX_SHARED_SECRET", "federation-secret-change-me")
ALGORITHM = "HS256"

class FederatedAuth:
    """
    Handles JWT generation and validation for federated peer-to-peer trust.
    """
    @staticmethod
    def create_token(
        node_id: str, 
        scopes: List[str] = ["query", "search"], 
        expires_in: int = 3600
    ) -> str:
        """Create a signed access token for a peer node."""
        payload = {
            "sub": node_id,
            "iat": int(time.time()),
            "exp": int(time.time()) + expires_in,
            "scopes": scopes
        }
        return jwt.encode(payload, SHARED_SECRET, algorithm=ALGORITHM)

    @staticmethod
    def validate_token(token: str) -> Dict[str, Any]:
        """
        Validate and decode a JWT. 
        Returns payload if valid, raises jwt.PyJWTError otherwise.
        """
        return jwt.decode(token, SHARED_SECRET, algorithms=[ALGORITHM])

    @staticmethod
    def has_scope(token_payload: Dict[str, Any], required_scope: str) -> bool:
        """Check if the token grants a specific capability."""
        return required_scope in token_payload.get("scopes", [])
