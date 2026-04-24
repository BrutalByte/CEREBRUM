"""
Federated Security for CEREBRUM.

Provides Ed25519-based identity and authorization for distributed reasoning clusters.
"""
import os
import time
from typing import Dict, List, Any, Optional
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

# Security: In production, load these keys from a secure vault or HSM.
# Generating defaults for local dev.
def _load_or_gen_keys():
    if not os.path.exists("data/cerebrum/node.key"):
        private_key = ed25519.Ed25519PrivateKey.generate()
        os.makedirs("data/cerebrum", exist_ok=True)
        with open("data/cerebrum/node.key", "wb") as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))
        return private_key
    with open("data/cerebrum/node.key", "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)

PRIVATE_KEY = _load_or_gen_keys()
PUBLIC_KEY = PRIVATE_KEY.public_key()

class FederatedAuth:
    """
    Handles Ed25519 signing and verification for federated peer-to-peer trust.
    """
    @staticmethod
    def sign_payload(payload: bytes) -> bytes:
        """Sign a byte payload using the node's private key."""
        return PRIVATE_KEY.sign(payload)

    @staticmethod
    def verify_signature(public_key_pem: str, signature: bytes, payload: bytes) -> bool:
        """Verify a signature using a peer's public key."""
        try:
            public_key = serialization.load_pem_public_key(public_key_pem.encode())
            public_key.verify(signature, payload)
            return True
        except Exception:
            return False
