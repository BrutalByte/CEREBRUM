import pytest
from api.peer_discovery import PeerDiscovery
from core.node_registry import NodeRegistry
from core.security import FederatedAuth, PUBLIC_KEY
from cryptography.hazmat.primitives import serialization
from unittest.mock import MagicMock

def test_peer_discovery_handshake():
    # Setup
    mock_engine = MagicMock()
    registry = NodeRegistry(registry_path="tmp/registry_handshake_test.json")
    peer_id = "peer-1"
    
    # Register dummy peer
    pub_key_pem = PUBLIC_KEY.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode()
    registry.register_peer(peer_id, pub_key_pem)
    
    discovery = PeerDiscovery(mock_engine, None, None, None, registry)
    
    # Perform handshake
    nonce = discovery.initiate_handshake(peer_id)
    signature = FederatedAuth.sign_payload(nonce)
    
    # Verify
    assert discovery.verify_handshake(peer_id, signature, nonce) is True
    
    # Test malicious peer
    assert discovery.verify_handshake("malicious", signature, nonce) is False
