import pytest
from core.security import FederatedAuth, PUBLIC_KEY
from core.node_registry import NodeRegistry
from cryptography.hazmat.primitives import serialization

def test_federated_security_handshake():
    # Setup
    registry = NodeRegistry(registry_path="tmp/node_registry_test.json")
    node_id = "test-node"
    pub_key_pem = PUBLIC_KEY.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode()
    registry.register_peer(node_id, pub_key_pem)
    
    # Sign payload
    payload = b"hello-world"
    signature = FederatedAuth.sign_payload(payload)
    
    # Verify signature
    peer_pub_key = registry.get_public_key(node_id)
    assert peer_pub_key is not None
    assert FederatedAuth.verify_signature(peer_pub_key, signature, payload) is True
    
    # Test invalid signature
    assert FederatedAuth.verify_signature(peer_pub_key, signature, b"wrong-payload") is False
