import pytest
import json
from unittest.mock import MagicMock
from core.graph_bridge import GraphBridgeEngine
from core.node_registry import NodeRegistry
from core.security import FederatedAuth, PUBLIC_KEY
from cryptography.hazmat.primitives import serialization

def test_bridge_edge_authentication():
    # Setup
    mock_adapter = MagicMock()
    mock_adapter._G = MagicMock()
    node_id = "local-node"
    
    registry = NodeRegistry(registry_path="tmp/registry_bridge_test.json")
    pub_key_pem = PUBLIC_KEY.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode()
    registry.register_peer(node_id, pub_key_pem)
    
    engine = GraphBridgeEngine(node_id=node_id, registry=registry)
    
    # Mock add_edge to intercept calls
    mock_G = MagicMock()
    src, tgt = "A", "B"
    attrs = {"confidence": 0.9}
    sim_val = 0.9
    src_comp, tgt_comp = "1", "2"
    
    # Execute authenticated edge addition
    engine._add_authenticated_edge(mock_G, src, tgt, attrs, sim_val, src_comp, tgt_comp)
    
    # Verify signature
    assert mock_G.add_edge.called
    args, kwargs = mock_G.add_edge.call_args
    assert "signature" in kwargs
    assert "signer" in kwargs
    assert kwargs["signer"] == node_id
    
    # Verify integrity of signed payload
    signature_hex = kwargs["signature"]
    signature = bytes.fromhex(signature_hex)
    
    # Retrieve the exact attrs passed to add_edge
    signed_attrs = kwargs.copy()
    del signed_attrs["signature"]
    del signed_attrs["signer"]
    
    expected_prov = attrs["provenance"]
    expected_payload = {"src": src, "tgt": tgt, "attrs": signed_attrs, "prov": expected_prov}
    encoded_payload = json.dumps(expected_payload, sort_keys=True).encode()
    
    # The signature in kwargs is signed on the payload, but the attrs in the payload
    # are the ones stored in the kwargs as well. The test verification logic should
    # rebuild the exact same object.
    assert FederatedAuth.verify_signature(pub_key_pem, signature, encoded_payload) is True
