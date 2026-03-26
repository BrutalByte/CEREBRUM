
import pytest
import hmac
import hashlib
import json
from unittest.mock import patch, MagicMock
from adapters.remote_adapter import RemoteCerebrumAdapter

def test_federated_hmac_verification_success():
    """Verify that valid HMAC signatures are accepted."""
    secret = "test-secret"
    adapter = RemoteCerebrumAdapter("http://remote-api", secret=secret)
    
    body = [{"source_id": "A", "target_id": "B", "relation_type": "knows"}]
    body_json = json.dumps(body).encode("utf-8")
    
    # Calculate signature
    sig = hmac.new(secret.encode("utf-8"), body_json, hashlib.sha256).hexdigest()
    
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = body_json
    mock_resp.headers = {"X-Signature": sig}
    mock_resp.json.return_value = body
    
    with patch("requests.get", return_value=mock_resp):
        edges = adapter.get_neighbors("A")
        assert len(edges) == 1
        assert edges[0].target_id == "B"

def test_federated_hmac_verification_failure():
    """Verify that invalid HMAC signatures cause empty results."""
    secret = "test-secret"
    adapter = RemoteCerebrumAdapter("http://remote-api", secret=secret)
    
    body = [{"source_id": "A", "target_id": "B", "relation_type": "knows"}]
    body_json = json.dumps(body).encode("utf-8")
    
    # WRONG signature
    sig = "invalid-sig"
    
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = body_json
    mock_resp.headers = {"X-Signature": sig}
    
    with patch("requests.get", return_value=mock_resp):
        edges = adapter.get_neighbors("A")
        # Should be empty because signature check failed
        assert len(edges) == 0

def test_federated_hmac_missing_signature():
    """Verify that missing signature is rejected when secret is configured."""
    secret = "test-secret"
    adapter = RemoteCerebrumAdapter("http://remote-api", secret=secret)
    
    body = {"id": "A", "label": "Entity A"}
    body_json = json.dumps(body).encode("utf-8")
    
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = body_json
    mock_resp.headers = {} # No signature
    
    with patch("requests.get", return_value=mock_resp):
        ent = adapter.get_entity("A")
        assert ent is None

def test_federated_no_secret_allows_all():
    """Verify that if no secret is set, signature is not required."""
    adapter = RemoteCerebrumAdapter("http://remote-api")
    
    body = [{"source_id": "A", "target_id": "B", "relation_type": "knows"}]
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = body
    
    with patch("requests.get", return_value=mock_resp):
        edges = adapter.get_neighbors("A")
        assert len(edges) == 1
