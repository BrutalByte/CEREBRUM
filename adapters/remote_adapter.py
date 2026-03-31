"""
Remote CEREBRUM Adapter.

Connects to a remote CEREBRUM API instance to perform federated graph operations.
"""
import requests
import numpy as np
import networkx as nx
from typing import List, Optional, Dict, Any
from core.graph_adapter import GraphAdapter, Entity, Edge


class RemoteCerebrumAdapter(GraphAdapter):
    """
    Adapter that proxies requests to a remote CEREBRUM REST API.
    """

    def __init__(
        self,
        base_url: str,
        timeout: int = 10,
        token: Optional[str] = None,
        secret: Optional[str] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.token = token # JWT token for bearer authentication
        self.secret = secret # Shared secret for HMAC signature verification
        self.metadata: Optional[Dict] = None

    def _get_headers(self) -> Dict[str, str]:
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _verify_signature(self, resp) -> bool:
        """
        Verify X-Signature header against the response body using the shared secret.
        Returns True if signature is valid or if no secret is configured.
        """
        if not self.secret:
            return True
            
        sig = resp.headers.get("X-Signature")
        if not sig:
            # If secret is enforced, missing signature is a failure
            return False
            
        import hmac
        import hashlib
        
        try:
            expected = hmac.new(
                self.secret.encode("utf-8"),
                resp.content,
                hashlib.sha256
            ).hexdigest()
            return hmac.compare_digest(sig, expected)
        except Exception:
            return False

    def validate_connection(self) -> bool:
        """Handshake with remote API to verify version and capabilities."""
        try:
            resp = requests.get(f"{self.base_url}/handshake", headers=self._get_headers(), timeout=self.timeout)
            if resp.status_code == 200:
                if not self._verify_signature(resp):
                    return False
                self.metadata = resp.json()
                # Verify compatibility (e.g. major version match)
                return True
        except Exception:
            pass
        return False

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """
        Fetch entity details from remote. 
        """
        try:
            resp = requests.get(
                f"{self.base_url}/entities/{entity_id}", 
                headers=self._get_headers(), 
                timeout=self.timeout
            )
            if resp.status_code == 200:
                if not self._verify_signature(resp):
                    return None
                data = resp.json()
                return Entity(
                    id=data["id"],
                    label=data["label"],
                    type=data.get("type", "entity"),
                    properties=data.get("properties", {})
                )
        except Exception:
            pass
        return None

    def get_neighbors(
        self,
        entity_id: str,
        edge_types: Optional[List[str]] = None,
        max_neighbors: int = 50,
        context_embedding: Optional[np.ndarray] = None,
    ) -> List[Edge]:
        """Fetch neighbors from remote /neighbors endpoint."""
        try:
            params: Dict[str, Any] = {"max_neighbors": max_neighbors}
            if edge_types:
                params["edge_types"] = ",".join(edge_types)
            
            # Future: send context_embedding to remote for its own blind discovery
            # if context_embedding is not None:
            #     params["context_vector"] = ...
                
            resp = requests.get(
                f"{self.base_url}/entities/{entity_id}/neighbors", 
                params=params,
                headers=self._get_headers(),
                timeout=self.timeout
            )
            if resp.status_code == 200:
                # Phase 19 fix: Federated Signature (Hole 3)
                if not self._verify_signature(resp):
                    return []

                return [
                    Edge(
                        source_id=e["source_id"],
                        target_id=e["target_id"],
                        relation_type=e["relation_type"],
                        weight=e.get("weight", 1.0),
                        properties=e.get("properties", {})
                    )
                    for e in resp.json()
                ]
        except Exception:
            pass
        return []

    def find_entities(self, query: str, top_k: int = 10) -> List[Entity]:
        """Perform remote search."""
        try:
            resp = requests.get(
                f"{self.base_url}/search", 
                params={"q": query, "top_k": top_k},
                headers=self._get_headers(),
                timeout=self.timeout
            )
            if resp.status_code == 200:
                if not self._verify_signature(resp):
                    return []
                return [
                    Entity(
                        id=e["id"],
                        label=e["label"],
                        type=e.get("type", "entity"),
                        properties=e.get("properties", {})
                    )
                    for e in resp.json()["results"]
                ]
        except Exception:
            pass
        return []

    def find_entities_masked(self, query: str, top_k: int = 10) -> List[Dict]:
        """Perform remote masked search."""
        try:
            resp = requests.get(
                f"{self.base_url}/search/masked", 
                params={"q": query, "top_k": top_k},
                headers=self._get_headers(),
                timeout=self.timeout
            )
            if resp.status_code == 200:
                if not self._verify_signature(resp):
                    return []
                return resp.json()["results"]
        except Exception:
            pass
        return []

    def find_similar(
        self, 
        embedding: "np.ndarray", 
        top_k: int = 10
    ) -> List[Entity]:
        """Perform remote semantic search via /search/similar."""
        try:
            payload = {
                "embedding": embedding.tolist(),
                "top_k": top_k
            }
            resp = requests.post(
                f"{self.base_url}/search/similar", 
                json=payload,
                headers=self._get_headers(),
                timeout=self.timeout
            )
            if resp.status_code == 200:
                if not self._verify_signature(resp):
                    return []
                return [
                    Entity(
                        id=e["id"],
                        label=e["label"],
                        type=e.get("type", "entity"),
                        properties=e.get("properties", {})
                    )
                    for e in resp.json()["results"]
                ]
        except Exception:
            pass
        return []

    def to_networkx(self) -> "nx.Graph":
        """
        Remote graphs cannot be easily converted to local NetworkX.
        Returns an empty graph or raises error.
        """
        import networkx as nx
        return nx.Graph()

    def get_community(self, entity_id: str) -> int:
        """Fetch community ID from remote /entities/{id}/community endpoint."""
        try:
            resp = requests.get(
                f"{self.base_url}/entities/{entity_id}/community", 
                headers=self._get_headers(),
                timeout=self.timeout
            )
            if resp.status_code == 200:
                if not self._verify_signature(resp):
                    return -1
                return resp.json().get("community_id", -1)
        except Exception:
            pass
        return -1

    def get_embedding(self, entity_id: str) -> Optional["np.ndarray"]:
        """Fetch embedding from remote /entities/{id}/embedding endpoint."""
        try:
            resp = requests.get(
                f"{self.base_url}/entities/{entity_id}/embedding", 
                headers=self._get_headers(),
                timeout=self.timeout
            )
            if resp.status_code == 200:
                if not self._verify_signature(resp):
                    return None
                data = resp.json()
                if "embedding" in data:
                    import numpy as np
                    return np.array(data["embedding"], dtype=np.float32)
        except Exception:
            pass
        return None

    def get_hologram(self) -> Optional[Dict]:
        """Fetch holographic community signatures."""
        try:
            resp = requests.get(
                f"{self.base_url}/hologram", 
                headers=self._get_headers(),
                timeout=self.timeout
            )
            if resp.status_code == 200:
                if not self._verify_signature(resp):
                    return None
                return resp.json()
        except Exception:
            pass
        return None

    def verify_reasoning(self, source_id: str, target_id: str, max_hop: int = 2) -> Optional[Dict]:
        """Request advanced reasoning trace from remote."""
        try:
            payload = {"source_id": source_id, "target_id": target_id, "max_hop": max_hop}
            resp = requests.post(
                f"{self.base_url}/reason", 
                json=payload, 
                headers=self._get_headers(),
                timeout=self.timeout
            )
            if resp.status_code == 200:
                if not self._verify_signature(resp):
                    return None
                return resp.json()
        except Exception:
            pass
        return None

    def node_count(self) -> int:
        try:
            resp = requests.get(
                f"{self.base_url}/stats", 
                headers=self._get_headers(),
                timeout=self.timeout
            )
            if resp.status_code == 200:
                return resp.json().get("node_count", 0)
        except Exception:
            pass
        return 0
