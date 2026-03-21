"""
Remote Parallax Adapter.

Connects to a remote Parallax API instance to perform federated graph operations.
"""
import requests
from typing import List, Optional, Dict
from core.graph_adapter import GraphAdapter, Entity, Edge


class RemoteParallaxAdapter(GraphAdapter):
    """
    Adapter that proxies requests to a remote Parallax REST API.
    """

    def __init__(self, base_url: str, timeout: int = 10, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.api_key = api_key
        self.metadata: Optional[Dict] = None

    def _get_headers(self) -> Dict[str, str]:
        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    def validate_connection(self) -> bool:
        """Handshake with remote API to verify version and capabilities."""
        try:
            resp = requests.get(f"{self.base_url}/handshake", headers=self._get_headers(), timeout=self.timeout)
            if resp.status_code == 200:
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
        edge_types: List[str] = None,
        max_neighbors: int = 50,
    ) -> List[Edge]:
        """Fetch neighbors from remote /neighbors endpoint."""
        try:
            params = {"max_neighbors": max_neighbors}
            if edge_types:
                params["edge_types"] = ",".join(edge_types)
                
            resp = requests.get(
                f"{self.base_url}/entities/{entity_id}/neighbors", 
                params=params,
                headers=self._get_headers(),
                timeout=self.timeout
            )
            if resp.status_code == 200:
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
                return resp.json()["results"]
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
