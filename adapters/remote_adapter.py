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

    def __init__(self, base_url: str, timeout: int = 10):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """
        Fetch entity details from remote. 
        Assumes a GET /entities/{id} endpoint exists or uses /query.
        """
        try:
            # For now, we'll use a hypothetical /entities endpoint
            resp = requests.get(f"{self.base_url}/entities/{entity_id}", timeout=self.timeout)
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
                    for e in resp.json()
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

    def node_count(self) -> int:
        try:
            resp = requests.get(f"{self.base_url}/stats", timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json().get("node_count", 0)
        except Exception:
            pass
        return 0
