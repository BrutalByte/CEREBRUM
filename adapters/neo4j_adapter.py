"""
Neo4j graph adapter for Parallax.

Connects to a Neo4j instance via the bolt driver and implements GraphAdapter.
Patterns ported from Home Assistant services/knowledge_service/main.py.

Requires: pip install neo4j
"""
from typing import List, Optional

import networkx as nx

from core.graph_adapter import GraphAdapter, Entity, Edge


class Neo4jAdapter(GraphAdapter):
    """
    GraphAdapter backed by a Neo4j graph database.

    Usage:
        from adapters.neo4j_adapter import Neo4jAdapter
        adapter = Neo4jAdapter(uri="bolt://localhost:7687", user="neo4j", password="...")
        adapter.connect()
        # ... use adapter ...
        adapter.close()

    Or as a context manager:
        with Neo4jAdapter.from_env() as adapter:
            paths = traversal.traverse(["einstein"])
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "neo4j",
        database: str = "neo4j",
        node_label: str = "Entity",
        name_property: str = "name",
        max_graph_nodes: int = 50_000,
    ):
        self._uri            = uri
        self._user           = user
        self._password       = password
        self._database       = database
        self._node_label     = node_label
        self._name_property  = name_property
        self._max_graph_nodes = max_graph_nodes
        self._driver         = None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        try:
            from neo4j import GraphDatabase
        except ImportError as e:
            raise ImportError("neo4j driver required: pip install neo4j") from e
        self._driver = GraphDatabase.driver(
            self._uri, auth=(self._user, self._password)
        )
        self._driver.verify_connectivity()

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            self._driver = None

    def __enter__(self) -> "Neo4jAdapter":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    @classmethod
    def from_env(cls) -> "Neo4jAdapter":
        """Load connection params from environment variables."""
        import os
        return cls(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "neo4j"),
            database=os.getenv("NEO4J_DATABASE", "neo4j"),
        )

    def _session(self):
        if self._driver is None:
            raise RuntimeError("Call connect() before using the adapter.")
        return self._driver.session(database=self._database)

    # ------------------------------------------------------------------
    # Required GraphAdapter methods
    # ------------------------------------------------------------------

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        with self._session() as s:
            row = s.run(
                f"MATCH (n:{self._node_label} {{{self._name_property}: $id}}) "
                "RETURN n LIMIT 1",
                {"id": entity_id},
            ).single()
        if row is None:
            return None
        node = row["n"]
        return Entity(
            id=entity_id,
            label=node.get(self._name_property, entity_id),
            type=node.get("type", "entity"),
            properties=dict(node),
        )

    def get_neighbors(
        self,
        entity_id: str,
        edge_types: Optional[List[str]] = None,
        max_neighbors: int = 50,
    ) -> List[Edge]:
        if edge_types:
            type_filter = ":" + "|".join(edge_types)
        else:
            type_filter = ""

        cypher = (
            f"MATCH (n:{self._node_label} {{{self._name_property}: $id}})"
            f"-[r{type_filter}]->(m:{self._node_label}) "
            f"RETURN m.{self._name_property} AS target, type(r) AS rel "
            f"LIMIT $limit"
        )
        with self._session() as s:
            rows = s.run(cypher, {"id": entity_id, "limit": max_neighbors}).data()

        return [
            Edge(
                source_id=entity_id,
                target_id=row["target"],
                relation_type=row["rel"],
            )
            for row in rows
            if row.get("target")
        ]

    def find_entities(self, query: str, top_k: int = 10) -> List[Entity]:
        cypher = (
            f"MATCH (n:{self._node_label}) "
            f"WHERE toLower(n.{self._name_property}) CONTAINS toLower($q) "
            f"RETURN n.{self._name_property} AS id LIMIT $limit"
        )
        with self._session() as s:
            rows = s.run(cypher, {"q": query, "limit": top_k}).data()
        return [
            e for r in rows
            if (e := self.get_entity(r["id"])) is not None
        ]

    def to_networkx(self) -> nx.Graph:
        """
        Export the Neo4j graph to NetworkX for community detection.
        Loads up to max_graph_nodes nodes to prevent OOM on large graphs.
        """
        cypher = (
            "MATCH (a)-[r]->(b) "
            f"RETURN a.{self._name_property} AS src, "
            f"       b.{self._name_property} AS tgt, "
            "        type(r) AS rel "
            f"LIMIT {self._max_graph_nodes * 10}"
        )
        with self._session() as s:
            rows = s.run(cypher).data()

        G = nx.DiGraph()
        for row in rows:
            if row.get("src") and row.get("tgt"):
                G.add_edge(row["src"], row["tgt"], relation=row.get("rel", "RELATED_TO"))
        return G



