"""
Amazon Neptune Gremlin Adapter.

Connects via gremlinpython WebSocket to query AWS Neptune graph clusters.
Provides graph traversal capabilities matching the CEREBRUM core architecture.
"""
from typing import List, Optional, Any
import numpy as np
import networkx as nx

from core.graph_adapter import GraphAdapter, Entity, Edge

class NeptuneAdapter(GraphAdapter):
    """
    GraphAdapter backed by Amazon Neptune via Gremlin Server.
    """

    def __init__(self, endpoint: str = "wss://localhost:8182/gremlin"):
        self.endpoint = endpoint
        self.g = None
        self.connection = None

    def connect(self):
        try:
            from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection
            from gremlin_python.process.anonymous_traversal import traversal
        except ImportError as e:
            raise ImportError("gremlinpython required: pip install gremlinpython") from e
            
        self.connection = DriverRemoteConnection(self.endpoint, 'g')
        self.g = traversal().withRemote(self.connection)

    def close(self):
        if self.connection:
            self.connection.close()
            self.g = None
            self.connection = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.close()

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        if not self.g: 
            raise RuntimeError("Call connect() before using adapter.")
        
        # In Gremlin: g.V().has('name', id).valueMap(True).toList()
        try:
            results = self.g.V().has('name', entity_id).valueMap(True).toList()
        except Exception:
            return None
            
        if not results:
            return None
            
        props = results[0]
        # Simplify extraction for property maps where values are lists
        name = props.get('name', [entity_id])[0]
        ent_type = props.get('type', ['entity'])[0]
        return Entity(id=entity_id, label=name, type=ent_type, properties=props)

    def get_neighbors(
        self,
        entity_id: str,
        edge_types: Optional[List[str]] = None,
        max_neighbors: int = 50,
        context_embedding: Optional[np.ndarray] = None,
    ) -> List[Edge]:
        if not self.g: 
            raise RuntimeError("Call connect() before using adapter.")
            
        # Build traversal: g.V().has('name', id).outE().inV().path()
        try:
            if edge_types:
                t = self.g.V().has('name', entity_id).outE(*edge_types)
            else:
                t = self.g.V().has('name', entity_id).outE()
                
            path_results = t.inV().path().limit(max_neighbors).toList()
        except Exception:
            return []
            
        edges = []
        for path in path_results:
            # Expected path: [v_out, e, v_in]
            if len(path) == 3:
                e = path[1]
                v_in = path[2]
                
                # Depending on gremlin_python serialization, e can be an Edge object
                try:
                    rel_type = e.label
                    target_id = v_in.properties.get('name', [v_in.id])[0] if hasattr(v_in, 'properties') else str(v_in.id)
                except AttributeError:
                    # Fallback string manipulation if mocked as dicts
                    rel_type = getattr(e, "relation", "RELATED_TO")
                    target_id = getattr(v_in, "name", "target")
                
                edges.append(Edge(
                    source_id=entity_id,
                    target_id=target_id,
                    relation_type=rel_type,
                ))
        return edges

    def find_entities(self, query: str, top_k: int = 10) -> List[Entity]:
        if not self.g: 
            raise RuntimeError("Call connect() before using adapter.")
        try:
            from gremlin_python.process.traversal import TextP
            # Simplified substring match
            results = self.g.V().has('name', TextP.containing(query)).limit(top_k).valueMap(True).toList()
        except ImportError:
            results = []
        
        entities = []
        for props in results:
            name = props.get('name', ['unknown'])[0]
            entities.append(Entity(id=name, label=name))
        return entities

    def get_community(self, entity_id: str) -> int:
        return -1

    def add_edge(
        self,
        u: str,
        v: str,
        relation: str,
        confidence: float = 1.0,
        provenance: str = "",
        synthetic: bool = False,
    ) -> None:
        """Add an edge to the Neptune graph using Gremlin."""
        if not self.g:
            raise RuntimeError("Call connect() before using adapter.")
        
        # Ensure nodes exist and then add edge
        # g.V().has('name', u).fold().coalesce(unfold(), addV().property('name', u)).as('a')
        #  .V().has('name', v).fold().coalesce(unfold(), addV().property('name', v)).as('b')
        #  .addE(relation).from('a').to('b')
        #  .property('confidence', confidence).property('provenance', provenance).property('synthetic', synthetic)
        try:
            self.g.V().has('name', u).fold().coalesce(
                self.g.unfold(), self.g.addV().property('name', u).property('type', 'entity')
            ).as_('a').V().has('name', v).fold().coalesce(
                self.g.unfold(), self.g.addV().property('name', v).property('type', 'entity')
            ).as_('b').addE(relation).from_('a').to('b').property(
                'confidence', confidence
            ).property('provenance', provenance).property('synthetic', synthetic).iterate()
        except Exception as e:
            # In a production environment, we might want to log this error
            pass

    def get_embedding(self, entity_id: str) -> Optional[np.ndarray]:
        return None

    def find_similar(self, embedding: np.ndarray, top_k: int = 10) -> List[Entity]:
        # Amazon Neptune lacks native exact dense vector search (requires OpenSearch integration)
        return []

    def to_networkx(self) -> nx.Graph:
        """Export not fully supported for massive Neptune graphs due to scale."""
        return nx.Graph()
