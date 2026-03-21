"""
RDF/SPARQL graph adapter for Parallax.

Connects to any SPARQL endpoint (Wikidata, DBpedia, local RDF store)
and implements GraphAdapter.

Requires: pip install SPARQLWrapper
"""
from typing import List, Optional

import networkx as nx

from core.graph_adapter import GraphAdapter, Entity, Edge


# Default SPARQL prefixes for Wikidata
WIKIDATA_PREFIXES = """
PREFIX wd:  <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
"""


class RDFAdapter(GraphAdapter):
    """
    GraphAdapter backed by a SPARQL endpoint.

    Usage (Wikidata):
        from adapters.rdf_adapter import RDFAdapter
        adapter = RDFAdapter(
            endpoint="https://query.wikidata.org/sparql",
            prefixes=WIKIDATA_PREFIXES,
        )
        # Seed with Wikidata entity IDs (e.g. "Q937" for Einstein)
        entities = adapter.find_entities("Albert Einstein")

    Usage (local Fuseki / Virtuoso):
        adapter = RDFAdapter(endpoint="http://localhost:3030/dataset/sparql")

    Note: SPARQL queries can be slow on remote endpoints. For production,
    consider caching results or using a local RDF store.
    """

    def __init__(
        self,
        endpoint: str,
        prefixes: str = "",
        label_predicate: str = "rdfs:label",
        lang_filter: str = "en",
        timeout: int = 30,
    ):
        self._endpoint        = endpoint
        self._prefixes        = prefixes
        self._label_predicate = label_predicate
        self._lang_filter     = lang_filter
        self._timeout         = timeout
        self._sparql          = None

    def _get_sparql(self):
        if self._sparql is None:
            try:
                from SPARQLWrapper import SPARQLWrapper, JSON
                self._sparql = SPARQLWrapper(self._endpoint)
                self._sparql.setReturnFormat(JSON)
                self._sparql.setTimeout(self._timeout)
            except ImportError as e:
                raise ImportError(
                    "SPARQLWrapper required: pip install SPARQLWrapper"
                ) from e
        return self._sparql

    def _query(self, sparql_body: str) -> list:
        from SPARQLWrapper import JSON
        sw = self._get_sparql()
        sw.setQuery(self._prefixes + "\n" + sparql_body)
        result = sw.queryAndConvert()
        return result["results"]["bindings"]

    # ------------------------------------------------------------------
    # Required GraphAdapter methods
    # ------------------------------------------------------------------

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        rows = self._query(
            f"SELECT ?label WHERE {{"
            f"  <{entity_id}> {self._label_predicate} ?label ."
            f'  FILTER(lang(?label) = "{self._lang_filter}")'
            f"}} LIMIT 1"
        )
        label = rows[0]["label"]["value"] if rows else entity_id
        return Entity(id=entity_id, label=label, type="entity")

    def get_neighbors(
        self,
        entity_id: str,
        edge_types: Optional[List[str]] = None,
        max_neighbors: int = 50,
    ) -> List[Edge]:
        rows = self._query(
            f"SELECT ?pred ?obj WHERE {{"
            f"  <{entity_id}> ?pred ?obj ."
            f"  FILTER(isIRI(?obj))"
            f"}} LIMIT {max_neighbors}"
        )
        edges = []
        for row in rows:
            pred = row["pred"]["value"]
            obj  = row["obj"]["value"]
            # Strip namespace to get a short relation label
            rel = pred.split("/")[-1].split("#")[-1]
            if edge_types and rel not in edge_types:
                continue
            edges.append(Edge(source_id=entity_id, target_id=obj, relation_type=rel))
        return edges

    def find_entities(self, query: str, top_k: int = 10) -> List[Entity]:
        rows = self._query(
            f"SELECT DISTINCT ?entity ?label WHERE {{"
            f"  ?entity {self._label_predicate} ?label ."
            f'  FILTER(lang(?label) = "{self._lang_filter}")'
            f'  FILTER(CONTAINS(LCASE(STR(?label)), LCASE("{query}")))'
            f"}} LIMIT {top_k}"
        )
        return [
            Entity(
                id=row["entity"]["value"],
                label=row["label"]["value"],
                type="entity",
            )
            for row in rows
        ]

    def to_networkx(self) -> nx.Graph:
        """
        Export a subgraph to NetworkX by sampling edges.
        For large endpoints, limit LIMIT to a manageable size.
        """
        rows = self._query(
            "SELECT ?s ?p ?o WHERE {"
            "  ?s ?p ?o ."
            "  FILTER(isIRI(?s) && isIRI(?o))"
            "} LIMIT 10000"
        )
        G = nx.DiGraph()
        for row in rows:
            s   = row["s"]["value"]
            o   = row["o"]["value"]
            rel = row["p"]["value"].split("/")[-1].split("#")[-1]
            G.add_edge(s, o, relation=rel)
        return G



