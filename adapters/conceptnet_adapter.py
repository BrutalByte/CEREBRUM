"""
Phase 228: ConceptNet 5 adapter for CEREBRUM.

Loads ConceptNet 5.x tab-separated CSV (plain or .gz) and wraps it as a
CEREBRUM GraphAdapter. Handles ConceptNet's multi-relational edges: the same
(subject, object) pair can have many relation types (e.g. dog → IsA → animal
AND dog → RelatedTo → animal). These are preserved in the multi-edge store
and exposed by get_neighbors; the projected single-edge DiGraph (max-weight
per pair) is used for community detection and structural encoding.

CN5 edge CSV format (5 tab-separated columns):
    assertion_uri  relation_uri  subject_uri  object_uri  metadata_json

URI formats:
    /r/RelatedTo          → "RelatedTo"
    /c/en/football        → "football"
    /c/en/football/n      → "football"   (POS tag stripped)
    /c/en/keep_pets       → "keep pets"  (underscore → space)

Download:
    wget https://s3.amazonaws.com/conceptnet/downloads/2019/edges/
         conceptnet-assertions-5.7.0.csv.gz

Usage
-----
    from adapters.conceptnet_adapter import load_conceptnet
    adapter = load_conceptnet("data/conceptnet-assertions-5.7.0.csv.gz",
                              lang="en", min_weight=1.0, max_edges=500_000)
    graph = CerebrumGraph.from_adapter(adapter)
    graph.build()

    # Or via the CerebrumGraph factory (Phase 228):
    graph = CerebrumGraph.from_conceptnet("data/conceptnet-assertions-5.7.0.csv.gz")
"""
from __future__ import annotations

import csv
import gzip
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx
import numpy as np

from adapters.networkx_adapter import NetworkXAdapter
from core.graph_adapter import Edge, Entity

log = logging.getLogger(__name__)

# Relations that carry no graph-structure signal (external links, raw text)
_DEFAULT_EXCLUDED = frozenset({"ExternalURL", "dbpedia", "Etymologically"})


# ---------------------------------------------------------------------------
# URI helpers
# ---------------------------------------------------------------------------

def _parse_relation(uri: str) -> str:
    """/r/RelatedTo → 'RelatedTo'"""
    return uri.strip("/").rsplit("/", 1)[-1]


def _parse_entity(uri: str) -> str:
    """/c/en/football/n → 'football', /c/en/keep_pets → 'keep pets'"""
    parts = uri.strip("/").split("/")
    if len(parts) < 3 or parts[0] != "c":
        return uri
    return parts[2].replace("_", " ")


def _entity_lang(uri: str) -> str:
    """/c/en/dog → 'en'; non-entity URIs → ''"""
    parts = uri.strip("/").split("/")
    if len(parts) < 3 or parts[0] != "c":
        return ""
    return parts[1]


# ---------------------------------------------------------------------------
# ConceptNetAdapter
# ---------------------------------------------------------------------------

class ConceptNetAdapter(NetworkXAdapter):
    """
    NetworkXAdapter subclass for ConceptNet's multi-relational edges.

    The underlying store is a nx.MultiDiGraph that preserves all relation
    types between each (subject, object) pair. A projected nx.DiGraph
    (highest-weight edge per pair) is passed to the parent NetworkXAdapter
    for structural operations (community detection, PageRank, etc.).

    get_neighbors() returns all relations from the multi-graph.
    to_networkx() returns the projected single-edge DiGraph.
    """

    def __init__(
        self,
        G_multi: nx.MultiDiGraph,
        entity_types: Optional[Dict[str, str]] = None,
    ) -> None:
        self._G_multi = G_multi

        # Project to DiGraph (max-weight edge per pair) for structural ops
        G_simple = nx.DiGraph()
        for u, v, data in G_multi.edges(data=True):
            existing = G_simple.get_edge_data(u, v)
            if existing is None or existing.get("weight", 0.0) < data.get("weight", 1.0):
                G_simple.add_edge(u, v, **data)

        super().__init__(G_simple, entity_types=entity_types)

    def get_neighbors(
        self,
        entity_id: str,
        edge_types: Optional[List[str]] = None,
        max_neighbors: int = 100,
        context_embedding: Optional[np.ndarray] = None,
    ) -> List[Edge]:
        if entity_id not in self._G_multi:
            return []

        edges: List[Edge] = []
        for _, neighbor, data in self._G_multi.out_edges(entity_id, data=True):
            rel_type = data.get("relation", "RELATED_TO")
            if edge_types and rel_type not in edge_types:
                continue
            edges.append(
                Edge(
                    source_id=entity_id,
                    target_id=neighbor,
                    relation_type=rel_type,
                    weight=float(data.get("weight", 1.0)),
                    properties=dict(data),
                    confidence=float(data.get("confidence", 1.0)),
                    provenance=str(data.get("provenance", "conceptnet")),
                    valid_from=None,
                    valid_to=None,
                )
            )

        if len(edges) > max_neighbors:
            random.shuffle(edges)
        return edges[:max_neighbors]

    def get_degree(self, entity_id: str) -> int:
        return self._G_multi.out_degree(entity_id)

    def add_edge(
        self,
        u: str,
        v: str,
        relation: str,
        confidence: float = 1.0,
        provenance: str = "",
        synthetic: bool = False,
    ) -> None:
        self._G_multi.add_edge(
            u, v,
            relation=relation,
            weight=confidence,
            confidence=confidence,
            provenance=provenance,
            synthetic=synthetic,
        )
        # Mirror into the projected simple graph
        existing = self._G.get_edge_data(u, v)
        if existing is None or existing.get("weight", 0.0) < confidence:
            self._G.add_edge(u, v, relation=relation, weight=confidence,
                             confidence=confidence, provenance=provenance)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_conceptnet(
    path: str,
    lang: str = "en",
    min_weight: float = 1.0,
    max_edges: Optional[int] = None,
    exclude_relations: Optional[List[str]] = None,
) -> ConceptNetAdapter:
    """
    Load ConceptNet 5 edge CSV (plain or .gz) into a ConceptNetAdapter.

    Parameters
    ----------
    path              : path to conceptnet-assertions-*.csv or *.csv.gz
    lang              : language code to keep (default "en"); both subject
                        and object must be in this language
    min_weight        : minimum edge weight (filters noise); CN5 weights
                        range from 1.0 (single source) to ~10+ (many sources)
    max_edges         : cap on edges loaded (None = all)
    exclude_relations : relation types to skip; defaults to ExternalURL,
                        dbpedia, Etymologically

    Returns
    -------
    ConceptNetAdapter wrapping the loaded CN5 sub-graph
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(
            f"ConceptNet file not found: {filepath.resolve()}\n"
            "Download from:\n"
            "  https://s3.amazonaws.com/conceptnet/downloads/2019/edges/"
            "conceptnet-assertions-5.7.0.csv.gz"
        )

    excluded = _DEFAULT_EXCLUDED | set(exclude_relations or [])
    G = nx.MultiDiGraph()
    loaded = skipped = 0

    opener = gzip.open if str(filepath).endswith(".gz") else open

    with opener(filepath, "rt", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            # Skip comment / malformed lines
            if len(row) < 4 or row[0].startswith("#"):
                skipped += 1
                continue

            rel_uri, subj_uri, obj_uri = row[1], row[2], row[3]
            meta_raw = row[4] if len(row) > 4 else "{}"

            if _entity_lang(subj_uri) != lang or _entity_lang(obj_uri) != lang:
                skipped += 1
                continue

            relation = _parse_relation(rel_uri)
            if relation in excluded:
                skipped += 1
                continue

            try:
                weight = float(json.loads(meta_raw).get("weight", 1.0))
            except (json.JSONDecodeError, TypeError, ValueError):
                weight = 1.0

            if weight < min_weight:
                skipped += 1
                continue

            subj = _parse_entity(subj_uri)
            obj_ = _parse_entity(obj_uri)
            if not subj or not obj_:
                skipped += 1
                continue

            G.add_edge(
                subj, obj_,
                relation=relation,
                weight=weight,
                confidence=min(weight / 10.0, 1.0),
                provenance="conceptnet5",
            )
            loaded += 1

            if max_edges and loaded >= max_edges:
                log.info("ConceptNet: capped at %d edges (max_edges limit)", max_edges)
                break

    log.info(
        "ConceptNet loaded: %d edges, %d nodes from %s "
        "(skipped %d | lang=%s | min_weight=%.1f)",
        G.number_of_edges(), G.number_of_nodes(),
        filepath.name, skipped, lang, min_weight,
    )

    if G.number_of_edges() == 0:
        raise ValueError(
            f"No edges loaded from {filepath}. "
            f"Check: lang='{lang}', min_weight={min_weight}."
        )

    return ConceptNetAdapter(G)
