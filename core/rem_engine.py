"""
REM Cycle — Graph Consolidation & Connection Synthesis.

Analogous to biological REM sleep, which consolidates memories by pruning
weak synaptic connections and creating new associative links. A REM cycle:

  1. Prune   — remove low-confidence edges that add noise without contributing
               to reasoning (never prunes BRIDGE_TWIN edges).
  2. Consolidate — re-run DSCF community detection on the pruned graph so
               community assignments reflect the cleaned topology.
  3. Synthesize — find pairs of nodes with high embedding similarity but no
               direct connection; propose synthetic edges as research hypotheses.
  4. Report  — return a REMReport summarising all changes made (or would make).

Dry-run mode (dry_run=True): zero graph mutations. Returns a report showing
exactly what a real run would do — safe to call at any time for inspection.

Rollback: one level of undo. Before each real run a compact diff snapshot is
saved. rollback() restores pruned edges and removes synthetic ones.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.bridge_engine import BRIDGE_RELATION as BRIDGE_TWIN_RELATION
from core.graph_adapter import GraphAdapter


# ---------------------------------------------------------------------------
# REMReport
# ---------------------------------------------------------------------------

@dataclass
class REMReport:
    """Summary of one REM cycle (real or dry-run)."""

    pruned_edges: int
    """Number of edges removed (0 if dry_run)."""

    synthesized_edges: int
    """Number of synthetic edges added (0 if dry_run)."""

    communities_updated: bool
    """Whether community re-detection ran (False if dry_run)."""

    duration_seconds: float
    """Wall-clock time the cycle took."""

    pruned_edge_list: List[Tuple[str, str, str]]
    """(source_id, target_id, relation) for each pruned/would-be-pruned edge."""

    synthesized_edge_list: List[Tuple[str, str, str]]
    """(source_id, target_id, 'rem_synthesized') for each proposed edge."""

    dry_run: bool = False
    """True if this report describes a hypothetical run with no mutations."""

    timestamp: float = field(default_factory=time.time)

    def __repr__(self) -> str:
        mode = " [DRY RUN]" if self.dry_run else ""
        return (
            f"REMReport{mode}(pruned={self.pruned_edges}, "
            f"synthesized={self.synthesized_edges}, "
            f"communities_updated={self.communities_updated}, "
            f"duration={self.duration_seconds:.3f}s)"
        )


# ---------------------------------------------------------------------------
# Snapshot (used for rollback)
# ---------------------------------------------------------------------------

@dataclass
class _Snapshot:
    """Compact diff captured before a real REM run — enables one-level undo."""
    pruned_edges: List[Dict]          # Full edge attribute dicts + (u, v) keys
    synthesized_edges: List[Tuple[str, str]]  # (u, v) pairs added


# ---------------------------------------------------------------------------
# REMEngine
# ---------------------------------------------------------------------------

class REMEngine:
    """
    Manages REM cycles for a graph adapter.

    Parameters
    ----------
    adapter : GraphAdapter
        The graph adapter whose underlying NetworkX graph will be modified.
        Must expose ``adapter._G`` (NetworkXAdapter convention).
    prune_confidence_threshold : float
        Edges with confidence strictly below this value are pruned. Default 0.2.
    synthesis_similarity_threshold : float
        Minimum cosine similarity for a node pair to be considered for a
        synthetic edge. Default 0.8.
    cross_component_similarity_threshold : float
        Minimum cosine similarity for a node pair in disconnected components
        to be considered for a cross-component synthetic edge. Default 0.85.
    max_synthesis_proposals : int
        Maximum number of synthetic edges to propose per cycle. Default 50.
    synthesis_confidence : float
        Confidence assigned to synthetic edges. Low by default (0.3) so they
        are prunable in future cycles if they remain untraversed.
    interval_seconds : float
        Period for scheduled background cycles. Default 3600 (hourly).
    """

    def __init__(
        self,
        adapter: GraphAdapter,
        prune_confidence_threshold: float = 0.2,
        synthesis_similarity_threshold: float = 0.8,
        cross_component_similarity_threshold: float = 0.85,
        max_synthesis_proposals: int = 50,
        synthesis_confidence: float = 0.3,
        interval_seconds: float = 3600.0,
        on_complete: Optional[callable] = None,
    ):
        self.adapter = adapter
        self.prune_confidence_threshold = prune_confidence_threshold
        self.synthesis_similarity_threshold = synthesis_similarity_threshold
        self.cross_component_similarity_threshold = cross_component_similarity_threshold
        self.max_synthesis_proposals = max_synthesis_proposals
        self.synthesis_confidence = synthesis_confidence
        self.interval_seconds = interval_seconds
        # Phase 227: optional callback fired in a background thread after each
        # real (non-dry) run.  Signature: on_complete(report: REMReport) -> None.
        # Used by CerebrumGraph to trigger NVMe consolidation after REM.
        self._on_complete = on_complete

        self._lock = threading.RLock()
        self._snapshot: Optional[_Snapshot] = None
        self._last_report: Optional[REMReport] = None
        self._timer: Optional[threading.Timer] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def last_report(self) -> Optional[REMReport]:
        """The most recent REMReport, or None if no cycle has run."""
        with self._lock:
            return self._last_report

    @property
    def can_rollback(self) -> bool:
        """True if a snapshot from the last non-dry-run cycle exists."""
        with self._lock:
            return self._snapshot is not None

    def run(self, dry_run: bool = False) -> REMReport:
        """
        Execute one full REM cycle.

        Parameters
        ----------
        dry_run : bool
            If True, compute and return the report without mutating the graph.

        Returns
        -------
        REMReport describing changes made (or that would be made).
        """
        t0 = time.time()
        G = self._get_graph()

        with self._lock:
            # --- Phase 1: Prune ---
            pruned = self._prune(G, dry_run=dry_run)

            # --- Phase 2: Consolidate ---
            consolidated = self._consolidate(G, dry_run=dry_run)

            # --- Phase 3: Synthesize ---
            synthesized = self._synthesize(G, dry_run=dry_run)

            # --- Snapshot (real runs only) ---
            if not dry_run:
                # Snapshot was built inside _prune/_synthesize; see _take_snapshot
                pass

            report = REMReport(
                pruned_edges=0 if dry_run else len(pruned),
                synthesized_edges=0 if dry_run else len(synthesized),
                communities_updated=consolidated and not dry_run,
                duration_seconds=time.time() - t0,
                pruned_edge_list=pruned,
                synthesized_edge_list=synthesized,
                dry_run=dry_run,
            )
            self._last_report = report

        # Phase 227: fire NVMe consolidation in a background thread so the
        # REM report returns immediately.  Runs only on real (non-dry) cycles.
        if not dry_run and self._on_complete is not None:
            cb = self._on_complete
            t = threading.Thread(target=cb, args=(report,), daemon=True,
                                 name="cerebrum-mmap-flush")
            t.start()

        return report

    def rollback(self) -> int:
        """
        Undo the most recent non-dry-run cycle.

        Restores all pruned edges with their original attributes and removes
        all synthetic edges added during that cycle.

        Returns
        -------
        int : total number of edge operations performed (restored + removed).

        Raises
        ------
        RuntimeError if no snapshot exists (no prior real run).
        """
        with self._lock:
            if self._snapshot is None:
                raise RuntimeError(
                    "No snapshot available — call run(dry_run=False) first."
                )
            G = self._get_graph()
            ops = 0

            # Restore pruned edges
            for edge_data in self._snapshot.pruned_edges:
                u = edge_data["_u"]
                v = edge_data["_v"]
                attrs = {k: val for k, val in edge_data.items() if not k.startswith("_")}
                if not G.has_edge(u, v):
                    G.add_edge(u, v, **attrs)
                    ops += 1

            # Remove synthetic edges
            for u, v in self._snapshot.synthesized_edges:
                if G.has_edge(u, v):
                    edge_data = G.get_edge_data(u, v) or {}
                    if edge_data.get("provenance") == "rem_synthesized":
                        G.remove_edge(u, v)
                        ops += 1

            self._snapshot = None
            return ops

    def schedule(self, interval_seconds: Optional[float] = None) -> None:
        """Start periodic background REM cycles."""
        if interval_seconds is not None:
            self.interval_seconds = interval_seconds
        self._schedule_next()

    def cancel(self) -> None:
        """Cancel any pending scheduled cycle."""
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_graph(self):
        """Return the underlying NetworkX graph from the adapter."""
        G = getattr(self.adapter, "_G", None)
        if G is None:
            raise AttributeError(
                "REMEngine requires adapter._G (NetworkXAdapter). "
                "Override _get_graph() for other backends."
            )
        return G

    def _prune(self, G, dry_run: bool) -> List[Tuple[str, str, str]]:
        """
        Collect and (if not dry_run) remove low-confidence edges.
        BRIDGE_TWIN edges are always preserved.
        """
        to_remove = []
        for u, v, data in G.edges(data=True):
            relation = data.get("relation", data.get("relation_type", ""))
            if relation == BRIDGE_TWIN_RELATION:
                continue
            confidence = data.get("confidence", 1.0)
            if confidence < self.prune_confidence_threshold:
                to_remove.append((u, v, relation, dict(data)))

        # Build snapshot before mutating (real runs only)
        if not dry_run:
            pruned_snapshot = []
            for u, v, _rel, data in to_remove:
                entry = {"_u": u, "_v": v}
                entry.update(data)
                pruned_snapshot.append(entry)
            # Store in a temporary attribute; finalized in run() after synthesize
            self._pending_pruned_snapshot = pruned_snapshot

            for u, v, _rel, _data in to_remove:
                G.remove_edge(u, v)

        return [(u, v, rel) for u, v, rel, _ in to_remove]

    def _consolidate(self, G, dry_run: bool) -> bool:
        """Re-run DSCF community detection and update the adapter's community map."""
        if dry_run:
            return False
        try:
            from core.community_engine import dscf_communities
            communities = dscf_communities(G)
            community_map = {}
            for cid, members in enumerate(communities):
                for node in members:
                    community_map[node] = cid
            # Push back to adapter if it exposes community_map
            if hasattr(self.adapter, "community_map"):
                self.adapter.community_map = community_map
            return True
        except Exception:
            return False

    def _synthesize(self, G, dry_run: bool) -> List[Tuple[str, str, str]]:
        """
        Find high-similarity unconnected node pairs and propose synthetic edges.
        Uses cosine similarity of node embeddings and NetworkX shortest-path
        proximity check (structural closeness <= 4 hops).
        """
        import networkx as nx

        proposals: List[Tuple[str, str, str]] = []
        nodes = list(G.nodes())
        # Use undirected view for proximity so hub→alpha, hub→beta counts as alpha↔beta
        G_und = G.to_undirected() if G.is_directed() else G

        # Collect embeddings in one pass to avoid repeated adapter calls
        emb_cache: Dict[str, Optional[np.ndarray]] = {}
        for node in nodes:
            emb_cache[node] = self.adapter.get_embedding(node)

        for i, u in enumerate(nodes):
            if len(proposals) >= self.max_synthesis_proposals:
                break
            emb_u = emb_cache[u]
            if emb_u is None:
                continue
            norm_u = float(np.linalg.norm(emb_u))
            if norm_u == 0:
                continue

            for j in range(i + 1, len(nodes)):
                if len(proposals) >= self.max_synthesis_proposals:
                    break
                v = nodes[j]
                if G.has_edge(u, v) or G.has_edge(v, u):
                    continue
                emb_v = emb_cache[v]
                if emb_v is None:
                    continue
                norm_v = float(np.linalg.norm(emb_v))
                if norm_v == 0:
                    continue

                similarity = float(np.dot(emb_u, emb_v) / (norm_u * norm_v))
                if similarity < self.synthesis_similarity_threshold:
                    continue

                # Structural proximity check (path length <= 4, undirected)
                try:
                    path_len = nx.shortest_path_length(G_und, u, v)
                    if path_len > 4:
                        continue
                except nx.NetworkXNoPath:
                    continue
                except nx.NodeNotFound:
                    continue

                proposals.append((u, v, "rem_synthesized"))

        # --- Phase 3b: Cross-Component Synthesis (SynapticBridge Detection) ---
        # Find high-similarity pairs in disconnected components.
        if len(proposals) < self.max_synthesis_proposals:
            components = list(nx.connected_components(G_und))
            if len(components) > 1:
                # Map nodes to component index for fast lookup
                node_to_comp = {}
                for idx, comp in enumerate(components):
                    for node in comp:
                        node_to_comp[node] = idx
                
                # Check cross-component pairs
                for i, u in enumerate(nodes):
                    if len(proposals) >= self.max_synthesis_proposals:
                        break
                    emb_u = emb_cache[u]
                    if emb_u is None: continue
                    norm_u = float(np.linalg.norm(emb_u))
                    if norm_u == 0: continue
                    
                    u_comp = node_to_comp[u]

                    for j in range(i + 1, len(nodes)):
                        if len(proposals) >= self.max_synthesis_proposals:
                            break
                        v = nodes[j]
                        if node_to_comp[v] == u_comp:
                            continue # Already in same component
                        
                        emb_v = emb_cache[v]
                        if emb_v is None: continue
                        norm_v = float(np.linalg.norm(emb_v))
                        if norm_v == 0: continue

                        similarity = float(np.dot(emb_u, emb_v) / (norm_u * norm_v))
                        # Higher threshold for cross-component to avoid noisy bridges
                        if similarity < self.cross_component_similarity_threshold:
                            continue

                        proposals.append((u, v, "rem_synthesized_SynapticBridge"))

        # Apply mutations (real runs only)
        if not dry_run:
            synthetic_pairs = []
            for u, v, rel in proposals:
                G.add_edge(
                    u, v,
                    relation=rel,
                    confidence=self.synthesis_confidence,
                    provenance="rem_synthesized",
                    weight=0.5,
                )
                synthetic_pairs.append((u, v))

            # Finalise snapshot
            pruned_snap = getattr(self, "_pending_pruned_snapshot", [])
            self._snapshot = _Snapshot(
                pruned_edges=pruned_snap,
                synthesized_edges=synthetic_pairs,
            )
            if hasattr(self, "_pending_pruned_snapshot"):
                del self._pending_pruned_snapshot

        return proposals

    def _schedule_next(self) -> None:
        """Schedule the next cycle via threading.Timer (recursive)."""
        def _fire():
            try:
                self.run(dry_run=False)
            finally:
                self._schedule_next()

        with self._lock:
            self._timer = threading.Timer(self.interval_seconds, _fire)
            self._timer.daemon = True
            self._timer.start()
