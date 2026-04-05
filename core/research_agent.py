"""
ResearchAgent — Autonomous Missing-Link Discovery Daemon (Phase 51).

Continuously mines (source, target) candidate pairs from the graph, calls
HypothesisEngine.generate() on each, and accumulates findings in a fixed-size
ring buffer for human review and optional materialization.

Three candidate sources (combined and ranked by discovery potential):
  1. Embedding similarity scan  — pairs with cosine sim ∈ [0.6, 0.95] and no
     direct edge.  Semantically related but structurally disconnected.
  2. InsightEngine seeding      — cross-community paths already flagged as
     surprising; surface them as hypothesis candidates.
  3. Manual override            — external callers push (source, target) directly.

Usage
-----
    from core.research_agent import ResearchAgent
    from core.hypothesis_engine import HypothesisEngine

    hypothesis_engine = HypothesisEngine(adapter)
    agent = ResearchAgent(adapter, hypothesis_engine)

    findings = agent.scan_once()          # one-shot blocking scan
    agent.start()                          # start background daemon (non-blocking)
    agent.approve(findings[0].finding_id) # materialize proposals to graph
    agent.stop()
"""
from __future__ import annotations

import collections
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ResearchCandidate:
    """A (source, target) pair selected for hypothesis generation."""

    source_id: str
    target_id: str
    discovery_potential: float
    """Combined score in [0, 1]: higher means more likely to be a novel connection."""

    gap_score: float
    """Cosine distance (1 − similarity): semantically related but disconnected."""

    community_distance: int
    """Ordinal proxy: 0 if same community, 1 if adjacent, 2+ if distant."""

    seeded_by: str
    """Origin: ``"embedding_scan"`` | ``"insight_engine"`` | ``"manual"``."""

    created_at: float = field(default_factory=time.time)


@dataclass
class ResearchFinding:
    """Result of running HypothesisEngine on a ResearchCandidate."""

    finding_id: str
    candidate: ResearchCandidate
    proposals: list
    """List[HypothesisProposal] from HypothesisEngine.generate()."""

    best_confidence: float
    """max(p.confidence for p in proposals), or 0.0 if empty."""

    literature_status: str = "unvalidated"
    """Set by ExternalValidator (Phase 52): novel | active_research | established | contested."""

    validation_report: Optional[Any] = None
    """ValidationReport from Phase 52; None until validated."""

    found_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# ResearchAgent
# ---------------------------------------------------------------------------

class ResearchAgent:
    """
    Autonomous background daemon that finds missing links in a knowledge graph.

    Parameters
    ----------
    adapter
        GraphAdapter with populated community_map and embeddings.
    hypothesis_engine
        Pre-constructed HypothesisEngine used to generate proposals.
    insight_engine
        Optional InsightEngine; when provided, recent InsightEvents seed
        additional candidates (hot-path cross-community crossings).
    scan_interval
        Seconds between background scan cycles (default 300 = 5 minutes).
    candidate_limit
        Maximum number of (source, target) pairs evaluated per scan cycle.
    min_discovery_potential
        Candidates below this threshold are discarded before running
        HypothesisEngine (saves compute on low-value pairs).
    min_confidence
        Proposals below this confidence are not stored as findings.
    findings_capacity
        Fixed-size ring buffer capacity for ResearchFindings.
    max_hop, beam_width, max_budget
        Forwarded to HypothesisEngine.generate() for each candidate.
    """

    def __init__(
        self,
        adapter,
        hypothesis_engine,
        insight_engine=None,
        scan_interval: float = 300.0,
        candidate_limit: int = 100,
        min_discovery_potential: float = 0.30,
        min_confidence: float = 0.20,
        findings_capacity: int = 500,
        max_hop: int = 3,
        beam_width: int = 8,
        max_budget: int = 300,
    ) -> None:
        self._adapter           = adapter
        self._hypothesis_engine = hypothesis_engine
        self._insight_engine    = insight_engine
        self.scan_interval      = scan_interval
        self.candidate_limit    = candidate_limit
        self.min_discovery_potential = min_discovery_potential
        self.min_confidence     = min_confidence
        self.max_hop            = max_hop
        self.beam_width         = beam_width
        self.max_budget         = max_budget

        self._lock              = threading.RLock()
        self._timer: Optional[threading.Timer] = None
        self._running: bool     = False
        self._total_scans: int  = 0
        self._total_findings: int = 0
        self._last_scan_at: Optional[float] = None
        self._last_edge_count: Optional[int] = None  # edge count checkpoint

        # Ring buffers
        self._findings: collections.deque = collections.deque(maxlen=findings_capacity)
        self._evaluated_pairs: Set[Tuple[str, str]] = set()

        # Optional external validator (set by server)
        self._validator: Optional[Any] = None

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background scanning daemon (idempotent)."""
        with self._lock:
            if self._running:
                return
            self._running = True
        self._schedule_next()
        logger.info("ResearchAgent started (interval=%.0fs)", self.scan_interval)

    def stop(self) -> None:
        """Stop the background scanning daemon."""
        with self._lock:
            self._running = False
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
        logger.info("ResearchAgent stopped")

    def scan_once(self) -> List[ResearchFinding]:
        """
        Trigger an immediate one-shot scan and return any new findings.
        Blocking: waits for the scan to complete before returning.
        """
        return self._run_scan()

    def set_validator(self, validator) -> None:
        """
        Attach an ExternalValidator (Phase 52).  When set, each finding is
        automatically validated before entering the ring buffer.
        """
        with self._lock:
            self._validator = validator

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    @property
    def findings(self) -> List[ResearchFinding]:
        """Snapshot of all pending findings in the ring buffer."""
        with self._lock:
            return list(self._findings)

    @property
    def status(self) -> Dict[str, Any]:
        """Current agent state."""
        with self._lock:
            return {
                "running":          self._running,
                "scan_interval":    self.scan_interval,
                "total_scans":      self._total_scans,
                "total_findings":   self._total_findings,
                "pending_findings": len(self._findings),
                "last_scan_at":     self._last_scan_at,
            }

    # ------------------------------------------------------------------
    # Human-in-the-loop
    # ------------------------------------------------------------------

    def approve(self, finding_id: str) -> int:
        """
        Materialize all proposals in a finding to the graph.

        Returns
        -------
        int
            Number of edges added to the graph.

        Raises
        ------
        ValueError
            If finding_id is not in the ring buffer.
        """
        finding = self._get_finding(finding_id)
        edges_added = self._hypothesis_engine.materialize(finding.proposals)
        return edges_added

    def reject(self, finding_id: str) -> bool:
        """
        Discard a finding from the ring buffer.

        Returns True if the finding was found and removed, False otherwise.
        """
        with self._lock:
            before = len(self._findings)
            # deque has no remove-by-predicate; rebuild without the target
            new_deque: collections.deque = collections.deque(
                maxlen=self._findings.maxlen
            )
            found = False
            for f in self._findings:
                if f.finding_id == finding_id:
                    found = True
                else:
                    new_deque.append(f)
            if not found:
                raise ValueError(f"Finding not found: {finding_id!r}")
            self._findings = new_deque
            return True

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_finding(self, finding_id: str) -> ResearchFinding:
        with self._lock:
            for f in self._findings:
                if f.finding_id == finding_id:
                    return f
        raise ValueError(f"Finding not found: {finding_id!r}")

    def _schedule_next(self) -> None:
        """Schedule the next scan via threading.Timer (recursive pattern from REMEngine)."""
        def _fire():
            try:
                self._run_scan()
            except Exception as exc:
                logger.exception("ResearchAgent scan error: %s", exc)
            finally:
                with self._lock:
                    if self._running:
                        self._schedule_next()

        with self._lock:
            self._timer = threading.Timer(self.scan_interval, _fire)
            self._timer.daemon = True
            self._timer.start()

    def _run_scan(self) -> List[ResearchFinding]:
        """Mine candidates, generate hypotheses, store findings. Returns new findings."""
        t0 = time.time()
        new_findings: List[ResearchFinding] = []

        # Detect graph changes; reset evaluated_pairs if graph grew
        try:
            current_edge_count = self._adapter.to_networkx().number_of_edges()
        except Exception:
            current_edge_count = None
        with self._lock:
            if current_edge_count != self._last_edge_count:
                self._evaluated_pairs.clear()
                self._last_edge_count = current_edge_count

        candidates = self._mine_candidates()

        for cand in candidates:
            pair = (cand.source_id, cand.target_id)
            with self._lock:
                if pair in self._evaluated_pairs:
                    continue
                self._evaluated_pairs.add(pair)

            try:
                proposals = self._hypothesis_engine.generate(
                    source_id=cand.source_id,
                    target_id=cand.target_id,
                    max_hop=self.max_hop,
                    beam_width=self.beam_width,
                    max_budget=self.max_budget,
                )
            except Exception as exc:
                logger.debug("HypothesisEngine.generate(%s, %s) failed: %s",
                             cand.source_id, cand.target_id, exc)
                continue

            if not proposals:
                continue

            good = [p for p in proposals if p.confidence >= self.min_confidence]
            if not good:
                continue

            finding = ResearchFinding(
                finding_id=str(uuid.uuid4()),
                candidate=cand,
                proposals=good,
                best_confidence=max(p.confidence for p in good),
            )

            # Phase 52: validate if validator is attached
            if self._validator is not None:
                try:
                    report = self._validator.validate(good[0])
                    finding.validation_report = report
                    finding.literature_status = report.literature_status
                except Exception as exc:
                    logger.debug("ExternalValidator failed for %s→%s: %s",
                                 cand.source_id, cand.target_id, exc)

            with self._lock:
                self._findings.append(finding)
                self._total_findings += 1

            new_findings.append(finding)

        with self._lock:
            self._total_scans += 1
            self._last_scan_at = time.time()

        elapsed = time.time() - t0
        logger.info(
            "ResearchAgent scan #%d: %d candidates → %d findings (%.2fs)",
            self._total_scans, len(candidates), len(new_findings), elapsed,
        )
        return new_findings

    def _mine_candidates(self) -> List[ResearchCandidate]:
        """
        Collect and rank candidate (source, target) pairs for hypothesis generation.

        Combines:
        1. Embedding similarity scan: pairs with cosine sim ∈ [0.6, 0.95] and no
           direct edge.
        2. InsightEngine hot candidates: recent InsightEvents with no direct edge.

        Returns at most ``candidate_limit`` candidates, sorted by discovery_potential
        descending.
        """
        candidates: List[ResearchCandidate] = []

        try:
            G = self._adapter.to_networkx()
            nodes = list(G.nodes())
            cmap: Dict[str, int] = getattr(self._adapter, "community_map", {}) or {}

            # --- Build embedding cache ---
            emb_cache: Dict[str, Optional[np.ndarray]] = {}
            for node in nodes:
                emb_cache[node] = self._adapter.get_embedding(node)

            # --- Embedding similarity scan ---
            for i, u in enumerate(nodes):
                if len(candidates) >= self.candidate_limit * 3:
                    break  # over-sample before scoring; trim below
                emb_u = emb_cache[u]
                if emb_u is None:
                    continue
                norm_u = float(np.linalg.norm(emb_u))
                if norm_u == 0:
                    continue

                for j in range(i + 1, len(nodes)):
                    v = nodes[j]
                    if G.has_edge(u, v) or G.has_edge(v, u):
                        continue
                    with self._lock:
                        if (u, v) in self._evaluated_pairs or (v, u) in self._evaluated_pairs:
                            continue
                    emb_v = emb_cache[v]
                    if emb_v is None:
                        continue
                    norm_v = float(np.linalg.norm(emb_v))
                    if norm_v == 0:
                        continue

                    similarity = float(np.dot(emb_u, emb_v) / (norm_u * norm_v))
                    # Keep pairs that are semantically related but not trivially similar
                    if not (0.60 <= similarity <= 0.95):
                        continue

                    gap_score = 1.0 - similarity
                    cid_u = cmap.get(u, 0)
                    cid_v = cmap.get(v, 0)
                    community_dist = 0 if cid_u == cid_v else 1

                    potential = self._score_discovery_potential(
                        gap_score=gap_score,
                        community_dist=community_dist,
                        G=G,
                        u=u,
                        v=v,
                        cmap=cmap,
                    )
                    if potential < self.min_discovery_potential:
                        continue

                    candidates.append(ResearchCandidate(
                        source_id=u,
                        target_id=v,
                        discovery_potential=potential,
                        gap_score=gap_score,
                        community_distance=community_dist,
                        seeded_by="embedding_scan",
                    ))

            # --- InsightEngine seeding ---
            if self._insight_engine is not None:
                try:
                    events = self._insight_engine.recent_events(50)
                    for evt in events:
                        u, v = evt.source, evt.target
                        if G.has_edge(u, v) or G.has_edge(v, u):
                            continue
                        with self._lock:
                            if (u, v) in self._evaluated_pairs:
                                continue
                        emb_u = emb_cache.get(u)
                        emb_v = emb_cache.get(v)
                        if emb_u is None or emb_v is None:
                            gap_score = 0.5
                        else:
                            norm_u = float(np.linalg.norm(emb_u))
                            norm_v = float(np.linalg.norm(emb_v))
                            if norm_u > 0 and norm_v > 0:
                                sim = float(np.dot(emb_u, emb_v) / (norm_u * norm_v))
                                gap_score = 1.0 - sim
                            else:
                                gap_score = 0.5

                        cid_u = cmap.get(u, 0)
                        cid_v = cmap.get(v, 0)
                        community_dist = 0 if cid_u == cid_v else 1

                        potential = self._score_discovery_potential(
                            gap_score=gap_score,
                            community_dist=community_dist,
                            G=G,
                            u=u,
                            v=v,
                            cmap=cmap,
                        )
                        candidates.append(ResearchCandidate(
                            source_id=u,
                            target_id=v,
                            discovery_potential=potential,
                            gap_score=gap_score,
                            community_distance=community_dist,
                            seeded_by="insight_engine",
                        ))
                except Exception as exc:
                    logger.debug("InsightEngine seeding error: %s", exc)

        except Exception as exc:
            logger.warning("ResearchAgent._mine_candidates() error: %s", exc)
            return []

        # Rank by discovery_potential and cap
        candidates.sort(key=lambda c: c.discovery_potential, reverse=True)
        return candidates[:self.candidate_limit]

    def _score_discovery_potential(
        self,
        gap_score: float,
        community_dist: int,
        G,
        u: str,
        v: str,
        cmap: Dict[str, int],
    ) -> float:
        """
        Compute discovery_potential ∈ [0, 1].

        Formula:
            0.4 * gap_score           (semantic gap — want high)
          + 0.4 * (1 − conn_density)  (sparse connection zone — unexplored territory)
          + 0.2 * cross_community     (1.0 if different communities, 0.0 if same)
        """
        cross_community = 1.0 if community_dist > 0 else 0.0

        # Connection density: fraction of possible pairs within the local neighbourhood
        # that are actually connected.  Use 2-hop ego subgraph as proxy.
        try:
            u_neighbors = set(G.successors(u)) | set(G.predecessors(u))
            v_neighbors = set(G.successors(v)) | set(G.predecessors(v))
            local_nodes = u_neighbors | v_neighbors | {u, v}
            local_sub = G.subgraph(local_nodes)
            n = len(local_nodes)
            possible = n * (n - 1)  # directed
            actual = local_sub.number_of_edges()
            conn_density = actual / possible if possible > 0 else 0.0
        except Exception:
            conn_density = 0.0

        potential = (
            0.4 * float(gap_score)
            + 0.4 * (1.0 - min(1.0, conn_density))
            + 0.2 * cross_community
        )
        return min(1.0, max(0.0, potential))
