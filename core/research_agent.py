"""
ResearchAgent — Autonomous Missing-Link Discovery Daemon (Phase 51+).

Continuously mines (source, target) candidate pairs from the graph, calls
HypothesisEngine.generate() on each, and accumulates findings in a fixed-size
ring buffer for human review and optional materialization.

Four candidate sources (combined and ranked by discovery potential):
  1. ANN embedding scan     — batched numpy dot-product finds semantically
                              related pairs in O(N·K) rather than O(N²).
                              Replaces the previous Python nested loop.
  2. Structural hole scan   — nodes with high degree and low local clustering
                              are Burt-constraint bridges; cross-community
                              targets around these hubs are high-value finds.
  3. InsightEngine seeding  — cross-community paths already flagged as
                              surprising by the InsightEngine.
  4. Manual override        — external callers push (source, target) directly.

Improvements over the original Phase 51 implementation:
  - _pushed_candidates initialised in __init__ (no more lazy hasattr dance).
  - Graph-signature delta dedup: _evaluated_pairs only resets when the graph
    grows by >1%; single-edge materializations no longer invalidate history.
  - All good proposals are validated together (validate_batch), not just the
    top-1; the most novel report sets the finding's literature_status.
  - Real inter-community distance from CSA metadata (not binary 0/1).
  - Optional Engram/SpeedTalkEngram: affinity stored as finding metadata so
    Engram-endorsed findings can be prioritised for approval.
  - Research thread clustering via thread_findings() — groups findings by
    shared intersection hub nodes using union-find.
  - Feedback learning via report_outcome() — per-community-pair weights
    updated online (EMA α=0.2) so the agent learns from human approvals.

Usage
-----
    from core.research_agent import ResearchAgent
    from core.hypothesis_engine import HypothesisEngine

    hypothesis_engine = HypothesisEngine(adapter)
    agent = ResearchAgent(adapter, hypothesis_engine)

    findings = agent.scan_once()           # one-shot blocking scan
    agent.start()                           # start background daemon
    agent.approve(findings[0].finding_id)  # materialise proposals to graph
    agent.report_outcome(findings[0].finding_id, correct=True)
    agent.stop()
"""
from __future__ import annotations

import collections
import logging
import threading
import time
import uuid
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# Phase 73 Batch B — optional components (imported lazily to avoid hard dependency)
try:
    from core.contradiction_resolver import ContradictionResolver, ContradictionRecord  # noqa: F401
except ImportError:
    ContradictionResolver = None  # type: ignore[assignment,misc]

try:
    from core.candidate_registry import CandidateRegistry  # noqa: F401
except ImportError:
    CandidateRegistry = None  # type: ignore[assignment,misc]

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
    """Origin: ``"embedding_scan"`` | ``"structural_hole"`` | ``"insight_engine"`` | ``"manual"``."""

    local_density: float = 0.0
    """Connection density of the local 2-hop neighbourhood (0 = sparse, 1 = dense)."""

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
    """Set by ExternalValidator: novel | active_research | established | contested."""

    validation_report: Optional[Any] = None
    """Best ValidationReport from Phase 52 (highest novelty_score); None until validated."""

    found_at: float = field(default_factory=time.time)

    metadata: Dict[str, Any] = field(default_factory=dict)
    """
    Extensible metadata bag.  Keys set by the agent:
      ``engram_affinity`` : float — how well the derived relation matches the
                            Engram's known-productive patterns (0–1).
      ``n_reports``       : int   — number of proposals validated.
    """


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
    engram
        Optional Engram or SpeedTalkEngram.  When provided, the affinity of
        each finding's derived relation to cached patterns is stored in
        ``finding.metadata["engram_affinity"]``.
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
    ann_batch_size
        Rows per batch in the ANN embedding scan (tune for memory budget).
    ann_top_k
        Maximum nearest-neighbour candidates to inspect per node.
    """

    def __init__(
        self,
        adapter,
        hypothesis_engine,
        insight_engine=None,
        engram=None,
        auto_approver=None,
        triangulation_engine=None,
        calibrator=None,
        scan_interval: float = 300.0,
        candidate_limit: int = 100,
        min_discovery_potential: float = 0.30,
        min_confidence: float = 0.20,
        findings_capacity: int = 500,
        max_hop: int = 3,
        beam_width: int = 8,
        max_budget: int = 300,
        ann_batch_size: int = 512,
        ann_top_k: int = 30,
        larql_endpoint: Optional[str] = None,
        larql_vindex: Optional[str] = None,
    ) -> None:
        self._adapter               = adapter
        self._hypothesis_engine     = hypothesis_engine
        self._insight_engine        = insight_engine
        self._engram                = engram
        self._auto_approver         = auto_approver
        self._triangulation_engine  = triangulation_engine
        self._calibrator            = calibrator
        # Phase 73 Batch B
        self._contradiction_resolver: Optional[Any] = None
        self._registry: Optional[Any]               = None
        self.scan_interval      = scan_interval
        self.candidate_limit    = candidate_limit
        self.min_discovery_potential = min_discovery_potential
        self.min_confidence     = min_confidence
        self.max_hop            = max_hop
        self.beam_width         = beam_width
        self.max_budget         = max_budget
        self.ann_batch_size     = ann_batch_size
        self.ann_top_k          = ann_top_k

        # LARQL Neural Discovery
        from core.larql_client import LarqlClient
        self._larql = LarqlClient(endpoint=larql_endpoint, vindex_path=larql_vindex)

        self._lock              = threading.RLock()
        self._timer: Optional[threading.Timer] = None
        self._running: bool     = False
        self._total_scans: int  = 0
        self._total_findings: int = 0
        self._last_scan_at: Optional[float] = None

        # Graph signature for smarter dedup: (node_count, edge_count).
        # _evaluated_pairs only resets when the graph grows by >1%, so
        # single-edge materializations don't invalidate scan history.
        self._graph_signature: Optional[Tuple[int, int]] = None

        # Ring buffers and state
        self._findings: collections.deque = collections.deque(maxlen=findings_capacity)
        self._evaluated_pairs: Set[Tuple[str, str]] = set()
        self._pushed_candidates: List[ResearchCandidate] = []
        # Phase 73 Batch B: revision candidates (discardable → skip; revision → queue here)
        self._revision_candidates: collections.deque = collections.deque(maxlen=50)
        # Phase 76: provenance ledger (optional — attached via set_provenance_ledger)
        self._provenance_ledger: Optional[Any] = None

        # Phase 98: working memory + goal hints (optional attachments)
        self._wm: Optional[Any] = None
        self._goal_hints: List[str] = []
        # Phase 101: emotional valence engine (optional)
        self._valence_engine: Optional[Any] = None

        # Optional external validator (set by server or constructor caller)
        self._validator: Optional[Any] = None

        # Feedback learning: (min_cid, max_cid) → weight multiplier in [0.3, 2.0]
        self._feedback_weights: Dict[Tuple[int, int], float] = {}

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

    def set_auto_approver(self, approver) -> None:
        """
        Attach an AutoApprover (Phase 71).  When set, each finding that enters
        the ring buffer is evaluated automatically and may be approved or
        rejected without human intervention.
        """
        with self._lock:
            self._auto_approver = approver

    def set_triangulation_engine(self, engine) -> None:
        """
        Attach a TriangulationEngine (Phase 72).  When set, each finding is
        evaluated from four independent perspectives before entering the ring
        buffer.  Results are stored in ``finding.metadata["triangulation"]``
        and feed features 12–15 of the AutoApprover classifier.
        """
        with self._lock:
            self._triangulation_engine = engine

    def set_contradiction_resolver(self, resolver) -> None:
        """
        Attach a ContradictionResolver (Phase 73 Batch B).  When set, findings
        with non-trivial contradiction_score are classified before reaching
        AutoApprover.  Discardable findings are auto-rejected; revision
        candidates are queued in ``_revision_candidates``.
        """
        with self._lock:
            self._contradiction_resolver = resolver

    def set_registry(self, registry) -> None:
        """
        Attach a CandidateRegistry (Phase 73 Batch B).  When set, replaces the
        flat ``_evaluated_pairs`` dedup with TTL-aware tracking + nomination
        boost on ``discovery_potential`` scoring.
        """
        with self._lock:
            self._registry = registry

    def set_calibrator(self, calibrator) -> None:
        """
        Attach a DiscoveryCalibrator (Phase 73).  When set, candidate scoring
        is reweighted by per-community inverse discovery rate so underrepresented
        communities receive proportionally more exploration effort.
        """
        with self._lock:
            self._calibrator = calibrator

    def set_provenance_ledger(self, ledger) -> None:
        """
        Attach a ProvenanceLedger (Phase 76).  When set, every approve() call
        records the materialized edges so they can be rolled back by batch or
        cycle number.
        """
        with self._lock:
            self._provenance_ledger = ledger

    def set_working_memory(self, wm: Any) -> None:
        """Attach a WorkingMemoryBuffer so approve() records outcomes (Gap 5)."""
        with self._lock:
            self._wm = wm

    def set_goal_hints(self, entities: List[str]) -> None:
        """Set goal-hinted entity IDs to boost discovery scoring (Gap 7)."""
        with self._lock:
            self._goal_hints = list(entities)

    def set_valence_engine(self, engine: Any) -> None:
        """Attach a ValenceEngine for emotional conditioning (Phase 101)."""
        with self._lock:
            self._valence_engine = engine

    def push_candidate(self, candidate: ResearchCandidate) -> bool:
        """
        Externally push a candidate for evaluation in the next scan cycle.
        Returns True if the candidate was added, False if already evaluated.
        """
        pair = (candidate.source_id, candidate.target_id)
        with self._lock:
            if pair in self._evaluated_pairs:
                return False
            self._pushed_candidates.append(candidate)
            return True

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

    def approve(self, finding_id: str, cycle_number: Optional[int] = None) -> int:
        """
        Materialise all proposals in a finding to the graph.

        Parameters
        ----------
        finding_id:
            ID of the finding to approve.
        cycle_number:
            Optional AutonomousDiscoveryLoop cycle index.  Passed through to
            the ProvenanceLedger (Phase 76) so edges can later be rolled back
            by cycle.

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

        # Phase 76: record to provenance ledger if attached
        ledger = self._provenance_ledger
        if ledger is not None and edges_added > 0:
            edge_triples = [
                (p.source_id, p.target_id, p.relation)
                for p in finding.proposals
                if hasattr(p, "source_id") and hasattr(p, "relation")
            ]
            if edge_triples:
                ledger.record_batch(
                    batch_id=finding_id,
                    finding_id=finding_id,
                    edges=edge_triples[:edges_added],
                    cycle_number=cycle_number,
                )

        # Phase 101: record positive valence for approved edges
        valence_engine = getattr(self, "_valence_engine", None)
        if valence_engine is not None and edges_added > 0:
            for p in finding.proposals[:edges_added]:
                src = getattr(p, "source_id", None)
                tgt = getattr(p, "target_id", None)
                rel = getattr(p, "relation", "")
                conf = getattr(p, "confidence", 1.0)
                if src and tgt:
                    try:
                        valence_engine.record_outcome(
                            [(src, rel, tgt)], outcome_score=float(conf)
                        )
                    except Exception:
                        pass

        # Phase 98 Gap 5: record approval to working memory
        wm = self._wm
        if wm is not None and edges_added > 0:
            try:
                import time as _time
                from core.working_memory import MemoryEntry
                for p in finding.proposals[:edges_added]:
                    src = getattr(p, "source_id", None)
                    tgt = getattr(p, "target_id", None)
                    rel = getattr(p, "relation", "")
                    conf = getattr(p, "confidence", 1.0)
                    if src and tgt:
                        wm.record(MemoryEntry(
                            timestamp=_time.time(),
                            seeds=[src],
                            answers=[tgt],
                            top_score=float(conf),
                            soliton_index=None,
                            prediction_error=None,
                            source="approval",
                            path_edges=[(src, rel, tgt)],
                        ))
            except Exception:
                pass

        return edges_added

    def reject(self, finding_id: str) -> bool:
        """
        Discard a finding from the ring buffer.

        Returns True if the finding was found and removed, False otherwise.
        """
        rejected_finding = None
        with self._lock:
            new_deque: collections.deque = collections.deque(
                maxlen=self._findings.maxlen
            )
            found = False
            for f in self._findings:
                if f.finding_id == finding_id:
                    found = True
                    rejected_finding = f
                else:
                    new_deque.append(f)
            if not found:
                raise ValueError(f"Finding not found: {finding_id!r}")
            self._findings = new_deque

        # Phase 101: record negative valence for rejected path
        valence_engine = getattr(self, "_valence_engine", None)
        if valence_engine is not None and rejected_finding is not None:
            for p in getattr(rejected_finding, "proposals", []):
                src = getattr(p, "source_id", None)
                tgt = getattr(p, "target_id", None)
                rel = getattr(p, "relation", "")
                if src and tgt:
                    try:
                        valence_engine.record_outcome([(src, rel, tgt)], outcome_score=-1.0)
                    except Exception:
                        pass

        return True

    def report_outcome(self, finding_id: str, correct: bool) -> None:
        """
        Report whether a finding's hypothesis was correct after human review.

        Updates per-community-pair weights (EMA α=0.2) so the agent learns
        from human approvals and rejections:
          correct=True  → boosts future candidates between the same communities
          correct=False → penalises them

        Weights are clamped to [0.3, 2.0] to prevent runaway adaptation.
        """
        try:
            finding = self._get_finding(finding_id)
        except ValueError:
            return

        cmap = getattr(self._adapter, "community_map", {}) or {}
        cid_source = cmap.get(finding.candidate.source_id, 0)
        cid_target = cmap.get(finding.candidate.target_id, 0)
        key = (min(cid_source, cid_target), max(cid_source, cid_target))

        with self._lock:
            current = self._feedback_weights.get(key, 1.0)
            alpha = 0.2
            signal = 1.2 if correct else 0.8
            updated = (1 - alpha) * current + alpha * signal
            self._feedback_weights[key] = min(2.0, max(0.3, updated))

    def thread_findings(self) -> Dict[str, List[ResearchFinding]]:
        """
        Group findings into research threads by shared intersection hub nodes.

        Two findings belong to the same thread when their proposals share at
        least one intersection node — the structural hub through which both
        hypotheses route.  Uses union-find (path-compressed) for O(F·H·α).

        Returns
        -------
        Dict[str, List[ResearchFinding]]
            Mapping from thread root key → list of findings in that thread.
            Findings with no intersection nodes form singleton threads keyed
            by their own finding_id.
        """
        with self._lock:
            findings = list(self._findings)

        if not findings:
            return {}

        # finding_id -> set of intersection hub nodes across all proposals
        finding_hubs: Dict[str, Set[str]] = {}
        for f in findings:
            hubs: Set[str] = set()
            for p in f.proposals:
                hubs.update(getattr(p, "intersection_nodes", []) or [])
            finding_hubs[f.finding_id] = hubs

        # Union-find with path compression
        parent: Dict[str, str] = {f.finding_id: f.finding_id for f in findings}

        def _find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # path halving
                x = parent[x]
            return x

        def _union(a: str, b: str) -> None:
            ra, rb = _find(a), _find(b)
            if ra != rb:
                parent[rb] = ra

        # Merge findings that share a hub
        hub_owner: Dict[str, str] = {}
        for fid, hubs in finding_hubs.items():
            for hub in hubs:
                if hub in hub_owner:
                    _union(fid, hub_owner[hub])
                else:
                    hub_owner[hub] = fid

        # Collect into threads
        threads: Dict[str, List[ResearchFinding]] = {}
        fid_map = {f.finding_id: f for f in findings}
        for f in findings:
            root = _find(f.finding_id)
            threads.setdefault(root, []).append(f)

        return threads

    # ------------------------------------------------------------------
    # Internal — scheduling
    # ------------------------------------------------------------------

    def _get_finding(self, finding_id: str) -> ResearchFinding:
        with self._lock:
            for f in self._findings:
                if f.finding_id == finding_id:
                    return f
        raise ValueError(f"Finding not found: {finding_id!r}")

    def _schedule_next(self) -> None:
        """Schedule the next scan via threading.Timer (recursive daemon pattern)."""
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

    # ------------------------------------------------------------------
    # Internal — main scan loop
    # ------------------------------------------------------------------

    def _run_scan(self) -> List[ResearchFinding]:
        """Mine candidates, generate hypotheses, store findings. Returns new findings."""
        t0 = time.time()
        new_findings: List[ResearchFinding] = []
        auto_count: int = 0  # tracks automatic decisions this scan cycle

        # Smarter dedup reset: only clear evaluated_pairs when the graph has
        # grown significantly (>1% change) — single-edge materializations don't
        # invalidate the full scan history.
        try:
            G_snapshot = self._adapter.to_networkx()
            current_sig = (G_snapshot.number_of_nodes(), G_snapshot.number_of_edges())
        except Exception:
            current_sig = None

        with self._lock:
            prev_sig = self._graph_signature
            if current_sig is not None:
                if prev_sig is None or (
                    current_sig[0] > prev_sig[0] * 1.01
                    or current_sig[1] > prev_sig[1] * 1.01
                ):
                    self._evaluated_pairs.clear()
                self._graph_signature = current_sig

        candidates = self._mine_candidates()

        for cand in candidates:
            pair = (cand.source_id, cand.target_id)
            # Registry-aware dedup (Phase 73 Batch B): when registry is attached,
            # it controls whether to re-evaluate based on TTL.  Fall back to the
            # flat _evaluated_pairs set when no registry is attached.
            if self._registry is not None:
                if not self._registry.should_evaluate(cand):
                    continue
            else:
                with self._lock:
                    if pair in self._evaluated_pairs:
                        continue
                    self._evaluated_pairs.add(pair)

            try:
                strategy = self._select_strategy(cand.local_density)
                logger.debug(
                    "ResearchAgent strategy for (%s->%s): density=%.3f "
                    "-> hop=%d beam=%d budget=%d",
                    cand.source_id, cand.target_id, cand.local_density,
                    strategy["max_hop"], strategy["beam_width"], strategy["max_budget"],
                )
                proposals = self._hypothesis_engine.generate(
                    source_id=cand.source_id,
                    target_id=cand.target_id,
                    **strategy,
                )
            except Exception as exc:
                logger.debug("HypothesisEngine.generate(%s, %s) failed: %s",
                             cand.source_id, cand.target_id, exc)
                continue

            if not proposals:
                # Direction fallback: structural-hole mining may generate the candidate
                # in the wrong direction (e.g. epilepsy→drug when only drug→epilepsy
                # paths exist).  Retry with swapped endpoints and, if successful,
                # re-orient the candidate so downstream code stays consistent.
                try:
                    rev_proposals = self._hypothesis_engine.generate(
                        source_id=cand.target_id,
                        target_id=cand.source_id,
                        **strategy,
                    )
                except Exception:
                    rev_proposals = []
                if rev_proposals:
                    logger.debug(
                        "ResearchAgent: reversed candidate direction %s->%s "
                        "(forward had 0 proposals, reverse has %d)",
                        cand.target_id, cand.source_id, len(rev_proposals),
                    )
                    proposals = rev_proposals
                    # Re-orient the candidate so source/target match the confirmed path
                    from dataclasses import replace as _dc_replace
                    cand = _dc_replace(
                        cand,
                        source_id=cand.target_id,
                        target_id=cand.source_id,
                    )
                else:
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

            # Phase 52+: validate ALL good proposals together, not just good[0].
            # Use the report with the highest novelty_score to set literature_status.
            if self._validator is not None:
                try:
                    if hasattr(self._validator, "validate_batch"):
                        reports = self._validator.validate_batch(good)
                    else:
                        reports = [self._validator.validate(p) for p in good]

                    if reports:
                        best_report = max(reports, key=lambda r: r.novelty_score)
                        finding.literature_status = best_report.literature_status
                        finding.validation_report = best_report
                        finding.metadata["n_reports"] = len(reports)
                except Exception as exc:
                    logger.debug("ExternalValidator failed for %s->%s: %s",
                                 cand.source_id, cand.target_id, exc)

            # Engram affinity: how well the derived relation matches known-
            # productive patterns in the Engram cache.
            if self._engram is not None:
                try:
                    if hasattr(self._engram, "affinity"):
                        aff = max(
                            self._engram.affinity([p.derived_relation]) for p in good
                        )
                        finding.metadata["engram_affinity"] = round(aff, 3)
                except Exception:
                    pass

            # Multi-perspective triangulation (Phase 72).
            # Runs four independent validation perspectives and stores the
            # TriangulationReport in finding.metadata["triangulation"] so
            # the AutoApprover can read features 12–15.
            if self._triangulation_engine is not None:
                try:
                    tri_report = self._triangulation_engine.evaluate(cand, good)
                    finding.metadata["triangulation"] = tri_report
                    if tri_report.is_SynapticBridge_candidate:
                        best_rel = good[0].derived_relation if good else "?"
                        logger.info(
                            "TriangulationEngine: SynapticBridge candidate %s -[%s]-> %s "
                            "(rev=%.2f agree=%.2f indep=%.2f sem=%.2f)",
                            cand.source_id, best_rel, cand.target_id,
                            tri_report.reverse_confidence,
                            tri_report.strategy_agreement,
                            tri_report.mean_path_independence,
                            tri_report.semantic_type_score,
                        )
                except Exception as exc:
                    logger.error(
                        "TriangulationEngine error for %s->%s: %s",
                        cand.source_id, cand.target_id, exc,
                    )

            # ContradictionResolver — Phase 73 Batch B.
            # Classifies findings with meaningful contradiction signals before
            # they reach the AutoApprover.  "discardable" findings are rejected
            # here and never enter the ring buffer or the AutoApprover queue.
            if self._contradiction_resolver is not None:
                try:
                    cr = self._contradiction_resolver.resolve(finding, good)
                    finding.metadata["contradiction_resolution"] = cr
                    if cr.resolution == "discardable":
                        logger.info(
                            "ContradictionResolver: discarding %s "
                            "(nor=%.3f contra=%.3f net=%.3f)",
                            finding.finding_id, cr.proposed_noisy_or,
                            cr.contradiction_score, cr.net_evidence_score,
                        )
                        # Teach AutoApprover to recognise this pattern
                        aa = self._auto_approver
                        if aa is not None:
                            aa.fit(finding, approved=False)
                        # Skip ring-buffer, calibrator, AutoApprover
                        continue
                    if cr.resolution == "revision_candidate":
                        logger.info(
                            "ContradictionResolver: revision candidate %s "
                            "(rev_weight=%.2f)",
                            finding.finding_id, cr.revision_weight,
                        )
                        with self._lock:
                            self._revision_candidates.append(finding)
                        # Fall through — still enters ring buffer for human review
                except Exception as exc:
                    logger.error(
                        "ContradictionResolver error for %s: %s",
                        finding.finding_id, exc,
                    )

            # Registry: mark candidate as evaluated (Phase 73 Batch B)
            if self._registry is not None:
                try:
                    self._registry.mark_evaluated(cand)
                except Exception:
                    pass

            with self._lock:
                self._findings.append(finding)
                self._total_findings += 1

            new_findings.append(finding)

            # Inform calibrator a discovery was made in this community (Phase 73)
            if self._calibrator is not None:
                cmap_now = getattr(self._adapter, "community_map", {}) or {}
                src_cid = cmap_now.get(cand.source_id, 0)
                self._calibrator.record_discovery(src_cid)

            # Auto-approve / auto-reject if an AutoApprover is attached.
            # report_outcome() is called before approve/reject so it can still
            # look up the finding in the ring buffer if needed.
            aa = self._auto_approver
            if aa is not None and auto_count < aa.policy.max_auto_per_scan:
                try:
                    decision = aa.decide(finding)
                    if decision.action == "approve":
                        self.report_outcome(finding.finding_id, correct=True)
                        self.approve(finding.finding_id)
                        aa.fit(finding, approved=True)
                        auto_count += 1
                        logger.info(
                            "AutoApprover approved %s (%s)",
                            finding.finding_id, decision.reason,
                        )
                    elif decision.action == "reject":
                        self.report_outcome(finding.finding_id, correct=False)
                        self.reject(finding.finding_id)
                        aa.fit(finding, approved=False)
                        auto_count += 1
                        logger.info(
                            "AutoApprover rejected %s (%s)",
                            finding.finding_id, decision.reason,
                        )
                    # "review" → stays in ring buffer for human
                except Exception as exc:
                    logger.error("AutoApprover error for %s: %s", finding.finding_id, exc)

        with self._lock:
            self._total_scans += 1
            self._last_scan_at = time.time()

        elapsed = time.time() - t0
        logger.info(
            "ResearchAgent scan #%d: %d candidates -> %d findings (%.2fs)",
            self._total_scans, len(candidates), len(new_findings), elapsed,
        )
        return new_findings

    # ------------------------------------------------------------------
    # Internal — candidate mining
    # ------------------------------------------------------------------

    def _mine_candidates(self) -> List[ResearchCandidate]:
        """
        Collect and rank candidate (source, target) pairs for hypothesis generation.

        Combines four sources:
          1. Pushed candidates (highest priority — from CerebellarEngine / manual).
          2. ANN embedding scan (vectorised numpy, replaces the O(N²) Python loop).
          3. Structural hole detection (degree × (1-clustering) Burt approximation).
          4. InsightEngine hot candidates.

        Returns at most ``candidate_limit`` candidates, sorted by
        discovery_potential descending.
        """
        candidates: List[ResearchCandidate] = []
        seen_pairs: Set[Tuple[str, str]] = set()

        try:
            # --- 1. Pushed candidates (highest priority) ---
            with self._lock:
                if self._pushed_candidates:
                    for c in self._pushed_candidates:
                        candidates.append(c)
                        seen_pairs.add((c.source_id, c.target_id))
                    self._pushed_candidates = []

            G = self._adapter.to_networkx()
            nodes = list(G.nodes())
            cmap: Dict[str, int] = getattr(self._adapter, "community_map", {}) or {}

            # Load real inter-community distances from CSA metadata if available
            csa_meta = getattr(self._adapter, "_csa_metadata", None) or {}
            self._comm_distances: Dict[Tuple[int, int], float] = (
                csa_meta.get("distances") or {}
            )

            # Build embedding cache
            emb_cache: Dict[str, Optional[np.ndarray]] = {}
            for node in nodes:
                emb_cache[node] = self._adapter.get_embedding(node)

            # --- 2. ANN embedding scan ---
            ann_cands = self._ann_scan(G, nodes, emb_cache, cmap, seen_pairs)
            candidates.extend(ann_cands)
            for c in ann_cands:
                seen_pairs.add((c.source_id, c.target_id))

            # --- 3. Structural hole scan ---
            hole_cands = self._structural_hole_candidates(G, nodes, cmap, seen_pairs)
            candidates.extend(hole_cands)
            for c in hole_cands:
                seen_pairs.add((c.source_id, c.target_id))

            # --- 4. InsightEngine seeding ---
            if self._insight_engine is not None:
                try:
                    events = self._insight_engine.recent_events(50)
                    for evt in events:
                        u, v = evt.source, evt.target
                        if (u, v) in seen_pairs or (v, u) in seen_pairs:
                            continue
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
                            nu = float(np.linalg.norm(emb_u))
                            nv = float(np.linalg.norm(emb_v))
                            if nu > 0 and nv > 0:
                                sim = float(np.dot(emb_u, emb_v) / (nu * nv))
                                gap_score = 1.0 - sim
                            else:
                                gap_score = 0.5

                        cid_u = cmap.get(u, 0)
                        cid_v = cmap.get(v, 0)
                        community_dist = 0 if cid_u == cid_v else 1
                        potential, density = self._score_discovery_potential(
                            gap_score=gap_score,
                            community_dist=community_dist,
                            G=G,
                            u=u,
                            v=v,
                            cmap=cmap,
                        )
                        seen_pairs.add((u, v))
                        candidates.append(ResearchCandidate(
                            source_id=u,
                            target_id=v,
                            discovery_potential=potential,
                            gap_score=gap_score,
                            community_distance=community_dist,
                            seeded_by="insight_engine",
                            local_density=density,
                        ))
                except Exception as exc:
                    logger.debug("InsightEngine seeding error: %s", exc)

            # --- 5. LARQL Neural Discovery scan ---
            larql_cands = self._larql_scan(G, nodes, emb_cache, cmap, seen_pairs)
            candidates.extend(larql_cands)
            for c in larql_cands:
                seen_pairs.add((c.source_id, c.target_id))

        except Exception as exc:
            logger.warning("ResearchAgent._mine_candidates() error: %s", exc)
            return []

        # Registry: register all candidates + apply nomination boost (Phase 73 Batch B)
        if self._registry is not None:
            try:
                for c in candidates:
                    self._registry.register(c)
                    boost = self._registry.get_nomination_boost(c)
                    if boost != 1.0:
                        c.discovery_potential = min(1.0, c.discovery_potential * boost)
            except Exception as exc:
                logger.debug("CandidateRegistry boost error: %s", exc)

        # Rank by discovery_potential and cap at candidate_limit
        candidates.sort(key=lambda c: c.discovery_potential, reverse=True)
        final = candidates[:self.candidate_limit]

        # Inform calibrator which communities were sampled this cycle (Phase 73)
        if self._calibrator is not None and final:
            sampled_communities = {
                cmap.get(c.source_id, 0) for c in final
            }
            self._calibrator.record_scan(sampled_communities)

        return final

    def _ann_scan(
        self,
        G,
        nodes: List[str],
        emb_cache: Dict[str, Optional[np.ndarray]],
        cmap: Dict[str, int],
        seen_pairs: Set[Tuple[str, str]],
    ) -> List[ResearchCandidate]:
        """
        Vectorised ANN embedding scan via batched BLAS matrix multiply.

        Builds a float32 (N, d) embedding matrix and processes rows in
        batches of ``ann_batch_size``, computing cosine similarities via
        dot-product (embeddings are L2-normalised).  Only upper-triangular
        pairs are inspected (j > global_i) to avoid duplicates.

        Time:   O(N · K · d) via BLAS, ~1000x faster than the Python loop.
        Memory: O(batch_size · N) per batch (configurable via ann_batch_size).
        """
        valid_nodes = [n for n in nodes if emb_cache.get(n) is not None]
        N = len(valid_nodes)
        if N < 2:
            return []

        # Stack into (N, d) float32; re-normalise to guard against non-unit vecs
        vectors = []
        for n in valid_nodes:
            v = emb_cache[n]
            if v is not None:
                vectors.append(v.astype(np.float32))
        
        if not vectors:
            return []
            
        E = np.stack(vectors).astype(np.float32)
        norms = np.linalg.norm(E, axis=1, keepdims=True).astype(np.float32)
        norms = np.where(norms > 1e-8, norms, 1.0).astype(np.float32)
        E = (E / norms).astype(np.float32)

        candidates: List[ResearchCandidate] = []
        budget = self.candidate_limit * 3  # oversample before final sort+trim

        for batch_start in range(0, N, self.ann_batch_size):
            if len(candidates) >= budget:
                break

            batch_end = min(batch_start + self.ann_batch_size, N)
            batch_E = E[batch_start:batch_end]          # (B, d)
            sims = batch_E @ E.T                        # (B, N) — BLAS sgemm

            for bi in range(batch_end - batch_start):
                global_i = batch_start + bi
                u = valid_nodes[global_i]
                row = sims[bi].copy()

                # Zero out lower-triangle and self to process each pair once
                row[: global_i + 1] = 0.0

                # Filter to similarity window [0.60, 0.95]
                eligible = np.where((row >= 0.60) & (row <= 0.95))[0]
                if len(eligible) == 0:
                    continue

                # Keep top-K by similarity
                if len(eligible) > self.ann_top_k:
                    order = np.argsort(row[eligible])[-self.ann_top_k:]
                    eligible = eligible[order]

                for j in eligible:
                    v = valid_nodes[j]
                    if (u, v) in seen_pairs or (v, u) in seen_pairs:
                        continue
                    if G.has_edge(u, v) or G.has_edge(v, u):
                        continue
                    with self._lock:
                        if (u, v) in self._evaluated_pairs or (v, u) in self._evaluated_pairs:
                            continue

                    similarity = float(row[j])
                    gap_score = 1.0 - similarity
                    cid_u = cmap.get(u, 0)
                    cid_v = cmap.get(v, 0)
                    community_dist = 0 if cid_u == cid_v else 1

                    potential, density = self._score_discovery_potential(
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
                        local_density=density,
                    ))
                    seen_pairs.add((u, v))

        return candidates

    def _structural_hole_candidates(
        self,
        G,
        nodes: List[str],
        cmap: Dict[str, int],
        seen_pairs: Set[Tuple[str, str]],
        top_holes: int = 20,
        targets_per_hole: int = 5,
    ) -> List[ResearchCandidate]:
        """
        Burt-constraint approximation via degree × (1 - clustering).

        Nodes with high degree and low local clustering are structural holes:
        they sit between otherwise disconnected regions.  Cross-community
        pairs that route THROUGH these nodes are high-value hypothesis targets.

        Clustering is O(|E|) via NetworkX; betweenness is not recomputed here
        (too expensive) — degree is used as a proxy instead.
        """
        if G.number_of_nodes() < 4:
            return []

        try:
            import networkx as nx
            raw_clust = nx.clustering(G)
            if isinstance(raw_clust, dict):
                clust: Dict[str, float] = {str(k): float(v) for k, v in raw_clust.items()}
            else:
                return []
        except Exception:
            return []

        # Hole score: degree × (1 − clustering); high = genuine bridge
        hole_scores: Dict[str, float] = {}
        for n in nodes:
            deg = G.degree(n)
            if deg < 2:
                continue
            hole_scores[n] = deg * (1.0 - clust.get(n, 0.0))

        if not hole_scores:
            return []

        max_hole = max(hole_scores.values())
        top = sorted(hole_scores, key=hole_scores.__getitem__, reverse=True)[:top_holes]

        candidates: List[ResearchCandidate] = []
        for h in top:
            cid_h = cmap.get(h, -1)
            h_nbrs = set(G.successors(h)) | set(G.predecessors(h))

            # Cross-community nodes not directly connected to h
            cross = [
                v for v in nodes
                if v != h
                and cmap.get(v, -2) != cid_h
                and v not in h_nbrs
            ]
            if not cross:
                continue

            for v in cross[:targets_per_hole]:
                if (h, v) in seen_pairs or (v, h) in seen_pairs:
                    continue
                if G.has_edge(h, v) or G.has_edge(v, h):
                    continue
                with self._lock:
                    if (h, v) in self._evaluated_pairs or (v, h) in self._evaluated_pairs:
                        continue

                cid_v = cmap.get(v, 0)
                community_dist = 0 if cid_h == cid_v else 1
                # gap_score unknown (no embedding comparison); use neutral 0.5
                potential, density = self._score_discovery_potential(
                    gap_score=0.5,
                    community_dist=community_dist,
                    G=G,
                    u=h,
                    v=v,
                    cmap=cmap,
                )
                # Boost by normalised hole score (max +0.2)
                hole_boost = 0.2 * (hole_scores[h] / max(1.0, max_hole))
                potential = min(1.0, potential + hole_boost)

                candidates.append(ResearchCandidate(
                    source_id=h,
                    target_id=v,
                    discovery_potential=potential,
                    gap_score=0.5,
                    community_distance=community_dist,
                    seeded_by="structural_hole",
                    local_density=density,
                ))
                seen_pairs.add((h, v))

        return candidates

    def _larql_scan(
        self,
        G,
        nodes: List[str],
        emb_cache: Dict[str, Optional[np.ndarray]],
        cmap: Dict[str, int],
        seen_pairs: Set[Tuple[str, str]],
        limit: int = 50,
    ) -> List[ResearchCandidate]:
        """
        Discovers "Neural Neighbors" using LARQL feature projection from LLM weights.
        """
        if not self._larql:
            return []
            
        candidates: List[ResearchCandidate] = []
        # Sample a subset of nodes to query per scan to stay within budget
        sample_size = min(len(nodes), 20)
        query_nodes = random.sample(nodes, sample_size) if nodes else []

        for u in query_nodes:
            if len(candidates) >= limit:
                break
                
            links = self._larql.find_neural_neighbors(u, top_k=5)
            for link in links:
                v = link.target
                if v not in nodes: # We only consider known entities for now
                    continue
                if (u, v) in seen_pairs or (v, u) in seen_pairs:
                    continue
                if G.has_edge(u, v) or G.has_edge(v, u):
                    continue
                with self._lock:
                    if (u, v) in self._evaluated_pairs or (v, u) in self._evaluated_pairs:
                        continue

                # Neural links are high-value gap candidates
                cid_u = cmap.get(u, 0)
                cid_v = cmap.get(v, 0)
                community_dist = 0 if cid_u == cid_v else 1
                
                # Use link confidence as a proxy for gap score (1-similarity)
                # If neural confidence is high, it's a strong "missing link".
                potential, density = self._score_discovery_potential(
                    gap_score=1.0 - link.confidence,
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
                    gap_score=1.0 - link.confidence,
                    community_distance=community_dist,
                    seeded_by="larql_neural_scan",
                    local_density=density,
                ))
                seen_pairs.add((u, v))
                
        return candidates

    # ------------------------------------------------------------------
    # Internal — scoring and strategy
    # ------------------------------------------------------------------

    def _select_strategy(self, local_density: float) -> Dict[str, int]:
        """
        Choose beam search parameters based on local graph density.

        Dense  (> 0.4) — paths are short and abundant: shallow + narrow beam.
        Sparse (< 0.1) — paths are long and rare: deep + wide beam.
        Transitional   — use the agent's configured defaults.
        """
        if local_density > 0.4:
            return {"max_hop": 2, "beam_width": 5, "max_budget": 150}
        elif local_density < 0.1:
            return {"max_hop": 4, "beam_width": 12, "max_budget": 500}
        else:
            return {
                "max_hop": self.max_hop,
                "beam_width": self.beam_width,
                "max_budget": self.max_budget,
            }

    def _score_discovery_potential(
        self,
        gap_score: float,
        community_dist: int,
        G,
        u: str,
        v: str,
        cmap: Dict[str, int],
    ) -> Tuple[float, float]:
        """
        Compute (discovery_potential, local_density), both in [0, 1].

        Formula (base):
            0.4 × gap_score           (semantic gap — want high)
          + 0.4 × (1 − conn_density)  (sparse connection zone)
          + 0.2 × cross_community     (1.0 if different communities, 0.0 if same)

        Modifiers:
          - Real inter-community distance (from CSA metadata) replaces the
            binary 0/1 community_dist when available.  Farther = more novel.
          - Per-community-pair feedback weight (from report_outcome()) scales
            the final potential so the agent learns from human decisions.
        """
        # Real community distance from CSA distances dict (if available)
        cid_u = cmap.get(u, 0)
        cid_v = cmap.get(v, 0)
        comm_distances: Dict = getattr(self, "_comm_distances", {})
        if comm_distances and cid_u != cid_v:
            real_dist = comm_distances.get((cid_u, cid_v)) or comm_distances.get((cid_v, cid_u))
            if real_dist is not None:
                # Normalise: typical max distance ~10; cap at 1.0
                cross_community = min(1.0, float(real_dist) / 10.0)
            else:
                cross_community = 1.0 if community_dist > 0 else 0.0
        else:
            cross_community = 1.0 if community_dist > 0 else 0.0

        # Local connection density: fraction of possible edges in the 1-hop ego subgraph
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

        # Feedback learning: scale by community-pair weight
        pair_key = (min(cid_u, cid_v), max(cid_u, cid_v))
        feedback_w = self._feedback_weights.get(pair_key, 1.0)
        potential = min(1.0, max(0.0, potential * feedback_w))

        # Calibration: boost underrepresented communities (Phase 73)
        if self._calibrator is not None:
            cal_w = self._calibrator.get_weight(cid_u)
            potential = min(1.0, potential * cal_w)

        # Phase 98 Gap 7: goal-hinted entities get a 2× discovery boost
        goal_hints = self._goal_hints
        if goal_hints and (u in goal_hints or v in goal_hints):
            potential = min(1.0, potential * 2.0)

        return potential, conn_density

    def approve_larql_candidates(self, min_confidence: float = 0.8) -> int:
        """
        Scan candidates for "larql_neural_scan", validate them via ExternalValidator,
        and materialize those that are novel with high confidence.
        """
        candidates = [f for f in self.findings if f.candidate.seeded_by == "larql_neural_scan"]
        approved_count = 0
        
        for f in candidates:
            if f.candidate.discovery_potential < min_confidence:
                continue
                
            # Validate via ExternalValidator (Phase 52)
            if self._validator:
                from core.hypothesis_engine import HypothesisProposal
                proposal = HypothesisProposal(
                    hypothesis_id=str(uuid.uuid4()),
                    source=f.candidate.source_id,
                    target=f.candidate.target_id,
                    derived_relation="RELATED_TO", # Placeholder relation
                )
                report = self._validator.validate(proposal)
                if report.literature_status != "novel":
                    continue
            
            # Materialize
            approved_count += self.approve(f.finding_id)
            
        return approved_count
