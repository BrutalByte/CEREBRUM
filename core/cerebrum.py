"""
CerebrumGraph - unified THALAMUS -> CORTEX pipeline.

This is the main entry point for building and querying a knowledge graph
with CEREBRUM.  It replaces the manual wiring that previously lived inside
individual benchmark scripts.

Pipeline
--------
THALAMUS (ingestion):
  1. Load graph via adapter (from_kb / from_csv / from_triples / from_adapter)
  2. Optional: apply provable completion rules (InverseRule, CompositionRule)
  3. Encode entity embeddings (RandomEngine or SentenceEngine)
  4. Detect communities (DSCF) and optionally coarsen
  5. Build CSA attention engine with community distance matrix

CORTEX (reasoning):
  6. BeamTraversal over the CSA-weighted graph
  7. AnswerExtractor with path scoring and convergence voting

Usage
-----
    from core.cerebrum import CerebrumGraph
    from core.graph_completion import InverseRule, CompositionRule

    # Build from a pipe-separated triples file (MetaQA KB format)
    graph = CerebrumGraph.from_kb(
        "benchmarks/data/metaqa/kb.txt",
        sep="|",
        embeddings="sentence",
    )

    # Optional: augment with provable inference rules before building
    graph.complete([
        InverseRule("starred_actors"),          # symmetric
        InverseRule("directed_by", "director_of"),
    ])

    # Run THALAMUS pipeline (embeddings + communities + CSA)
    graph.build(cache_dir="cache/metaqa", min_community_size=20)

    # Query
    answers = graph.query(["Tom Hanks"], top_k=10, max_hop=2, min_hop=1)
    for a in answers:
        print(a.entity_id, a.score, a.path_confidence)
"""
from __future__ import annotations
from core.frontal_engine import FrontalEngine, ReasoningStrategy

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import networkx as nx
import numpy as np

from adapters.networkx_adapter import NetworkXAdapter
from adapters.mmap_adapter import MmapAdapter
from core.hardware import MemoryGovernor
from core.attention_engine import CSAEngine, HomeostaticModulator
from core.embedding_engine import EmbeddingEngine, RandomEngine
from core.graph_adapter import GraphAdapter
from core.chemical_modulator import ChemicalModulator   # Phase 68
from core.predictive_coder import PredictiveCodingEngine, PredictionResult  # Phase 69
from reasoning.answer_extractor import Answer, extract
from reasoning.traversal import BeamTraversal
from core.insight_validator import ProvenanceValidator
from core.telemetry import NeuralEvent, NeuralEventType

from typing import TYPE_CHECKING, Callable
if TYPE_CHECKING:
    from reasoning.trace import ReasoningTrace

logger = logging.getLogger("cerebrum.graph")


def _compute_beam_widths(mh: int, bw: int, factor: float) -> Dict[int, int]:
    """
    Funnel beam profile: linearly ramp beam width from bw*1.0 at hop 1
    to bw*factor at the penultimate hop. Terminal hop is never pruned
    (BeamTraversal skips pruning when hop == max_hop), so it is excluded.

    Examples (bw=10, factor=3.0):
      mh=1 -> {}
      mh=2 -> {1: 30}
      mh=3 -> {1: 10, 2: 30}
      mh=4 -> {1: 10, 2: 20, 3: 30}
    """
    if mh <= 1:
        return {}
    non_terminal = list(range(1, mh))
    n = len(non_terminal)
    widths: Dict[int, int] = {}
    for i, hop in enumerate(non_terminal):
        mult = factor if n == 1 else 1.0 + (factor - 1.0) * (i / (n - 1))
        widths[hop] = max(bw, int(bw * mult))
    return widths


# ---------------------------------------------------------------------------
# CerebrumGraph
# ---------------------------------------------------------------------------

class CerebrumGraph:
    """
    Encapsulates the full THALAMUS -> CORTEX pipeline for a single KG.

    Parameters
    ----------
    adapter          : NetworkXAdapter (or any GraphAdapter) wrapping the graph
    embedding_engine : EmbeddingEngine instance.  Defaults to RandomEngine(64).
    beam_width       : default beam width for traversal (default 10)
    max_hop          : maximum hop depth for traversal (default 3)
    max_neighbors    : per-node neighbor expansion cap (default 100)
    probabilistic    : enable Bayesian Beta-distribution path model (default False)
    warm_start_strength : warm-start strength for probabilistic mode (default 0)
    """

    def __init__(
        self,
        adapter:              GraphAdapter,
        embedding_engine:     Optional[EmbeddingEngine] = None,
        beam_width:           int   = 10,
        max_hop:              int   = 3,
        max_neighbors:        int   = 100,
        probabilistic:        bool  = False,
        warm_start_strength:  float = 0.0,
        beam_profile:         str   = "funnel",
        beam_profile_factor:  float = 3.0,
        expansion_k:          int   = 20,
        use_adaptive_expansion: bool = True,
        use_mmap:             bool = False,
        auto_hybrid:          bool = False,
        max_ram_gb:           Optional[float] = None,
        max_vram_gb:          Optional[float] = None,
    ):
        self.adapter             = adapter
        self.use_mmap            = use_mmap
        self.auto_hybrid         = auto_hybrid
        self.governor            = MemoryGovernor(max_ram_gb=max_ram_gb, max_vram_gb=max_vram_gb)
        self._embedding_engine   = embedding_engine or RandomEngine(dim=64)
        self._beam_width         = beam_width
        self._max_hop            = max_hop
        self._max_neighbors      = max_neighbors
        self._probabilistic      = probabilistic
        self._warm_start_strength = warm_start_strength
        self._beam_profile        = beam_profile
        self._beam_profile_factor = beam_profile_factor
        self._expansion_k         = expansion_k
        self._use_adaptive_expansion = use_adaptive_expansion
        self.homeostatic_modulator = HomeostaticModulator(target_activity=1.0) # Phase 143
        self._lateral_inhibition_ratio: float = 0.0  # Phase 100

        # Telemetry (Phase 63+)
        self._event_callbacks: List[Callable[[NeuralEvent], None]] = []

        # Neuro-Chemical Modulation (Phase 68)
        self.modulator = ChemicalModulator()
        self.frontal = FrontalEngine(modulator=self.modulator)
        self._research_agent = None

        # Global Workspace (Phase 110)
        from core.global_workspace import GlobalWorkspace
        self.global_workspace = GlobalWorkspace()

        # Prefrontal Bridge (Phase 117)
        from core.symbolic_engine import SymbolicValidator
        self.symbolic_validator = SymbolicValidator(self.adapter)

        self._csa:       Optional[CSAEngine]    = None
        self._traversal: Optional[BeamTraversal] = None
        self.predictive_coder: Optional[PredictiveCodingEngine] = None
        self._built      = False

        # ResearchAgent + AutonomousDiscoveryLoop (Phase 74+)
        self.research_agent: Optional[Any] = None
        self.autonomous_loop: Optional[Any] = None

        # Working Memory + Goal Stack (Phase 95)
        self._working_memory: Optional[Any] = None
        self._goal_stack: Optional[Any] = None

        # Persistence & REM Cycle (Phase 172)
        from core.persistence import QueryLog
        self.query_log = QueryLog()
        from core.consolidation_engine import ConsolidationEngine
        self.consolidation_engine = ConsolidationEngine(self.adapter, self, query_log=self.query_log)

    def set_research_agent(self, agent: Any) -> None:
        self.research_agent = agent
        self._research_agent = agent
        if hasattr(agent, "_adapter"):
            agent._adapter.graph = self

    def start_autonomous_loop(
        self,
        cycle_interval: float = 300.0,
        active_inference: bool = True,
        gui_adaptation: bool = False,
        gui_toolkit_url: str = "http://localhost:3000",
        autonomous_research: bool = True,
    ) -> Any:
        from core.autonomous_loop import AutonomousDiscoveryLoop, LoopConfig
        if not self.research_agent:
            raise ValueError("ResearchAgent must be set before starting loop.")
        config = LoopConfig(
            cycle_interval=cycle_interval,
            active_inference=active_inference,
            gui_adaptation=gui_adaptation,
            gui_toolkit_url=gui_toolkit_url,
            autonomous_research=autonomous_research,
            default_mode=True,
            working_memory=True,
        )
        self.autonomous_loop = AutonomousDiscoveryLoop(self.research_agent, config)
        self.autonomous_loop.start()
        return self.autonomous_loop

    def attach_gui_engine(self, engine: Any) -> None:
        """Attach a GUIAdaptationEngine directly (alternative to loop integration)."""
        self._gui_engine = engine
        logger.info("GUIAdaptationEngine attached.")

    def attach_sleep_cycle(self, orchestrator: Any) -> None:
        """Attach a SleepCycleOrchestrator (Phase 119)."""
        self._sleep_orchestrator = orchestrator
        logger.info("SleepCycleOrchestrator attached.")

    def attach_metacognitive_monitor(self, monitor: Any) -> None:
        """Attach a MetacognitiveMonitor (Phase 121) and auto-wire available sub-engines."""
        self._metacognitive_monitor = monitor
        # Wire sub-engines that are already on this graph into the monitor so EU
        # has real signal sources rather than neutral defaults (PE=0.5, etc.).
        if getattr(monitor, "pc", None) is None and self.predictive_coder is not None:
            monitor.pc = self.predictive_coder
        if getattr(monitor, "cm", None) is None:
            monitor.cm = self.modulator
        if getattr(monitor, "wm", None) is None and self._working_memory is not None:
            monitor.wm = self._working_memory
        logger.info(
            "MetacognitiveMonitor attached (auto-wired: pc=%s cm=%s wm=%s)",
            monitor.pc is not None,
            monitor.cm is not None,
            monitor.wm is not None,
        )

    def attach_epistemic_gate(self, gate: Any) -> None:
        """Attach an EpistemicGate (Phase 122) - evaluated by server after each query."""
        self._epistemic_gate = gate
        logger.info("EpistemicGate attached.")

    def register_confounders(self, confounder_nodes: list) -> None:
        """Phase 131: register known confounder nodes as NO_BACKDOOR constraints."""
        sv = getattr(self._traversal, "symbolic_validator", None) if self._traversal else None
        if sv is None:
            logger.warning("register_confounders: no SymbolicValidator attached to traversal.")
            return
        sv.register_confounders(confounder_nodes)
        logger.info("Phase 131: registered %d confounder nodes.", len(confounder_nodes))

    async def run_sleep_cycle(self, dry_run: bool = False) -> Any:
        """Trigger an offline consolidation sleep pass (Phase 119)."""
        orc = getattr(self, "_sleep_orchestrator", None)
        if orc is None:
            raise RuntimeError("SleepCycleOrchestrator not attached - call attach_sleep_cycle() first.")
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, orc.run, dry_run)

    def attach_engram(self, engram) -> None:
        """
        Wire an Engram (or SpeedTalkEngram) to enable predictive coding (Phase 69).

        Call this after ``build()`` and after the Engram has been warmed up.
        Safe to call multiple times - replaces the previous engine.
        """
        self.predictive_coder = PredictiveCodingEngine(engram, self.adapter)
        logger.info("PredictiveCodingEngine attached (%d cached patterns).", engram.size())

    def subscribe(self, callback: Callable[[NeuralEvent], None]):
        """Subscribe to real-time neural events."""
        if callback not in self._event_callbacks:
            self._event_callbacks.append(callback)

    def unsubscribe(self, callback: Callable[[NeuralEvent], None]):
        """Unsubscribe from neural events."""
        if callback in self._event_callbacks:
            self._event_callbacks.remove(callback)

    def emit(self, event: NeuralEvent):
        """Broadcast a neural event to all subscribers."""
        for cb in self._event_callbacks:
            try:
                cb(event)
            except Exception as e:
                logger.warning("Telemetry callback failed: %s", e)

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_kb(
        cls,
        path:             Union[str, Path],
        sep:              str  = "|",
        directed:         bool = True,
        embeddings:       str  = "random",
        embedding_dim:    int  = 64,
        pipeline=None,
        **kwargs,
    ) -> "CerebrumGraph":
        """
        Load from a flat triples file (one triple per line).

        Format:  head<sep>relation<sep>tail

        Parameters
        ----------
        path          : path to the KB file
        sep           : field separator (default "|" for MetaQA)
        directed      : use DiGraph if True (default), Graph if False
        embeddings    : "random" or "sentence"
        embedding_dim : dimension for random embeddings (ignored for sentence)
        pipeline      : optional IngestionPipeline for normalisation
        """
        G = nx.DiGraph() if directed else nx.Graph()
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                
                # Basic header detection
                if line.lower().startswith("source") or line.lower().startswith("head"):
                    continue

                parts = line.split(sep)
                if len(parts) < 3:
                    # Try comma/tab fallback if split fails
                    if sep == "|" and "," in line:
                        parts = line.split(",")
                    elif sep == "|" and "\t" in line:
                        parts = line.split("\t")
                    else:
                        continue
                
                if len(parts) < 3:
                    continue

                s = parts[0].strip()
                r = parts[1].strip()
                o = parts[2].strip()
                if not s or not r or not o:
                    continue

                if pipeline is not None:
                    edge = pipeline.process(s, o, r)
                    G.add_edge(
                        edge.source, edge.target,
                        relation=edge.relation,
                        confidence=float(edge.confidence),
                        provenance=str(edge.provenance),
                        weight=float(edge.weight),
                    )
                else:
                    G.add_edge(s, o, relation=r, weight=1.0, confidence=1.0,
                               provenance="kb_file")

        adapter = NetworkXAdapter(G)
        eng     = cls._make_engine(embeddings, embedding_dim)
        return cls(adapter, embedding_engine=eng, **kwargs)

    @classmethod
    def from_csv(
        cls,
        path:          Union[str, Path],
        embeddings:    str = "random",
        embedding_dim: int = 64,
        **kwargs,
    ) -> "CerebrumGraph":
        """
        Load from a CSV triples file (CEREBRUM CSV adapter format).
        """
        from adapters.csv_adapter import load_csv_adapter
        adapter = load_csv_adapter(str(path))
        eng     = cls._make_engine(embeddings, embedding_dim)
        return cls(adapter, embedding_engine=eng, **kwargs)

    @classmethod
    def from_triples(
        cls,
        triples:       List[tuple],
        directed:      bool = True,
        pipeline=None,
        embeddings:    str  = "random",
        embedding_dim: int  = 64,
        **kwargs,
    ) -> "CerebrumGraph":
        """
        Build from a list of (subject, predicate, object[, meta_dict]) tuples.
        """
        adapter = NetworkXAdapter.from_triples(triples, directed=directed,
                                               pipeline=pipeline)
        eng     = cls._make_engine(embeddings, embedding_dim)
        return cls(adapter, embedding_engine=eng, **kwargs)

    @classmethod
    def from_adapter(
        cls,
        adapter:       GraphAdapter,
        embeddings:    str = "random",
        embedding_dim: int = 64,
        **kwargs,
    ) -> "CerebrumGraph":
        """Wrap an existing adapter (e.g. Neo4jAdapter, FederatedAdapter)."""
        eng = cls._make_engine(embeddings, embedding_dim)
        return cls(adapter, embedding_engine=eng, **kwargs)

    @staticmethod
    def _make_engine(embeddings: str, dim: int) -> EmbeddingEngine:
        if embeddings == "sentence":
            from core.embedding_engine import SentenceEngine
            return SentenceEngine()
        elif embeddings == "random":
            return RandomEngine(dim=dim)
        else:
            raise ValueError(
                f"Unknown embeddings type {embeddings!r}. "
                "Use 'random' or 'sentence'."
            )

    # ------------------------------------------------------------------
    # Provable graph completion (pre-build step)
    # ------------------------------------------------------------------

    def complete(self, rules: List[Any]) -> "CerebrumGraph":
        """
        Apply provable inference rules to augment the graph before building.

        Every rule adds synthetic edges with full provenance metadata.
        No statistical predictions are made - only logical deductions.

        Parameters
        ----------
        rules : list of CompletionRule instances
                (InverseRule, CompositionRule, ...)

        Returns self for chaining.
        """
        total = 0
        for rule in rules:
            n = rule.apply(self.adapter)
            logger.info("%s: +%d edges", rule.describe(), n)
            total += n
        if total:
            logger.info("graph_completion: %d synthetic edges added total", total)
        # Invalidate built state - must rebuild after completing
        self._built = False
        return self

    # ------------------------------------------------------------------
    # Proactive enhancement (second layer)
    # ------------------------------------------------------------------

    def enhance(self, enhancers: List[Any]) -> "CerebrumGraph":
        """
        Apply proactive enhancement engines to the graph before building.

        Enhancements (like GraphBridgeEngine) use heuristics or embeddings to
        synthesize new links, unlike complete() which uses pure logic.

        Parameters
        ----------
        enhancers : list of enhancement engine instances
                    (GraphBridgeEngine, ...)

        Returns self for chaining.
        """
        total = 0
        for enhancer in enhancers:
            # We pass self._embedding_engine because many enhancers
            # (like GraphBridgeEngine) require it for similarity compute.
            n = enhancer.apply(self.adapter, self._embedding_engine)
            logger.info("%s: +%d edges", enhancer.describe(), n)
            total += n
        if total:
            logger.info("graph_enhancement: %d synthetic edges added total", total)
        # Invalidate built state
        self._built = False
        return self

    # ------------------------------------------------------------------
    # THALAMUS build pipeline
    # ------------------------------------------------------------------

    def _spill_to_mmap(self):
        """Phase 169: Convert current NetworkXAdapter to MmapAdapter to save RAM."""
        from adapters.mmap_adapter import MmapAdapter
        import tempfile

        if isinstance(self.adapter, MmapAdapter):
            return

        logger.warning("Memory pressure detected. Spilling graph topology to MmapAdapter...")
        
        # 1. Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix="cerebrum_mmap_")
        
        # 2. Serialize current adapter to binary
        MmapAdapter.from_adapter(self.adapter, temp_dir)
        
        # 3. Swap adapter and preserve metadata
        old_adapter = self.adapter
        self.adapter = MmapAdapter(temp_dir, governor=self.governor)
        
        # Preserve community_map and embeddings if they were already computed
        if hasattr(old_adapter, "community_map"):
            self.adapter.community_map = old_adapter.community_map
        if hasattr(old_adapter, "embeddings"):
            self.adapter.embeddings = old_adapter.embeddings
            
        logger.info("Spill complete. Disk-backed reasoning active at: %s", temp_dir)

    def build(
        self,
        cache_dir:            Optional[Union[str, Path]] = None,
        min_community_size:   int   = 0,
        coarsen_target:       Optional[int] = None,
        n_trials:             int   = 1,
        seed:                 int   = 42,
        force_rebuild:        bool  = False,
        community_engine:     str   = "dscf",
        resolution:           float = 1.0,
        callback:             Optional[callable] = None,
        use_graphsage:        bool  = False,
        graphsage_self_weight: float = 0.5,
        graphsage_neighbor_weight: float = 0.5,
        # Phase 135: KGE topology-aware embedding enrichment
        use_kge:     bool  = False,
        kge_model:   str   = "transe",   # "transe" | "rotate"
        kge_epochs:  int   = 100,
        kge_dim:     int   = 64,
        kge_blend:   float = 0.5,
        kge_device:  Optional[str] = None,  # None -> auto (picks RTX 5090)
    ) -> "CerebrumGraph":
        """
        Run the THALAMUS pipeline: embeddings -> DSCF communities -> CSA.

        Parameters
        ----------
        cache_dir           : directory for caching embeddings and communities.
                              Pass None to skip caching.
        min_community_size  : merge communities smaller than this (0 = disabled).
        coarsen_target      : alternatively, coarsen to at most this many
                              communities (e.g. 300 for GrailQA).
        n_trials            : DSCF runs to take the best of (default 1).
        seed                : random seed for reproducibility.
        force_rebuild       : ignore cache and recompute everything.
        community_engine    : 'dscf', 'leiden', 'lpa', or 'tsc' (default 'dscf').
                              'tsc' runs vectorized Triple-Signal Consensus with auto-PageRank.
                              'leiden' is significantly faster for large graphs (>1M nodes).
        callback            : optional function(progress_float, status_str) for real-time updates.

        Returns self for chaining.
        """
        # Phase 169: check memory governor before starting expensive build
        if self.governor.is_spill_needed():
            self._spill_to_mmap()

        from core.community_engine import dscf_communities, leiden_communities, lpa_communities
        from core.structural_encoder import (
            build_community_distance_matrix,
            build_community_graph,
            adjacent_community_pairs,
            coarsen_communities,
            compute_structural_features,
            encode_structural_features,
        )
        import time

        cache = Path(cache_dir) if cache_dir else None
        if cache:
            cache.mkdir(parents=True, exist_ok=True)

        G     = self.adapter.to_networkx()
        nodes = list(G.nodes())

        if not nodes:
            raise ValueError("Graph has no nodes - nothing to build.")

        # ----------------------------------------------------------
        # 1. Embeddings
        # ----------------------------------------------------------
        if callback: callback(0.1, "Step 1/5: Loading/Encoding Embeddings...")
        # Phase 152: embedding cache key includes engine type so random and
        # sentence-transformer caches never collide.
        _emb_tag = type(self._embedding_engine).__name__.lower().replace("embeddingengine", "") or "base"
        emb_cache = cache / f"embeddings_{_emb_tag}.pkl" if cache else None

        # Track whether embeddings came from cache so later steps know
        # whether structural enrichment and GraphSAGE need to be applied.
        _sage_cache = cache / "embeddings_sage.pkl" if (cache and use_graphsage) else None
        _raw_cache  = emb_cache   # alias for clarity

        _loaded_from_raw_cache  = bool(not force_rebuild and _raw_cache  and _raw_cache.exists())
        _loaded_from_sage_cache = bool(not force_rebuild and _sage_cache and _sage_cache.exists())

        if _loaded_from_sage_cache:
            # Fast path: load fully-smoothed embeddings - skip encoding AND SAGE
            logger.info("Loading cached GraphSAGE embeddings from %s", _sage_cache)
            with open(_sage_cache, "rb") as f:
                self.adapter.embeddings = pickle.load(f)
        elif _loaded_from_raw_cache:
            logger.info("Loading cached embeddings from %s", _raw_cache)
            with open(_raw_cache, "rb") as f:
                self.adapter.embeddings = pickle.load(f)
        else:
            # Build label map: node_id -> human-readable label
            entity_labels: Dict[str, str] = {}
            for n in nodes:
                data  = G.nodes[n]
                label = data.get("label") or data.get("friendly_name") or str(n)
                entity_labels[str(n)] = label

            logger.info(
                "Encoding %d entities with %s",
                len(entity_labels),
                type(self._embedding_engine).__name__,
            )
            self.adapter.embeddings = self._embedding_engine.encode_entities(
                entity_labels
            )

            if _raw_cache:
                with open(_raw_cache, "wb") as f:
                    pickle.dump(self.adapter.embeddings, f)

        # ----------------------------------------------------------
        # 1.3. KGE topology-aware embedding enrichment (Phase 135, optional)
        # Train TransE/RotatE on graph triples, then blend with base embeddings.
        # Runs BEFORE GraphSAGE so smoothing propagates enriched vectors.
        # Skipped when loaded from sage_cache (embeddings already finalized).
        # ----------------------------------------------------------
        if use_kge and self.adapter.embeddings and not _loaded_from_sage_cache:
            if callback: callback(0.15, "Step 1.3/5: KGE Embedding Enrichment...")
            _kge_cache = (
                cache / f"embeddings_kge_{kge_model}_{kge_dim}_{kge_epochs}.pkl"
                if cache else None
            )
            _kge_embs = None
            if _kge_cache and _kge_cache.exists() and not force_rebuild:
                logger.info("Loading cached KGE embeddings from %s", _kge_cache)
                with open(_kge_cache, "rb") as _f:
                    _kge_embs = pickle.load(_f)
            else:
                from core.kge_engine import TransEEngine, RotatEEngine
                _KGECls = TransEEngine if kge_model.lower() == "transe" else RotatEEngine
                _kge_eng = _KGECls(dim=kge_dim, seed=seed)
                _kge_result = _kge_eng.fit(self.adapter, n_epochs=kge_epochs, device=kge_device)
                logger.info(
                    "KGE %s trained: loss=%.4f in %.1fs",
                    kge_model, _kge_result.final_loss, _kge_result.duration_seconds,
                )
                _kge_embs = {
                    eid: _kge_eng.get_embedding(eid)
                    for eid in self.adapter.embeddings
                }
                # Remove entities the KGE engine couldn't embed (None values)
                _kge_embs = {k: v for k, v in _kge_embs.items() if v is not None}
                if _kge_cache:
                    with open(_kge_cache, "wb") as _f:
                        pickle.dump(_kge_embs, _f)

            if _kge_embs:
                from core.kge_engine import blend_kge_embeddings
                self.adapter.embeddings = blend_kge_embeddings(
                    base=self.adapter.embeddings, kge=_kge_embs, blend=kge_blend,
                )
                logger.info("KGE blend applied (model=%s, blend=%.2f)", kge_model, kge_blend)

        # ----------------------------------------------------------
        # 1.5. GraphSAGE neighborhood smoothing (optional)
        # Guarded by a separate cache key so smoothing is applied exactly
        # once - loading from sage_cache above already skips this block.
        # ----------------------------------------------------------
        if use_graphsage and self.adapter.embeddings and not _loaded_from_sage_cache:
            if callback: callback(0.2, "Step 1.5/5: GraphSAGE Neighborhood Smoothing...")
            from core.embedding_engine import smooth_with_graphsage
            logger.info(
                "Applying GraphSAGE smoothing (self=%.2f, neighbor=%.2f)",
                graphsage_self_weight, graphsage_neighbor_weight,
            )
            self.adapter.embeddings = smooth_with_graphsage(
                self.adapter.embeddings,
                G,
                self_weight=graphsage_self_weight,
                neighbor_weight=graphsage_neighbor_weight,
                num_layers=2,
            )
            if _sage_cache:
                # Write smoothed result to its own cache file, never to the
                # raw embeddings.pkl (which must stay un-smoothed for reuse).
                _sage_cache.parent.mkdir(parents=True, exist_ok=True)
                with open(_sage_cache, "wb") as f:
                    pickle.dump(self.adapter.embeddings, f)

        # ----------------------------------------------------------
        # 2. Structural & Temporal Enrichment (Phase 33)
        # ----------------------------------------------------------
        if callback: callback(0.3, "Step 2/5: Computing Structural Features...")
        # Compute raw graph features (PageRank, Betweenness, Recency)
        logger.info("Computing structural features (Phase 33)...")
        struct_features = compute_structural_features(G, current_time=time.time())

        # Apply structural enrichment only when embeddings were freshly encoded.
        # When loaded from either cache the enrichment is already baked in.
        _freshly_encoded = not _loaded_from_raw_cache and not _loaded_from_sage_cache
        if force_rebuild or _freshly_encoded:
            # Encode features into a vector of the same dimension as existing embeddings
            # We use a residual connection: h = LayerNorm(h + structural_encoding)
            if self.adapter.embeddings:
                sample_emb = next(iter(self.adapter.embeddings.values()))
                emb_dim = len(sample_emb)
                
                logger.info("Encoding structural features (dim=%d)...", emb_dim)
                encoded_struct = encode_structural_features(struct_features, dim=emb_dim, seed=seed)
                
                # Residual addition + normalization
                for node, s_vec in encoded_struct.items():
                    if node in self.adapter.embeddings:
                        h = self.adapter.embeddings[node] + s_vec
                        norm = np.linalg.norm(h)
                        if norm > 0:
                            self.adapter.embeddings[node] = h / norm

                # Write enriched embeddings back to the raw cache so future
                # loads start from the post-structural state.
                if _raw_cache and not _loaded_from_raw_cache:
                    with open(_raw_cache, "wb") as f:
                        pickle.dump(self.adapter.embeddings, f)

        # ----------------------------------------------------------
        # 3. Community detection
        # ----------------------------------------------------------
        if callback: callback(0.5, "Step 3/5: DSCF Community Detection (Attention Heads)...")
        comm_cache = cache / f"communities_{community_engine}.pkl" if cache else None
        
        # Use G_und consistently throughout this block
        G_und = G.to_undirected() if G.is_directed() else G

        if not force_rebuild and comm_cache and comm_cache.exists():
            logger.info("Loading cached communities from %s", comm_cache)
            with open(comm_cache, "rb") as f:
                parts = pickle.load(f)
        else:
            logger.info(
                "Running %s on %d nodes, %d edges...",
                community_engine, G_und.number_of_nodes(), G_und.number_of_edges(),
            )
            if community_engine == "dscf":
                if n_trials > 1:
                    from core.community_engine import best_of_n_dscf
                    parts = best_of_n_dscf(G_und, n_trials=n_trials, seed=seed)
                else:
                    parts = dscf_communities(G_und)
            elif community_engine == "leiden":
                parts = leiden_communities(G_und)
            elif community_engine == "lpa":
                parts = lpa_communities(G_und)
            elif community_engine == "tsc":
                from core.community_engine import tsc_communities
                pr_weights = {n: d.get("pagerank", 1.0) for n, d in struct_features.items()}
                parts = tsc_communities(G_und, resolution=resolution, centrality_weights=pr_weights)
            else:
                raise ValueError(f"Unknown community_engine {community_engine!r}")

            if comm_cache:
                with open(comm_cache, "wb") as f:
                    pickle.dump(parts, f)

        self.adapter._partition = parts
        cm: Dict = {n: cid for cid, members in enumerate(parts) for n in members}
        n_raw = len(set(cm.values()))
        logger.info("%d raw DSCF communities", n_raw)

        # ----------------------------------------------------------
        # 3. Optional coarsening
        # ----------------------------------------------------------
        # Automatic coarsen if over 2000 (structural matrix cap)
        if n_raw > 2000 and coarsen_target is None and min_community_size == 0:
            coarsen_target = 2000
            logger.warning("Community count %d exceeds 2000. Auto-coarsening to 2000.", n_raw)

        if min_community_size > 0:
            from core.structural_encoder import coarsen_communities as _size_coarsen
            # merge_small_communities is in metaqa_eval; use structural_encoder's version
            cm = _size_coarsen(G_und, cm, min_size=min_community_size)
            logger.info(
                "%d communities after min_size=%d coarsening",
                len(set(cm.values())), min_community_size,
            )
        elif coarsen_target is not None:
            cm = coarsen_communities(G_und, cm, target_max=coarsen_target)
            logger.info(
                "%d communities after target_max=%d coarsening",
                len(set(cm.values())), coarsen_target,
            )

        self.adapter.community_map = cm

        # ----------------------------------------------------------
        # 4. CSA engine
        # ----------------------------------------------------------
        if callback: callback(0.8, "Step 4/5: Building CSA Attention Engine...")
        logger.info("Building CSA engine...")
        distances = build_community_distance_matrix(G_und, cm)
        adj       = adjacent_community_pairs(G_und, cm)
        cg        = build_community_graph(G_und, cm)

        self._csa = CSAEngine(self.adapter)
        self._csa.set_community_graph(distances, adj, community_graph=cg)

        # Phase 33: Pass PageRank from structural features to CSA
        pr_map = {node: data.get("pagerank", 0.0) for node, data in struct_features.items()}
        self._csa.set_pagerank(pr_map)

        # ----------------------------------------------------------
        # 5. StructuralRelationInferrer (Phase 172)
        # ----------------------------------------------------------
        if callback: callback(0.88, "Step 5/6: Building Structural Relation Inferrer...")
        from core.structural_relation_inferrer import StructuralRelationInferrer
        self._sri = StructuralRelationInferrer()
        self._sri.build(self.adapter)
        self._sri.build_community_fingerprints(self.adapter)  # Phase 172
        # Phase 172: build semantic TRB index when a non-random embedding engine is available
        _is_random_engine = type(self._embedding_engine).__name__.lower() == "randomengine"
        if not _is_random_engine:
            self._sri.build_semantic_index(self.adapter, self._embedding_engine)
        logger.info(
            "SRI built: %d relation types, %d communities fingerprinted, semantic_index=%s",
            len(self._sri._rel_freq),
            len(self._sri._community_dominant_rel),
            self._sri._semantic_index_built,
        )

        # Phase 172: Terminal-Anchor Source Index — O(E) pass
        # _anchor_sources[rel] = set of entity IDs with at least one outgoing rel edge.
        # Used at query time to bias the penultimate-hop beam toward entities that can
        # directly produce answers of the correct type.
        self._anchor_sources: Dict[str, Any] = {}
        try:
            _G = self.adapter.to_networkx()
            for _u, _v, _edata in _G.edges(data=True):
                _rel = _edata.get("relation", "")
                if _rel:
                    if _rel not in self._anchor_sources:
                        self._anchor_sources[_rel] = set()
                    self._anchor_sources[_rel].add(_u)
            logger.info("Phase 172: anchor source index built (%d relation types)", len(self._anchor_sources))
        except Exception as _exc:
            logger.warning("Phase 172: anchor source index build failed: %s", _exc)

        # Phase 172: GraphProfiler — auto query strategy selection
        try:
            from core.graph_profiler import GraphProfiler
            self._query_profile = GraphProfiler.profile(self.adapter, self._anchor_sources)
            logger.info(
                "Phase 172: graph profile computed — regime=%s "
                "hub_score=%.3f mean_rel_coverage=%.3f "
                "recommended: hop_expand=%s trb_auto=%s anchor_bonus=%s",
                self._query_profile.regime,
                self._query_profile.hub_score,
                self._query_profile.mean_rel_coverage,
                self._query_profile.recommended_hop_expand,
                self._query_profile.recommended_trb_auto,
                self._query_profile.recommended_anchor_bonus,
            )
        except Exception as _exc:
            logger.warning("Phase 172: graph profiling failed: %s", _exc)
            self._query_profile = None

        # ----------------------------------------------------------
        # 6. BeamTraversal
        # ----------------------------------------------------------
        self._traversal = BeamTraversal(
            adapter             = self.adapter,
            csa_engine          = self._csa,
            beam_width          = self._beam_width,
            max_hop             = self._max_hop,
            max_neighbors       = self._max_neighbors,
            probabilistic       = self._probabilistic,
            warm_start_strength = self._warm_start_strength,
        )
        self._traversal.global_workspace = self.global_workspace
        self._traversal.predictive_coder = self.predictive_coder
        self._traversal.lateral_inhibition_ratio = self._lateral_inhibition_ratio

        # Phase 124: Causal edge index - O(1) lookup during beam scoring.
        # Build once at graph load; no CausalEngine invocation required.
        from core.causal_engine import CAUSAL_RELATIONS as _CAUSAL_RELS
        _causal_idx: set = set()
        for u, v, data in G.edges(data=True):
            rel = data.get("relation", data.get("relation_type", ""))
            if rel in _CAUSAL_RELS:
                _causal_idx.add((u, v))
        self._traversal._causal_edge_index = _causal_idx
        logger.info("Phase 124: causal edge index built (%d causal edges).", len(_causal_idx))

        self._built = True
        n_comm = len(set(cm.values()))
        logger.info(
            "Build complete: %d nodes, %d edges, %d communities",
            G.number_of_nodes(), G.number_of_edges(), n_comm,
        )
        return self

    # ------------------------------------------------------------------
    # Phase 172: Query profile access
    # ------------------------------------------------------------------

    @property
    def query_profile(self):
        """Return the QueryProfile computed during build(), or None if unavailable."""
        return getattr(self, "_query_profile", None)

    # ------------------------------------------------------------------
    # CORTEX query
    # ------------------------------------------------------------------

    def query(
        self,
        seeds:             List[str],
        top_k:             int            = 10,
        min_hop:           int            = 1,
        max_hop:           Optional[int]  = None,
        beam_width:        Optional[int]  = None,
        query_embedding:   Optional[np.ndarray] = None,
        relation_prior=None,
        vote_weight:       float          = 0.45,
        memory_threshold_pct: float       = 95.0,
        trace_info:        Optional["ReasoningTrace"] = None,
        max_loops:         int            = 1,
        context_seeds:     Optional[List[str]] = None,
        beam_profile:      Optional[str]   = None,
        beam_profile_factor: Optional[float] = None,
        hop_expand:        Optional[bool]  = None,
        expansion_k:       Optional[int]   = None,
        branch_bonus_weight: float         = 0.0,
        residual_k:        int             = 10,
        min_diversity_target: int          = 15,
        terminal_relation_boost:     Optional[Dict[str, float]] = None,
        penultimate_relation_boost:  Optional[Dict[str, float]] = None,
        beam_widths:                 Optional[Dict[int, int]] = None,
        degree_penalty_weight:       float = 0.0,
        penultimate_decay:           float = 0.0,
        auto_infer_terminal_relation: Optional[bool] = None,
        anchor_bonus:                Optional[float] = None,
    ) -> List[Answer]:
        """
        Traverse the graph from ``seeds`` and return ranked answers.

        Parameters
        ----------
        seeds           : list of entity IDs to start from
        top_k           : number of answers to return
        min_hop         : minimum hop depth (1 = include 1-hop; 2 = exclude seeds)
        max_hop         : maximum traversal depth. Defaults to the value set at
                          build time. Override per-query to constrain depth (e.g.,
                          max_hop=1 for 1-hop eval avoids flooding with deep paths).
        beam_width      : beam width for this query (default build-time value).
        query_embedding : optional query vector for semantic alignment
        relation_prior  : optional RelationPathPrior or GraphRelationPrior
        vote_weight     : convergence voting weight (default 0.30)
        branch_bonus_weight : Phase 144 DBC - multiplicative bonus for answers
                          reached via multiple distinct hop-1 branches (default 0.25)
        memory_threshold_pct : safety threshold for resource usage (default 95.0)
        trace_info      : optional ReasoningTrace to populate (Phase 62).
        max_loops       : LoopLM-style iterative refinement depth (Phase 70,
                          arXiv:2510.25741). 1 = single-pass (default). >1
                          applies BeamTraversal T times, using top answer
                          entities as additional seeds each loop. Early exit
                          when PE converges or answers stabilise.

        Returns
        -------
        List[Answer] sorted by score descending.
        """
        if not self._built:
            raise RuntimeError(
                "Graph has not been built. Call graph.build() first."
            )

        # Phase 172: Resolve None params against QueryProfile defaults.
        # Explicit True/False always wins; None = use profile recommendation.
        _qp = getattr(self, "_query_profile", None)
        if hop_expand is None:
            hop_expand = _qp.recommended_hop_expand if _qp else False
        if auto_infer_terminal_relation is None:
            auto_infer_terminal_relation = _qp.recommended_trb_auto if _qp else False
        if anchor_bonus is None and _qp is not None:
            anchor_bonus = _qp.recommended_anchor_bonus  # may still be None

        # Phase 119: notify sleep orchestrator of query activity
        orc = getattr(self, "_sleep_orchestrator", None)
        if orc is not None:
            orc.notify_activity()

        # Phase 95: merge goal-directed context seeds into query seeds
        original_seeds = list(seeds)
        if context_seeds:
            seeds = list(dict.fromkeys(list(seeds) + list(context_seeds)))[:max(len(seeds), 5)]

        # Handle per-query overrides by creating a temporary traversal if needed
        bw = beam_width or self._beam_width
        mh = max_hop or self._max_hop

        # Phase 125: Epistemic-adaptive beam width using previous query's EU.
        # On first query _last_eu=0.5 (neutral) so the formula returns bw unchanged.
        _gate = getattr(self, "_epistemic_gate", None)
        if _gate is not None:
            _prev_eu = getattr(_gate, "_last_eu", 0.5)
            bw = min(bw * 3, max(bw, int(bw * (1.0 + _prev_eu))))

        # Phase 68: Neuro-chemical modulation of traversal configuration
        mod_cfg = self.modulator.modulate_traversal({
            "beam_width": bw,
            "max_hop": mh
        })
        bw = mod_cfg["beam_width"]
        mh = mod_cfg["max_hop"]

        # Phase 68: Modulate CSA parameters (alpha, beta, gamma)
        csa_overrides = {}
        if self._csa:
            base_p = {"alpha": self._csa.alpha, "beta": self._csa.beta, "gamma": self._csa.gamma}
            csa_overrides = self.modulator.modulate_params(base_p)

        # Phase 136: Funnel beam profile - widen intermediate hops to prevent
        # catastrophic path loss on multi-hop queries.
        _profile        = beam_profile        or self._beam_profile
        _profile_factor = beam_profile_factor or self._beam_profile_factor
        _auto_beam_widths: Dict[int, int] = beam_widths if beam_widths is not None else (
            _compute_beam_widths(mh, bw, _profile_factor)
            if _profile == "funnel" else {}
        )

        needs_custom = (mh != self._max_hop or bw != self._beam_width or memory_threshold_pct != 95.0 or bool(csa_overrides) or hop_expand or (beam_widths is not None))
        _prev_widths: Dict[int, int] = {}  # only meaningful when not needs_custom

        # Phase 172/163: Structural / semantic auto-inference of terminal relation boost.
        # Phase 172: when query_embedding is available and a semantic index was built,
        # use query-relation cosine similarity (STRB) — more accurate than structural SRI.
        # Phase 172: structural hard-select fallback when no embedding is available.
        if auto_infer_terminal_relation and not terminal_relation_boost:
            _sri = getattr(self, "_sri", None)
            if _sri is not None:
                if query_embedding is not None and _sri._semantic_index_built:
                    # Phase 172: semantic TRB via embedding similarity
                    terminal_relation_boost = _sri.semantic_trb(
                        query_embedding=query_embedding,
                        boost_factor=5.0,
                        soft_mode=True,
                    )
                elif seeds:
                    # Phase 172: structural hard-select fallback
                    terminal_relation_boost = _sri.to_boost_dict(
                        seed_id=seeds[0],
                        n_hops=max_hop or self._max_hop,
                        adapter=self.adapter,
                        hard_select=True,
                        hard_select_factor=5.0,
                        min_confidence_ratio=3.0,
                    )

        _trb = terminal_relation_boost or {}
        _prb = penultimate_relation_boost or {}

        # Phase 172: Terminal-Anchor hints — bias penultimate-hop beam toward entities
        # that are sources of the terminal relation (can directly produce answer-type entities).
        # anchor_hints uses sub-traversal hop indices: for 3-hop, hop-1 of sub-traversal
        # (= hop-2 of full query) is the critical penultimate selection point.
        _anchor_hints: Optional[Dict[int, Any]] = None
        _stage1_anchor: Optional[Any] = None
        if _trb and mh >= 3:
            _best_rel = max(_trb, key=_trb.get)
            _anchor_set = getattr(self, "_anchor_sources", {}).get(_best_rel)
            if _anchor_set:
                _sub_penultimate = mh - 2  # e.g. 3-hop → 1, 4-hop → 2
                _anchor_bonus = anchor_bonus if anchor_bonus is not None else 2.0
                _anchor_hints = {_sub_penultimate: (_anchor_set, _anchor_bonus)}
                # Stage-1 anchor: prefer hop-1 entities that have outgoing edges to R3-source type
                # (entities in anchor_set are sources of R3; their reverse-neighbors are good hop-1 entities)
                # For simplicity use the anchor_set itself for stage-1 when seeds are OF that type
                # (conservative: only activate when hop-1 entities are likely in anchor_set)
                _stage1_anchor = None  # Disabled: too broad for most graphs, enable per-domain
        if needs_custom:
            from core.resource_governor import ResourceGovernor
            # Use the existing governor if available, else initialize
            gov = getattr(self, "governor", ResourceGovernor(memory_threshold_pct=memory_threshold_pct))
            traversal = BeamTraversal(
                adapter             = self.adapter,
                csa_engine          = self._csa,
                beam_width          = bw,
                max_hop             = mh,
                max_neighbors       = self._max_neighbors,
                probabilistic       = self._probabilistic,
                warm_start_strength = self._warm_start_strength,
                governor            = gov,
                beam_widths         = _auto_beam_widths,  # Phase 136
                terminal_relation_boost    = _trb,
                penultimate_relation_boost = _prb,  # Phase 156
                penultimate_decay          = penultimate_decay,
                **csa_overrides # Inject hormonal overrides
            )
            traversal.global_workspace = self.global_workspace
            traversal.predictive_coder = self.predictive_coder
            # Copy Phase 99-101 state from the default traversal
            traversal.lateral_inhibition_ratio = self._lateral_inhibition_ratio
            _ve = getattr(self._traversal, "_valence_engine", None)
            if _ve is not None:
                traversal._valence_engine = _ve
        else:
            traversal = self._traversal
            _prev_widths = traversal._beam_widths
            traversal._beam_widths = _auto_beam_widths  # Phase 136: temporary override

        # Phase 137: H1SE - replace traversal with HopExpandedTraversal when
        # hop_expand=True. needs_custom=True guarantees no shared-traversal
        # restore is needed in the finally block.
        if hop_expand and mh >= 2:
            from reasoning.expanded_traversal import HopExpandedTraversal
            _ek = expansion_k if expansion_k is not None else self._expansion_k
            _uae = self._use_adaptive_expansion
            traversal = HopExpandedTraversal(
                adapter               = self.adapter,
                csa_engine            = self._csa,
                beam_width            = bw,
                max_hop               = mh,
                max_neighbors         = self._max_neighbors,
                expansion_k           = _ek,
                beam_profile_factor   = _profile_factor,
                max_budget            = getattr(self._traversal, "max_budget", 10_000),
                governor              = getattr(self._traversal, "governor", None),
                probabilistic         = self._probabilistic,
                warm_start_strength   = self._warm_start_strength,
                modulator             = self.modulator,
                use_adaptive_expansion = _uae,
                min_diversity_target  = min_diversity_target,
                residual_k            = residual_k,
                terminal_relation_boost    = _trb,
                penultimate_relation_boost = _prb,  # Phase 156
                beam_widths                = _auto_beam_widths,
                penultimate_decay          = penultimate_decay,
                anchor_hints               = _anchor_hints,       # Phase 172
                stage1_anchor_hint         = _stage1_anchor,      # Phase 172
                **csa_overrides,
            )
            traversal._causal_edge_index = getattr(
                self._traversal, "_causal_edge_index", set()
            )

        # Phase 69: Generate predictive prior before traversal
        pred_prior = None
        if self.predictive_coder is not None:
            try:
                pred_prior = self.predictive_coder.predict(seeds)
            except Exception as exc:
                logger.warning("PredictiveCodingEngine.predict() failed: %s", exc)

        # Phase 99: Thalamic Gating - build WM priming map
        _priming_map = self._build_priming_map()

        # Phase 70: Looped Reasoning (Recursive refinement)
        looped = None
        if max_loops > 1:
            from reasoning.looped_traversal import LoopedBeamTraversal
            looped = LoopedBeamTraversal(
                traversal=traversal,
                predictive_coder=self.predictive_coder,
                max_loops=max_loops
            )

        try:
            if max_loops > 1 and looped is not None:
                paths = looped.traverse(seeds, query_embedding=query_embedding, trace_info=trace_info, node_priming=_priming_map if _priming_map else None)[0]
            else:
                paths = traversal.traverse(seeds, query_embedding=query_embedding, trace_info=trace_info, node_priming=_priming_map if _priming_map else None)
            

            # Phase 150: Frontal Engine executive strategy
            entropy = getattr(traversal, "_last_entropy", 0.0)
            strategy = self.frontal.determine_strategy(paths, entropy=entropy)
            logger.debug("FrontalEngine: strategy=%s agent_attached=%s gap_count=%d", strategy, self._research_agent is not None, len(getattr(traversal, "epistemic_gaps", [])))


            if strategy == ReasoningStrategy.DEEP and self._research_agent and getattr(traversal, "epistemic_gaps", []):
                logger.info("FrontalEngine: Suspending for DEEP research...")
                for gap in traversal.epistemic_gaps[:2]:
                    # Step 4: Targeted ResearchAgent coupling
                    if hasattr(self._research_agent, "push_candidate"):
                        from core.research_agent import ResearchCandidate
                        self._research_agent.push_candidate(ResearchCandidate(
                            source_id=gap["source"],
                            target_id=gap["target"],
                            discovery_potential=0.9,
                            gap_score=1.0 - gap["score"],
                            community_distance=1,
                            seeded_by="frontal_engine"
                        ))
                
                # Trigger immediate scan
                self._research_agent.scan_once()
                # Retry with new knowledge
                logger.info("FrontalEngine: Resuming query after research.")
                paths = traversal.traverse(seeds, query_embedding=query_embedding, node_priming=_priming_map)

            elif strategy == ReasoningStrategy.HYBRID and self._research_agent and getattr(traversal, "epistemic_gaps", []):
                logger.info("FrontalEngine: Dispatching HYBRID async research.")
                for gap in traversal.epistemic_gaps[:2]:
                    if hasattr(self._research_agent, "push_candidate"):
                        from core.research_agent import ResearchCandidate
                        self._research_agent.push_candidate(ResearchCandidate(
                            source_id=gap["source"],
                            target_id=gap["target"],
                            discovery_potential=0.7,
                            gap_score=1.0 - gap["score"],
                            community_distance=1,
                            seeded_by="frontal_engine"
                        ))
                # Agent will pick these up in its next background cycle

            # Phase 149: Cingulate Engine recursive refinement (Legacy fallback)
            if strategy == ReasoningStrategy.FAST and ProvenanceValidator.is_hub_flooded(paths):
                logger.info('CingulateEngine: Retrying with strict constraints...')
                traversal.beam_width = max(2, traversal.beam_width // 2)
                paths = traversal.traverse(seeds, query_embedding=query_embedding, node_priming=_priming_map)


        finally:
            # Phase 68: Natural decay of hormonal state
            self.modulator.step()
            # Phase 136: restore shared traversal's beam widths
            if not needs_custom:
                self._traversal._beam_widths = _prev_widths

        # Phase 69: Compute prediction error and dispatch to regulators
        if self.predictive_coder is not None:
            try:
                pred_result = self.predictive_coder.update(pred_prior, paths)
                pe  = pred_result.prediction_error
                sol = pred_result.soliton_stability

                # Drive ChemicalModulator from PE signal
                self.modulator.update_arousal(pe)
                self.modulator.update_novelty(pe)
                self.modulator.update_reinforcement(pred_result.reinforcement_signal)

                # Attach to ERT trace (Phase 62 integration)
                if trace_info is not None:
                    if pred_prior is not None:
                        trace_info.prior = {
                            "predicted_relations": pred_prior.predicted_relations,
                            "predicted_nodes":     pred_prior.predicted_nodes,
                            "confidence":          pred_prior.confidence,
                        }
                    trace_info.prediction_error = pe
                    trace_info.soliton_index    = sol

                logger.debug(
                    "Predictive coding: PE=%.3f soliton=%.3f reinforcement=%.3f",
                    pe, sol, pred_result.reinforcement_signal,
                )
            except Exception as exc:
                logger.warning("PredictiveCodingEngine.update() failed: %s", exc)

        # Phase 94: Emit METABOLIC_FLUX so UE5 HUD bars stay live
        try:
            lr_scale = getattr(self.modulator, "learning_rate_scale", 1.0)
            self.emit(NeuralEvent.flux(state=self.modulator.state, learning_rate_scale=lr_scale))
        except Exception as exc:
            logger.warning("METABOLIC_FLUX emit failed: %s", exc)

        answers = extract(
            paths,
            top_k                 = top_k,
            min_hop               = min_hop,
            query_embedding       = query_embedding,
            relation_prior        = relation_prior,
            vote_weight           = vote_weight,
            branch_bonus_weight   = branch_bonus_weight,
            degree_penalty_weight = degree_penalty_weight,  # Phase 151 bug fix
            adapter               = self.adapter,           # Phase 151 bug fix
        )

        # Phase 172: community consensus post-traversal re-ranking
        if auto_infer_terminal_relation and answers:
            _sri = getattr(self, "_sri", None)
            if _sri is not None:
                _sri.community_consensus_boost(answers, self.adapter)

        # Phase 95: record query result into working memory buffer
        if self._working_memory is not None:
            try:
                from core.working_memory import MemoryEntry
                import time as _time
                _pe  = trace_info.prediction_error if trace_info is not None else None
                _sol = trace_info.soliton_index    if trace_info is not None else None
                # Phase 96: extract edge triples from best path for consolidation
                _path_edges = []
                if answers and getattr(answers[0], "best_path", None) is not None:
                    ns = answers[0].best_path.nodes
                    _path_edges = [(ns[i-1], ns[i], ns[i+1]) for i in range(1, len(ns), 2)]
                self._working_memory.record(MemoryEntry(
                    timestamp        = _time.time(),
                    seeds            = original_seeds,
                    answers          = [a.entity_id for a in answers[:5]],
                    top_score        = answers[0].score if answers else 0.0,
                    soliton_index    = _sol,
                    prediction_error = _pe,
                    source           = "query",
                    path_edges       = _path_edges,
                ))
            except Exception as exc:
                logger.warning("WorkingMemory record failed: %s", exc)

        return answers

    # ------------------------------------------------------------------
    # Phase 95: Working Memory + Goal Stack attachment
    # ------------------------------------------------------------------

    def attach_valence_engine(self, engine: Any) -> None:
        """Phase 101: attach a ValenceEngine to the BeamTraversal."""
        if self._traversal is not None:
            self._traversal._valence_engine = engine

    def attach_working_memory(self, wm: Any) -> None:
        self._working_memory = wm

    def attach_goal_stack(self, stack: Any) -> None:
        self._goal_stack = stack

    def _build_priming_map(self) -> Dict[str, float]:
        """Phase 99: Thalamic Gating - build a recency-weighted node priming map from WM.

        For each entry in the last 20 WM records, contribute
        ``recency * top_score`` to every seed and answer node.
        Result is normalized to [0, 1] and returned as a dict.
        Empty dict when no WM is attached.
        """
        import math as _math
        import time as _time
        wm = self._working_memory
        if wm is None:
            return {}
        priming: Dict[str, float] = {}
        now = _time.time()
        for entry in wm.recent(20):
            age = max(0.0, now - entry.timestamp)
            recency = _math.exp(-0.01 * age)
            contribution = recency * entry.top_score
            for node in list(entry.seeds) + list(entry.answers):
                if node:
                    priming[node] = priming.get(node, 0.0) + contribution
        if not priming:
            return {}
        max_val = max(priming.values())
        if max_val <= 0.0:
            return {}
        return {k: v / max_val for k, v in priming.items()}

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def node_count(self) -> int:
        return self.adapter.node_count()

    @property
    def edge_count(self) -> int:
        return self.adapter.to_networkx().number_of_edges()

    @property
    def community_count(self) -> int:
        cm = getattr(self.adapter, "community_map", {})
        return len(set(cm.values()))

    @property
    def communities(self) -> List[frozenset[str]]:
        """Return the partition of nodes into communities."""
        # adapter._partition stores the list of frozensets
        return getattr(self.adapter, "_partition", [])

    @property
    def is_built(self) -> bool:
        return self._built

    def __repr__(self) -> str:
        status = "built" if self._built else "not built"
        return (
            f"CerebrumGraph({self.node_count} nodes, "
            f"{self.edge_count} edges, "
            f"emb={type(self._embedding_engine).__name__}, "
            f"{status})"
        )

    async def run_rem_cycle(self):
        """Phase 172: Run asynchronous shortcut synthesis (REM cycle)."""
        await self.consolidation_engine.run_rem_cycle()
