"""
CausalDiscoveryEngine — Phase 216-B.

Infers causal direction from graph co-occurrence statistics, eliminating the
need for human annotation of CAUSAL_RELATIONS.

Algorithm
---------
For each relation type r in the graph we compute three signals:

1. Fan-out asymmetry
   P(B | A, r)  = avg targets per source node for relation r
   P(A | B, r⁻) = avg sources per target node (reverse direction)
   asymmetry = |fan_out_forward - fan_out_reverse| / (fan_out_forward + fan_out_reverse + ε)
   High asymmetry → directional (causal) relation.

2. Collider density
   For every triple A→[r]→B, count V-structures where a third node C also
   connects to B via a *different* relation type (A→B←C pattern with no A–C
   edge).  High collider density indicates B is a common effect — evidence
   that r is causal.

3. Temporal consistency (if edges have valid_from / valid_to timestamps)
   What fraction of A→[r]→B triples have valid_from(A) <= valid_from(B)?
   Temporal ordering consistent with the direction → causal signal.

Final causal_score = weighted sum of the three signals, normalised to [0, 1].
Relations scoring above threshold (default 0.7) are injected into SymbolicValidator
as CAUSAL_ORDERING constraints.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.graph_adapter import GraphAdapter
    from core.symbolic_engine import SymbolicValidator

logger = logging.getLogger("cerebrum.causal_discovery")


class CausalDiscoveryEngine:
    """
    Discovers causal relations from graph topology and injects them as
    SymbolicValidator constraints.

    Parameters
    ----------
    asymmetry_weight
        Weight for fan-out asymmetry signal (default 0.5).
    collider_weight
        Weight for collider density signal (default 0.3).
    temporal_weight
        Weight for temporal consistency signal (default 0.2).
    max_sample_per_relation
        Cap on edges sampled per relation type (keeps O(E) bounded).
    """

    def __init__(
        self,
        asymmetry_weight: float = 0.5,
        collider_weight: float = 0.3,
        temporal_weight: float = 0.2,
        max_sample_per_relation: int = 500,
    ) -> None:
        self.asymmetry_weight = asymmetry_weight
        self.collider_weight = collider_weight
        self.temporal_weight = temporal_weight
        self.max_sample_per_relation = max_sample_per_relation
        self._scores: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def discover(self, adapter: "GraphAdapter") -> Dict[str, float]:
        """
        Analyse the graph and return a {relation_type: causal_score} dict.
        Scores are in [0, 1]; higher = more likely causal.
        Results are cached; call again after graph changes.
        """
        logger.info("CausalDiscoveryEngine: scanning graph for causal signals...")
        try:
            G = adapter.to_networkx()
        except Exception as e:
            logger.warning("CausalDiscoveryEngine: to_networkx() failed: %s", e)
            return {}

        # Collect per-relation statistics
        fwd_sources: Dict[str, set] = defaultdict(set)   # rel → set of source nodes
        fwd_targets: Dict[str, set] = defaultdict(set)   # rel → set of target nodes
        fwd_edges: Dict[str, list] = defaultdict(list)   # rel → [(src, tgt, valid_from)]
        rev_sources: Dict[str, set] = defaultdict(set)   # rel → set of nodes targeted (reverse)

        for src, tgt, data in G.edges(data=True):
            rel = data.get("relation", data.get("relation_type", ""))
            if not rel:
                continue
            lst = fwd_edges[rel]
            if len(lst) < self.max_sample_per_relation:
                lst.append((src, tgt, data.get("valid_from", None)))
            fwd_sources[rel].add(src)
            fwd_targets[rel].add(tgt)
            rev_sources[rel].add(tgt)   # targets in forward = sources in reverse

        all_rels = set(fwd_edges.keys())
        if not all_rels:
            logger.warning("CausalDiscoveryEngine: no labelled edges found")
            return {}

        results: Dict[str, float] = {}

        for rel in all_rels:
            edges = fwd_edges[rel]
            n_fwd_src = max(1, len(fwd_sources[rel]))
            n_fwd_tgt = max(1, len(fwd_targets[rel]))

            # --- Signal 1: Fan-out asymmetry ---
            fan_out_fwd = len(fwd_targets[rel]) / n_fwd_src  # avg targets per source
            fan_out_rev = len(fwd_sources[rel]) / n_fwd_tgt  # avg sources per target
            denom = fan_out_fwd + fan_out_rev + 1e-9
            asym = abs(fan_out_fwd - fan_out_rev) / denom

            # --- Signal 2: Collider density ---
            collider_hits = 0
            total_checked = 0
            for src, tgt, _ in edges[:min(100, len(edges))]:
                total_checked += 1
                # Count nodes with edges into tgt via a different relation
                in_rels = set()
                for pred in G.predecessors(tgt):
                    if pred != src:
                        for _, _, d in G.in_edges(tgt, data=True):
                            r2 = d.get("relation", d.get("relation_type", ""))
                            if r2 and r2 != rel:
                                in_rels.add(r2)
                                break
                # V-structure: at least one other inbound relation at tgt
                if in_rels:
                    # Check no edge from src to any of those predecessors
                    no_common = all(
                        not G.has_edge(src, pred)
                        for pred in G.predecessors(tgt)
                        if pred != src
                    )
                    if no_common:
                        collider_hits += 1
            collider_score = collider_hits / max(1, total_checked)

            # --- Signal 3: Temporal consistency ---
            temporal_hits = 0
            temporal_total = 0
            for src, tgt, vf_src in edges:
                if vf_src is None:
                    continue
                # Find valid_from for tgt in any of its outgoing edges as a proxy
                vf_tgt = None
                for _, _, d in G.out_edges(tgt, data=True):
                    vf_tgt = d.get("valid_from", None)
                    if vf_tgt is not None:
                        break
                if vf_tgt is not None:
                    temporal_total += 1
                    if vf_src <= vf_tgt:
                        temporal_hits += 1
            temporal_score = (temporal_hits / temporal_total) if temporal_total > 0 else 0.5

            causal_score = (
                self.asymmetry_weight * asym
                + self.collider_weight * collider_score
                + self.temporal_weight * (temporal_score - 0.5) * 2.0  # centre on 0
            )
            causal_score = float(max(0.0, min(1.0, causal_score)))
            results[rel] = causal_score

        self._scores = results
        logger.info(
            "CausalDiscoveryEngine: scored %d relations; top causal: %s",
            len(results),
            sorted(results, key=results.get, reverse=True)[:5],
        )
        return results

    def inject_priors(
        self,
        validator: "SymbolicValidator",
        threshold: float = 0.7,
    ) -> int:
        """
        Register relations above threshold as causal in the SymbolicValidator.
        Returns the number of relations injected.
        """
        if not self._scores:
            logger.warning("CausalDiscoveryEngine.inject_priors(): no scores — run discover() first")
            return 0
        injected = 0
        for rel, score in self._scores.items():
            if score >= threshold:
                validator.register_causal_relation(rel)
                logger.debug("Injected causal relation '%s' (score=%.3f)", rel, score)
                injected += 1
        logger.info("CausalDiscoveryEngine: injected %d causal relations (threshold=%.2f)", injected, threshold)
        return injected

    def to_dict(self) -> dict:
        return {"scores": dict(self._scores)}

    @classmethod
    def from_dict(cls, d: dict, **kwargs) -> "CausalDiscoveryEngine":
        obj = cls(**kwargs)
        obj._scores = d.get("scores", {})
        return obj

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str, **kwargs) -> "CausalDiscoveryEngine":
        with open(path) as f:
            return cls.from_dict(json.load(f), **kwargs)
