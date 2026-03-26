"""
THALAMUS — Ingestion Preprocessing Pipeline.

IngestionPipeline is a composable preprocessing layer that normalizes
raw graph data before it enters the CEREBRUM graph. It addresses three
structural GIGO problems at ingest time rather than after the fact:

Problem 1 — Entity fragmentation
  "Tom Hanks", "tom hanks", "TomHanks", "T. Hanks" become four
  disconnected nodes. Traversal starting from any one of them misses
  all paths through the other three. Entity normalization and
  deduplication collapse variants to a single canonical node ID.

Problem 2 — Relation type inconsistency
  The contradiction engine's CONTRADICTION_PAIRS dict uses uppercase
  canonical strings ("ACTIVATES", "INHIBITS"). The CSA formula's
  edge_type_weight lookup also uses relation type strings. If your
  source data uses "activates", "act", "ACTIVATION", or
  "positively_regulates", these never match — contradiction detection
  is blind and attention weights are inconsistent. Relation
  normalization maps all variants to canonical form at ingest.

Problem 3 — Confidence defaults to 1.0
  The REM engine prunes edges below a confidence threshold. The
  contradiction engine computes authority_delta from confidence scores.
  Neither does anything useful if every edge enters the graph at the
  default confidence=1.0. Confidence-at-ingest lets you assign
  meaningful scores based on source metadata before edges are stored.

Usage
-----
    from core.thalamus import IngestionPipeline
    from adapters.csv_adapter import load_csv_adapter

    pipeline = IngestionPipeline(
        entity_normalizer=lambda s: s.lower().strip().replace(" ", "_"),
        entity_dedup_map={
            "tom hanks": "tom_hanks",
            "tomhanks":  "tom_hanks",
            "t. hanks":  "tom_hanks",
        },
        relation_map={
            "activates":             "ACTIVATES",
            "positively_regulates":  "ACTIVATES",
            "upregulates":           "ACTIVATES",
            "inhibits":              "INHIBITS",
            "negatively_regulates":  "INHIBITS",
            "downregulates":         "INHIBITS",
        },
        confidence_fn=lambda src, tgt, rel, meta: float(meta.get("score", 1.0)),
        provenance_fn=lambda src, tgt, rel, meta: meta.get("source_db", ""),
    )

    adapter = load_csv_adapter("kg.csv", pipeline=pipeline)

All parameters are optional. An IngestionPipeline() with no arguments
applies only safe defaults: strip whitespace from entity IDs and
uppercase relation strings. Existing graphs load identically.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# ProcessedEdge — output of one pipeline pass
# ---------------------------------------------------------------------------

@dataclass
class ProcessedEdge:
    """
    A single edge after normalization and enrichment by IngestionPipeline.

    This is the unit that adapters write into the graph. All fields that
    live on Edge (confidence, provenance, weight) are populated here so
    that adapters don't need to know which fields the pipeline set.
    """
    source: str
    target: str
    relation: str
    confidence: float = 1.0
    provenance: str = ""
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Built-in normalizer defaults
# ---------------------------------------------------------------------------

def _default_entity_normalizer(raw: str) -> str:
    """Strip leading/trailing whitespace. No case transformation by default
    — case changes are too risky for existing graphs (e.g. MetaQA entity IDs
    are case-sensitive). Pass an explicit entity_normalizer to change this."""
    return raw.strip()


def _default_relation_normalizer(raw: str) -> str:
    """Strip whitespace and uppercase. Relations in CEREBRUM are canonical
    UPPERCASE strings throughout (CSA weights, contradiction pairs, etc.)."""
    normed = raw.strip().upper()
    return normed if normed else "RELATED_TO"


# ---------------------------------------------------------------------------
# IngestionPipeline
# ---------------------------------------------------------------------------

class IngestionPipeline:
    """
    Composable preprocessing pipeline for CEREBRUM edge ingestion.

    All parameters are optional. Unset parameters apply safe defaults
    (whitespace-strip for entities, uppercase for relations). Existing
    code that passes no pipeline is completely unaffected.

    Parameters
    ----------
    entity_normalizer : callable(str) -> str, optional
        Applied to every source and target entity ID before dedup lookup.
        Default: strip whitespace only.
        Example: ``lambda s: s.lower().strip().replace(" ", "_")``

    entity_dedup_map : dict {alias -> canonical_id}, optional
        Applied *after* entity_normalizer. Maps normalized variant strings
        to a canonical ID so multiple raw spellings collapse to one node.
        Lookup is exact on the already-normalized string — run the same
        entity_normalizer on your keys when building this map.
        Example: ``{"tom hanks": "tom_hanks", "tomhanks": "tom_hanks"}``

    relation_map : dict {raw -> canonical} or callable(str) -> str, optional
        Normalizes relation type strings.
        - If a dict: keys are matched case-insensitively (stored lowercase
          internally). Unmapped relations fall through to the default
          normalizer (uppercase + strip).
        - If a callable: replaces the default normalizer entirely.
        Example dict: ``{"activates": "ACTIVATES", "positively_regulates": "ACTIVATES"}``
        Example callable: ``lambda r: r.strip().upper().replace(" ", "_")``

    confidence_fn : callable(src, tgt, rel, meta) -> float, optional
        Returns edge confidence in [0, 1]. Called after entity and relation
        normalization so src/tgt/rel are already canonical. ``meta`` is the
        dict of extra attributes on the edge (e.g. extra CSV columns).
        Result is clamped to [0.0, 1.0] automatically.
        If None, confidence falls through to meta.get("confidence", 1.0).
        Example: ``lambda src, tgt, rel, meta: float(meta.get("score", 1.0))``

    provenance_fn : callable(src, tgt, rel, meta) -> str, optional
        Returns a provenance string for the edge (e.g. "pubmed:123",
        "wikidata:Q42", "internal:csv_import"). Used by the contradiction
        engine to compute authority_delta between conflicting claims.
        If None, provenance falls through to meta.get("provenance", "").
        Example: ``lambda src, tgt, rel, meta: meta.get("source_db", "")``
    """

    def __init__(
        self,
        entity_normalizer: Optional[Callable[[str], str]] = None,
        entity_dedup_map: Optional[Dict[str, str]] = None,
        relation_map: Optional[Any] = None,   # dict or callable
        confidence_fn: Optional[Callable] = None,
        provenance_fn: Optional[Callable] = None,
        namespace: str = "",
    ) -> None:
        self._entity_norm   = entity_normalizer or _default_entity_normalizer
        self._dedup_map     = entity_dedup_map or {}
        self._confidence_fn = confidence_fn
        self._provenance_fn = provenance_fn
        self._namespace     = namespace

        # Build relation normalizer from whatever the caller passed
        if relation_map is None:
            self._relation_norm: Callable[[str], str] = _default_relation_normalizer
        elif callable(relation_map):
            self._relation_norm = relation_map
        else:
            # dict — keys stored lowercase for case-insensitive lookup;
            # unmapped relations fall through to _default_relation_normalizer
            _rmap: Dict[str, str] = {
                k.strip().lower(): v.strip()
                for k, v in relation_map.items()
            }
            def _dict_norm(raw: str, _m: Dict[str, str] = _rmap) -> str:
                return _m.get(raw.strip().lower(), _default_relation_normalizer(raw))
            self._relation_norm = _dict_norm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        source: str,
        target: str,
        relation: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProcessedEdge:
        """
        Apply the full pipeline to one raw edge and return a ProcessedEdge.

        Steps (in order):
          1. Entity normalization    — entity_normalizer(source/target)
          2. Entity deduplication   — dedup_map lookup
          3. Relation normalization  — relation_map / default uppercase
          4. Confidence assignment   — confidence_fn or meta fallback
          5. Provenance assignment   — provenance_fn or meta fallback
          6. Weight + extra props    — pass-through from metadata

        Parameters
        ----------
        source   : raw source entity ID
        target   : raw target entity ID
        relation : raw relation type string
        metadata : optional dict of extra attributes (e.g. extra CSV columns,
                   StreamEvent.metadata, JSON payload fields)
        """
        meta = metadata or {}

        # 1. Entity normalization
        src = self._entity_norm(source)
        tgt = self._entity_norm(target)

        # 2. Entity deduplication
        src = self._dedup_map.get(src, src)
        tgt = self._dedup_map.get(tgt, tgt)

        # 2b. Namespace isolation — applied after dedup so aliases collapse first
        if self._namespace:
            src = f"{self._namespace}:{src}"
            tgt = f"{self._namespace}:{tgt}"

        # 3. Relation normalization
        rel = self._relation_norm(relation)

        # 4. Confidence — fn takes priority; fallback to metadata key
        if self._confidence_fn is not None:
            try:
                conf = float(self._confidence_fn(src, tgt, rel, meta))
            except Exception:
                conf = 1.0
        else:
            try:
                conf = float(meta.get("confidence", 1.0))
            except (TypeError, ValueError):
                conf = 1.0
        conf = max(0.0, min(1.0, conf))  # clamp to [0, 1]

        # 5. Provenance — fn takes priority; fallback to metadata key
        if self._provenance_fn is not None:
            try:
                prov = str(self._provenance_fn(src, tgt, rel, meta))
            except Exception:
                prov = ""
        else:
            prov = str(meta.get("provenance", ""))

        # 6. Weight (pass through from metadata)
        try:
            weight = float(meta.get("weight", 1.0))
        except (TypeError, ValueError):
            weight = 1.0

        # Remaining metadata — fields already promoted to first-class attrs
        # are excluded so they don't double-up as edge properties
        _promoted = {"confidence", "provenance", "weight"}
        props = {k: v for k, v in meta.items() if k not in _promoted}

        return ProcessedEdge(
            source=src,
            target=tgt,
            relation=rel,
            confidence=conf,
            provenance=prov,
            weight=weight,
            properties=props,
        )

    def process_triple(
        self,
        triple: Tuple[str, str, str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProcessedEdge:
        """
        Convenience wrapper for (source, relation, target) tuples.

        Note: argument order is (source, relation, target) — the standard
        triple convention in CEREBRUM, not (source, target, relation).
        """
        src, rel, tgt = triple
        return self.process(src, tgt, rel, metadata)

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    @property
    def has_dedup(self) -> bool:
        """True if an entity dedup map was provided."""
        return bool(self._dedup_map)

    @property
    def has_confidence(self) -> bool:
        """True if a confidence_fn was provided (vs. metadata fallback)."""
        return self._confidence_fn is not None

    @property
    def has_provenance(self) -> bool:
        """True if a provenance_fn was provided (vs. metadata fallback)."""
        return self._provenance_fn is not None

    def dedup_size(self) -> int:
        """Number of alias entries in the dedup map."""
        return len(self._dedup_map)
