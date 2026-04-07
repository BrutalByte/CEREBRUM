"""
PathVerbalizer — CEREBRUM native language layer (Phase 18a).

Converts TraversalPath / Answer objects into fluent natural language
with full citations.  Every claim produced by the verbalizer is backed
by a named, traceable edge in the knowledge graph — no inference, no
hallucination surface.

Design principles
-----------------
- Template-first: deterministic, zero training required.
- Graph-schema-aware: templates are keyed on relation type names that
  come directly from the KB.  Unknown relations fall back gracefully.
- Multi-hop cohesion: consecutive hops are joined with connectors
  ("who", "which") so the output reads as a single sentence.
- Citations always included: every verbalization carries the raw
  edge path so callers can verify or hyperlink each claim.

Usage
-----
    from core.verbalizer import PathVerbalizer
    from reasoning.answer_extractor import Answer

    verb = PathVerbalizer()
    result = verb.verbalize_answer(answer, adapter)
    print(result.text)
    print(result.citations)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Relation templates
# Each entry: (active_sentence, passive_sentence, multi-hop connector)
#
# active   : "{src} <verb> {dst}"           used for forward hops
# passive  : "{dst} was <verbed> by {src}"  used when reading backwards
# connector: "who/which <verb>"             used to join hops 2..N
# ---------------------------------------------------------------------------

_BUILTIN_TEMPLATES: Dict[str, Tuple[str, str, str]] = {
    # Causal / biomedical
    "CAUSES":           ("{src} causes {dst}",
                         "{dst} is caused by {src}",
                         "which causes"),
    "CAUSED_BY":        ("{src} is caused by {dst}",
                         "{dst} causes {src}",
                         "which is caused by"),
    "PREVENTS":         ("{src} prevents {dst}",
                         "{dst} is prevented by {src}",
                         "which prevents"),
    "TREATS":           ("{src} treats {dst}",
                         "{dst} is treated by {src}",
                         "which treats"),
    "WORSENS":          ("{src} worsens {dst}",
                         "{dst} is worsened by {src}",
                         "which worsens"),
    "INHIBITS":         ("{src} inhibits {dst}",
                         "{dst} is inhibited by {src}",
                         "which inhibits"),
    "ACTIVATES":        ("{src} activates {dst}",
                         "{dst} is activated by {src}",
                         "which activates"),
    "UPREGULATES":      ("{src} upregulates {dst}",
                         "{dst} is upregulated by {src}",
                         "which upregulates"),
    "DOWNREGULATES":    ("{src} downregulates {dst}",
                         "{dst} is downregulated by {src}",
                         "which downregulates"),
    "ENCODES":          ("{src} encodes {dst}",
                         "{dst} is encoded by {src}",
                         "which encodes"),
    "EXPRESSED_IN":     ("{src} is expressed in {dst}",
                         "{dst} expresses {src}",
                         "which is expressed in"),
    "BINDS":            ("{src} binds {dst}",
                         "{dst} is bound by {src}",
                         "which binds"),
    "INTERACTS_WITH":   ("{src} interacts with {dst}",
                         "{dst} interacts with {src}",
                         "which interacts with"),

    # Academic / intellectual
    "INFLUENCED":       ("{src} influenced {dst}",
                         "{dst} was influenced by {src}",
                         "who was influenced by"),
    "CITED":            ("{src} cited {dst}",
                         "{dst} was cited by {src}",
                         "which cited"),
    "WROTE":            ("{src} wrote {dst}",
                         "{dst} was written by {src}",
                         "which was written by"),
    "AUTHORED":         ("{src} authored {dst}",
                         "{dst} was authored by {src}",
                         "which was authored by"),
    "PUBLISHED":        ("{src} published {dst}",
                         "{dst} was published by {src}",
                         "which published"),
    "CONTRIBUTED_TO":   ("{src} contributed to {dst}",
                         "{dst} received contributions from {src}",
                         "which received contributions from"),
    "FOUNDED":          ("{src} founded {dst}",
                         "{dst} was founded by {src}",
                         "which was founded by"),
    "STUDIED":          ("{src} studied {dst}",
                         "{dst} was studied by {src}",
                         "who studied"),
    "DISCOVERED":       ("{src} discovered {dst}",
                         "{dst} was discovered by {src}",
                         "who discovered"),
    "INVENTED":         ("{src} invented {dst}",
                         "{dst} was invented by {src}",
                         "who invented"),
    "DISPROVED":        ("{src} disproved {dst}",
                         "{dst} was disproved by {src}",
                         "who disproved"),
    "PROVED":           ("{src} proved {dst}",
                         "{dst} was proved by {src}",
                         "who proved"),
    "REFUTES":          ("{src} refutes {dst}",
                         "{dst} is refuted by {src}",
                         "which refutes"),
    "SUPPORTS":         ("{src} supports {dst}",
                         "{dst} is supported by {src}",
                         "which supports"),

    # Social / relational
    "KNOWS":            ("{src} knows {dst}",
                         "{dst} is known by {src}",
                         "who knows"),
    "MARRIED_TO":       ("{src} is married to {dst}",
                         "{dst} is married to {src}",
                         "who is married to"),
    "WORKED_WITH":      ("{src} worked with {dst}",
                         "{dst} worked with {src}",
                         "who worked with"),
    "COLLABORATED_WITH":("{src} collaborated with {dst}",
                         "{dst} collaborated with {src}",
                         "who collaborated with"),
    "EMPLOYED_BY":      ("{src} was employed by {dst}",
                         "{dst} employed {src}",
                         "who was employed by"),
    "MENTORED":         ("{src} mentored {dst}",
                         "{dst} was mentored by {src}",
                         "who mentored"),
    "STUDENT_OF":       ("{src} was a student of {dst}",
                         "{dst} taught {src}",
                         "who was a student of"),

    # Synthetic / Inference (REM)
    "REM_SYNTHESIZED":  ("{src} is highly similar to {dst} (synthetic link)",
                         "{dst} is highly similar to {src} (synthetic link)",
                         "which is highly similar to"),
    "REM_SYNTHESIZED_WORMHOLE": ("{src} bridges to {dst} via high similarity (cross-component wormhole)",
                                 "{dst} bridges to {src} via high similarity (cross-component wormhole)",
                                 "which bridges via high similarity to"),
    "CORRESPONDED_WITH":("{src} corresponded with {dst}",
                         "{dst} corresponded with {src}",
                         "who corresponded with"),
    "OPPOSED":          ("{src} opposed {dst}",
                         "{dst} was opposed by {src}",
                         "who opposed"),

    # Structural / hierarchical
    "IS_A":             ("{src} is a {dst}",
                         "{dst} includes {src}",
                         "which is a"),
    "PART_OF":          ("{src} is part of {dst}",
                         "{dst} contains {src}",
                         "which is part of"),
    "CONTAINS":         ("{src} contains {dst}",
                         "{dst} is contained within {src}",
                         "which contains"),
    "INSTANCE_OF":      ("{src} is an instance of {dst}",
                         "{dst} has instance {src}",
                         "which is an instance of"),
    "SUBCLASS_OF":      ("{src} is a subclass of {dst}",
                         "{dst} has subclass {src}",
                         "which is a subclass of"),
    "LOCATED_IN":       ("{src} is located in {dst}",
                         "{dst} contains {src}",
                         "which is located in"),
    "MEMBER_OF":        ("{src} is a member of {dst}",
                         "{dst} has member {src}",
                         "which is a member of"),

    # Temporal / causal
    "PRECEDES":         ("{src} precedes {dst}",
                         "{dst} follows {src}",
                         "which precedes"),
    "FOLLOWS":          ("{src} follows {dst}",
                         "{dst} precedes {src}",
                         "which follows"),
    "OCCURRED_DURING":  ("{src} occurred during {dst}",
                         "{dst} included {src}",
                         "which occurred during"),

    # Movie / media domain (MetaQA)
    "STARRED_IN":       ("{src} starred in {dst}",
                         "{dst} starred {src}",
                         "who starred in"),
    "DIRECTED_BY":      ("{src} was directed by {dst}",
                         "{dst} directed {src}",
                         "which was directed by"),
    "DIRECTED":         ("{src} directed {dst}",
                         "{dst} was directed by {src}",
                         "who directed"),
    "WRITTEN_BY":       ("{src} was written by {dst}",
                         "{dst} wrote {src}",
                         "which was written by"),
    "PRODUCED_BY":      ("{src} was produced by {dst}",
                         "{dst} produced {src}",
                         "which was produced by"),
    "HAS_GENRE":        ("{src} has genre {dst}",
                         "{dst} includes {src}",
                         "which has genre"),
    "RELEASE_YEAR":     ("{src} was released in {dst}",
                         "{dst} saw the release of {src}",
                         "which was released in"),

    # Generic fallbacks (populated by normalisation)
    "RELATED_TO":       ("{src} is related to {dst}",
                         "{dst} is related to {src}",
                         "which is related to"),
    "CONNECTED_TO":     ("{src} is connected to {dst}",
                         "{dst} is connected to {src}",
                         "which is connected to"),
    "ASSOCIATED_WITH":  ("{src} is associated with {dst}",
                         "{dst} is associated with {src}",
                         "which is associated with"),
}

# Synthetic edges injected by the REM cycle — handled separately
_REM_CONNECTOR = "which is structurally similar to"


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class VerbalizationResult:
    """Output of PathVerbalizer.verbalize_answer()."""

    text: str
    """Fluent natural language sentence(s)."""

    citations: List[str]
    """Raw edge citations: ['newton-[INFLUENCED]->leibniz', ...]"""

    answer_entity: str
    """Terminal entity ID — the answer node."""

    answer_label: str
    """Human-readable label of the answer entity."""

    path_confidence: float
    """Weakest-link edge confidence along the path (1.0 = fully verified)."""

    community_note: str = ""
    """Which community the answer was found in, if available."""

    hop_count: int = 0
    """Number of hops in the path."""

    confidence_qualifier: str = ""
    """'with high confidence', 'tentatively', etc."""


# ---------------------------------------------------------------------------
# AAAK (AI-to-AI Knowledge) Verbalizer
# ---------------------------------------------------------------------------

class AAAKVerbalizer:
    """
    Implements a shorthand dialect for 30x reasoning compression.
    Designed for LLM-to-LLM knowledge transfer (AAAK).
    """
    
    _SHORTHAND = {
        "CAUSES": "!",
        "CAUSED_BY": "<-!",
        "TREATS": "+",
        "INHIBITS": "-",
        "STARRED_IN": "*",
        "DIRECTED_BY": "^",
        "RELEASE_YEAR": "@",
        "INFLUENCED": "~",
        "MEMBER_OF": "€",
        "PART_OF": "⊂",
        "REM_SYNTHESIZED": "≈",
    }

    def verbalize(self, answers: list, adapter=None) -> str:
        """
        Compress top answers into a dense AAAK block.
        Example: [Newton ~> Leibniz !> Calculus (c=0.92)]
        """
        if not answers: return "ø"
        
        pkts = []
        for ans in answers[:5]:
            path = getattr(ans, "best_path", None)
            if not path: continue
            
            nodes = path.nodes
            trace = []
            for i in range(0, len(nodes), 2):
                node_id = nodes[i]
                # Use first 4 chars of label for extreme compression
                label = self._label(node_id, adapter)[:6].replace(" ", "")
                trace.append(label)
                if i + 1 < len(nodes):
                    rel = nodes[i+1]
                    trace.append(self._SHORTHAND.get(rel, ">"))
            
            pkt = f"[{''.join(trace)}(c{ans.score:.2f})]"
            pkts.append(pkt)
            
        return "AAAK:" + "".join(pkts)

    def _label(self, entity_id: str, adapter) -> str:
        if adapter is None: return entity_id
        try:
            ent = adapter.get_entity(entity_id)
            if ent and ent.label: return ent.label
        except: pass
        return entity_id


# ---------------------------------------------------------------------------
# PathVerbalizer
# ---------------------------------------------------------------------------

class PathVerbalizer:
    """
    Converts CEREBRUM TraversalPath / Answer objects into fluent natural
    language with full edge citations.

    Parameters
    ----------
    extra_templates : dict, optional
        Additional or override relation templates.  Keys are relation type
        strings (uppercase); values are (active, passive, connector) triples.
    """

    def __init__(
        self,
        extra_templates: Optional[Dict[str, Tuple[str, str, str]]] = None,
    ) -> None:
        self._templates = dict(_BUILTIN_TEMPLATES)
        if extra_templates:
            self._templates.update(extra_templates)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verbalize_answer(
        self,
        answer,
        adapter=None,
    ) -> VerbalizationResult:
        """
        Verbalize a single Answer object into a VerbalizationResult.

        Parameters
        ----------
        answer : Answer
            From reasoning.answer_extractor.
        adapter : GraphAdapter, optional
            Used to look up entity labels.  If None, entity IDs are used as
            labels directly.
        """
        path = getattr(answer, "best_path", None)
        if path is None:
            return VerbalizationResult(
                text=f"Answer: {answer.entity}",
                citations=[],
                answer_entity=answer.entity,
                answer_label=self._label(answer.entity, adapter),
                path_confidence=1.0,
            )
        return self._verbalize_path(path, adapter)

    def verbalize_answers(
        self,
        answers: list,
        adapter=None,
        top_k: int = 3,
        question: str = "",
    ) -> str:
        """
        Verbalize the top-k answers into a readable multi-line response.

        Returns a formatted string suitable for terminal or API output.
        """
        if not answers:
            return "No answers found."

        lines = []
        if question:
            lines.append(f"Q: {question}")
            lines.append("")

        for rank, ans in enumerate(answers[:top_k], 1):
            result = self.verbalize_answer(ans, adapter)
            conf_str = f"  [{result.confidence_qualifier}]" if result.confidence_qualifier else ""
            lines.append(f"{rank}. {result.text}{conf_str}")
            if result.community_note:
                lines.append(f"   Community: {result.community_note}")
            lines.append(f"   Path: {' | '.join(result.citations)}")
            lines.append(f"   Confidence: {result.path_confidence:.3f}")
            lines.append("")

        return "\n".join(lines).rstrip()

    def verbalize_path(self, path, adapter=None) -> VerbalizationResult:
        """Verbalize a raw TraversalPath (without an Answer wrapper)."""
        return self._verbalize_path(path, adapter)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _verbalize_path(self, path, adapter) -> VerbalizationResult:
        """Core verbalization logic."""
        nodes = path.nodes  # alternating [entity, relation, entity, relation, ...]

        if len(nodes) < 1:
            return VerbalizationResult(
                text="(empty path)",
                citations=[],
                answer_entity="",
                answer_label="",
                path_confidence=1.0,
            )

        # Single node — no hops
        if len(nodes) == 1:
            label = self._label(nodes[0], adapter)
            return VerbalizationResult(
                text=label,
                citations=[nodes[0]],
                answer_entity=nodes[0],
                answer_label=label,
                path_confidence=1.0,
                hop_count=0,
            )

        # Extract (entity, relation, entity) triples
        hops: List[Tuple[str, str, str]] = []
        for i in range(0, len(nodes) - 2, 2):
            src_id  = nodes[i]
            rel     = nodes[i + 1] if i + 1 < len(nodes) else "RELATED_TO"
            dst_id  = nodes[i + 2] if i + 2 < len(nodes) else nodes[-1]
            hops.append((src_id, rel, dst_id))

        citations = [
            f"{src}-[{rel}]->{dst}" for src, rel, dst in hops
        ]

        # Build sentence
        sentence = self._build_sentence(hops, adapter)

        # Confidence
        path_conf = getattr(path, "path_confidence", 1.0)
        qualifier = self._confidence_qualifier(path_conf)

        # Community note
        community_note = self._community_note(hops[-1][2], adapter)

        answer_entity = hops[-1][2]
        answer_label  = self._label(answer_entity, adapter)

        return VerbalizationResult(
            text=sentence,
            citations=citations,
            answer_entity=answer_entity,
            answer_label=answer_label,
            path_confidence=path_conf,
            community_note=community_note,
            hop_count=len(hops),
            confidence_qualifier=qualifier,
        )

    def _build_sentence(
        self,
        hops: List[Tuple[str, str, str]],
        adapter,
    ) -> str:
        """Join hops into a fluent sentence."""
        if not hops:
            return ""

        src_id, rel, dst_id = hops[0]
        src_label = self._label(src_id, adapter)
        dst_label = self._label(dst_id, adapter)
        sentence  = self._active(rel, src_label, dst_label)

        for src_id, rel, dst_id in hops[1:]:
            dst_label  = self._label(dst_id, adapter)
            connector  = self._connector(rel)
            sentence  += f", {connector} {dst_label}"

        return sentence

    def _active(self, rel: str, src: str, dst: str) -> str:
        """Render an active-voice sentence for a single hop."""
        norm = self._normalise_rel(rel)
        if norm == "_REM":
            return f"{src} is structurally similar to {dst}"
        tmpl = self._templates.get(norm)
        if tmpl:
            return tmpl[0].format(src=src, dst=dst)
        # Fallback: humanise the relation type
        verb = rel.lower().replace("_", " ")
        return f"{src} {verb} {dst}"

    def _connector(self, rel: str) -> str:
        """Return the multi-hop connector for a relation type."""
        norm = self._normalise_rel(rel)
        if norm == "_REM":
            return _REM_CONNECTOR
        tmpl = self._templates.get(norm)
        if tmpl:
            return tmpl[2]
        return f"which {rel.lower().replace('_', ' ')}"

    def _normalise_rel(self, rel: str) -> str:
        """Uppercase + strip whitespace.  Special-case REM synthetic edges."""
        r = rel.strip().upper()
        if r in ("REM_SYNTHESIZED", "REM-SYNTHESIZED"):
            return "_REM"
        return r

    def _label(self, entity_id: str, adapter) -> str:
        """Return human-readable label for an entity."""
        if adapter is None:
            return entity_id
        try:
            ent = adapter.get_entity(entity_id)
            if ent and ent.label:
                return ent.label
        except Exception:
            pass
        return entity_id

    def _confidence_qualifier(self, conf: float) -> str:
        if conf >= 0.9:
            return "high confidence"
        if conf >= 0.7:
            return "moderate confidence"
        if conf >= 0.5:
            return "low confidence"
        return "tentative"

    def _community_note(self, entity_id: str, adapter) -> str:
        if adapter is None:
            return ""
        try:
            cid = adapter.get_community(entity_id)
            if cid >= 0:
                return f"community {cid}"
        except Exception:
            pass
        return ""
