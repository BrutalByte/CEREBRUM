"""
TextIngestor — CEREBRUM text-to-graph pipeline (Phase 21).

Reads plain text and extracts (subject, relation, object) triples, adding
new knowledge to the graph with full provenance.  No LLM required.

How it works
------------
**Stage 1 — Entity anchoring** (always available, no dependencies)
  Scan every sentence for known graph entity labels.  Entities in the graph
  are the anchors.  Between adjacent anchor pairs, extract the verb phrase
  and map it to a graph relation type.

  "Newton influenced Leibniz who invented calculus."
  → Newton (known) → influenced (→ INFLUENCED) → Leibniz (known)
  → Leibniz (known) → invented (→ INVENTED) → calculus (known)

  New entities (not already in the graph) are detected via capitalization
  heuristics and added as new nodes if confidence exceeds the threshold.

**Stage 2 — spaCy enhancement** (optional; installed separately)
  When ``en_core_web_sm`` or ``en_core_web_md`` is available, the pipeline
  also uses dependency-tree SVO extraction, which catches relations that
  don't rely on two entities appearing close together.

  Install:  pip install spacy && python -m spacy download en_core_web_sm

**Provenance**
  Every edge added via text ingest carries:
    provenance = "text_ingest"
    source_text = the sentence it was extracted from
    source_hash = sha256[:8] of the full input text
    confidence  = computed from extraction + linking quality

**Rollback**
  A snapshot of all added edges is kept after each run.
  ``rollback()`` removes them atomically.

Usage
-----
    from adapters.networkx_adapter import NetworkXAdapter
    from core.text_ingestor import TextIngestor

    adapter  = NetworkXAdapter.from_csv("kb.csv")
    ingestor = TextIngestor(adapter)

    report = ingestor.ingest_text(
        "Aspirin inhibits COX-2 which promotes inflammation.",
        dry_run=True,
    )
    print(report.summary())

    report = ingestor.ingest_file("paper.txt")
    print(f"Added {report.edges_added} edges from {report.sentences_processed} sentences")
"""
from __future__ import annotations

import re
import time
import hashlib
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Verb → relation type mapping
# Keyed by stem prefix (first 6 chars, lowercase) for fast matching.
# Value: (RELATION_TYPE, confidence_factor)
# ---------------------------------------------------------------------------

_VERB_MAP: Dict[str, Tuple[str, float]] = {
    # Academic / intellectual
    "influe": ("INFLUENCED",       0.90),
    "inspir": ("INSPIRED",         0.85),
    "discov": ("DISCOVERED",       0.90),
    "invent": ("INVENTED",         0.90),
    "wrote":  ("WROTE",            0.90),  # irregular kept verbatim
    "write":  ("WROTE",            0.85),
    "author": ("AUTHORED",         0.85),
    "publis": ("PUBLISHED",        0.85),
    "studie": ("STUDIED",          0.85),
    "studi":  ("STUDIED",          0.85),
    "proved": ("PROVED",           0.90),
    "prove":  ("PROVED",           0.85),
    "dispro": ("DISPROVED",        0.85),
    "refute": ("REFUTES",          0.85),
    "refut":  ("REFUTES",          0.85),
    "contri": ("CONTRIBUTED_TO",   0.80),
    "mentor": ("MENTORED",         0.85),
    "teache": ("MENTORED",         0.80),
    "taught": ("MENTORED",         0.85),
    "cited":  ("CITED",            0.90),
    "cites":  ("CITED",            0.85),
    "citat":  ("CITED",            0.80),
    "founde": ("FOUNDED",          0.85),
    "found":  ("FOUNDED",          0.80),
    # Social / organizational
    "collab": ("COLLABORATED_WITH",0.85),
    "worked": ("WORKED_WITH",      0.80),
    "work":   ("WORKED_WITH",      0.70),
    "employ": ("EMPLOYED_BY",      0.85),
    "hired":  ("EMPLOYED_BY",      0.80),
    "knows":  ("KNOWS",            0.80),
    "know":   ("KNOWS",            0.70),
    "marrie": ("MARRIED_TO",       0.90),
    "opposi": ("OPPOSED",          0.85),
    "oppose": ("OPPOSED",          0.85),
    "suppor": ("SUPPORTS",         0.80),
    "allied": ("ALLIED",           0.85),
    "rivals": ("RIVALED",          0.85),
    "rivale": ("RIVALED",          0.85),
    "admir":  ("ADMIRED",          0.80),
    "admire": ("ADMIRED",          0.80),
    # Causal / biological
    "causes": ("CAUSES",           0.90),
    "caused": ("CAUSES",           0.85),
    "cause":  ("CAUSES",           0.85),
    "treats": ("TREATS",           0.90),
    "treat":  ("TREATS",           0.85),
    "preve":  ("PREVENTS",         0.85),
    "inhibi": ("INHIBITS",         0.90),
    "inhibit":("INHIBITS",         0.90),
    "activa": ("ACTIVATES",        0.90),
    "promot": ("PROMOTES",         0.85),
    "encodes":("ENCODES",          0.90),
    "encode": ("ENCODES",          0.85),
    "binds":  ("BINDS",            0.90),
    "bindin": ("BINDS",            0.85),
    "inter":  ("INTERACTS_WITH",   0.75),
    "expres": ("EXPRESSED_IN",     0.80),
    "upregu": ("UPREGULATES",      0.90),
    "downre": ("DOWNREGULATES",    0.90),
    "reduce": ("INHIBITS",         0.75),
    "reduc":  ("INHIBITS",         0.70),
    "increa": ("ACTIVATES",        0.70),
    "worsens":("WORSENS",          0.85),
    "worsen": ("WORSENS",          0.80),
    # Structural / spatial
    "locate": ("LOCATED_IN",       0.85),
    "locat":  ("LOCATED_IN",       0.80),
    "lived":  ("LIVED_IN",         0.90),
    "lives":  ("LIVED_IN",         0.85),
    "reside": ("LOCATED_IN",       0.85),
    "partof": ("PART_OF",          0.85),
    "member": ("MEMBER_OF",        0.80),
    "preced": ("PRECEDES",         0.85),
    "follow": ("FOLLOWS",          0.80),
    # Film / media
    "direct": ("DIRECTED",         0.85),
    "starre": ("STARRED_IN",       0.90),
    "starred":("STARRED_IN",       0.90),
    "appear": ("STARRED_IN",       0.75),
    "produc": ("PRODUCED_BY",      0.80),
    # History / politics
    "ruled":  ("RULED",            0.90),
    "rules":  ("RULED",            0.85),
    "rule":   ("RULED",            0.80),
    "led":    ("LED",              0.85),
    "leads":  ("LED",              0.80),
    "lead":   ("LED",              0.75),
    "governe":("RULED",            0.80),
    "conqui": ("LED",              0.75),
    "visite": ("VISITED",          0.85),
    "visits": ("VISITED",          0.85),
}

# Auxiliary verbs to skip when scanning verb tokens
_AUX_VERBS: Set[str] = {
    "is", "was", "were", "are", "be", "been", "being",
    "has", "have", "had", "having",
    "will", "would", "shall", "should",
    "may", "might", "can", "could", "must",
    "do", "does", "did",
}

# Prepositions that can appear between verb and object — strip them
_PREPS: Set[str] = {
    "by", "with", "to", "of", "from", "in", "on", "at", "for", "as",
}

# Minimum length for a token to be considered a potential new entity
_MIN_ENTITY_LEN = 3


# ---------------------------------------------------------------------------
# Internal dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RawTriple:
    """Unverified triple extracted from text."""
    subject_text: str
    verb_text: str
    object_text: str
    sentence: str
    confidence: float    # extraction quality
    method: str          # "anchored" | "spacy" | "capitalized"


@dataclass
class LinkedTriple:
    """Triple with entities resolved to graph IDs."""
    source_id: str
    source_label: str
    source_new: bool      # True if added as a new node
    relation: str
    target_id: str
    target_label: str
    target_new: bool
    confidence: float
    sentence: str
    verb_text: str


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class IngestReport:
    """Result of one TextIngestor run."""

    text_length: int
    sentences_processed: int
    entities_found: int
    entities_linked: int       # matched to existing graph nodes
    entities_new: int          # added as new nodes
    triples_extracted: int
    triples_accepted: int
    triples_skipped_duplicate: int
    triples_skipped_low_confidence: int
    edges_added: int
    nodes_added: int
    added_triples: List[Tuple[str, str, str, float]]   # (src, rel, dst, conf)
    skipped_triples: List[Tuple[str, str, str, str]]   # (src, rel, dst, reason)
    duration_seconds: float
    provenance: str            # "text_ingest:{sha256[:8]}"
    dry_run: bool
    timestamp: float = field(default_factory=time.time)

    def summary(self) -> str:
        mode = "dry-run" if self.dry_run else "applied"
        lines = [
            f"IngestReport ({mode})",
            f"  Text length    : {self.text_length} chars",
            f"  Sentences      : {self.sentences_processed}",
            f"  Entities found : {self.entities_found}"
            f"  ({self.entities_linked} linked, {self.entities_new} new)",
            f"  Triples        : {self.triples_extracted} extracted"
            f"  -> {self.triples_accepted} accepted"
            f"  ({self.triples_skipped_duplicate} dup,"
            f"  {self.triples_skipped_low_confidence} low-conf)",
            f"  Edges added    : {self.edges_added}",
            f"  Nodes added    : {self.nodes_added}",
            f"  Provenance     : {self.provenance}",
            f"  Duration       : {self.duration_seconds:.3f}s",
        ]
        if self.added_triples:
            lines.append("  Added:")
            for src, rel, dst, conf in self.added_triples[:10]:
                lines.append(f"    {src} -[{rel}]-> {dst}  (conf={conf:.2f})")
            if len(self.added_triples) > 10:
                lines.append(f"    ... and {len(self.added_triples)-10} more")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# TextIngestor
# ---------------------------------------------------------------------------

class TextIngestor:
    """
    Extract triples from plain text and ingest them into the graph.

    Parameters
    ----------
    adapter : GraphAdapter
        Target graph.  Must expose ``to_networkx()``.
    min_confidence : float
        Triples below this confidence are discarded.
    create_new_entities : bool
        When True, entities not already in the graph are created as new nodes.
        When False, only triples where both endpoints already exist are added.
    use_spacy : bool
        Attempt to use spaCy for enhanced extraction.  Falls back silently
        if spaCy or its models are not installed.
    """

    def __init__(
        self,
        adapter,
        min_confidence: float = 0.30,
        create_new_entities: bool = True,
        use_spacy: bool = True,
    ) -> None:
        self._adapter           = adapter
        self._min_confidence    = min_confidence
        self._create_new        = create_new_entities
        self._use_spacy         = use_spacy
        self._lock              = threading.RLock()
        self._snapshot: Optional[List[dict]] = None
        self._last_report: Optional[IngestReport] = None

        # Build entity label index from graph
        self._entity_index: Optional[Dict[str, str]] = None  # lower_label → entity_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_text(self, text: str, dry_run: bool = False) -> IngestReport:
        """
        Extract triples from *text* and ingest into the graph.

        Parameters
        ----------
        text : str
            Plain text — sentences, paragraphs, or a full document.
        dry_run : bool
            If True, extract and report without modifying the graph.

        Returns
        -------
        IngestReport
        """
        with self._lock:
            t0 = time.time()
            text = text.strip()
            if not text:
                return self._empty_report(dry_run, t0)

            prov = "text_ingest:" + hashlib.sha256(text.encode()).hexdigest()[:8]

            # Rebuild entity index from current graph state
            self._entity_index = self._build_entity_index()

            # Stage 1: extract raw triples
            raw = self._extract_triples(text)

            # Stage 2: link entities + map relations
            linked, skipped = self._link_triples(raw)

            # Stage 3: filter by confidence
            accepted = [t for t in linked if t.confidence >= self._min_confidence]
            low_conf = len(linked) - len(accepted)

            # Stage 4: deduplicate against graph
            G = self._adapter.to_networkx()
            fresh, duplicates = self._dedup(G, accepted)

            # Stage 5: materialize (unless dry_run)
            if not dry_run:
                self._snapshot = []
                added_count, new_nodes = self._materialize(G, fresh, prov)
            else:
                added_count  = len(fresh)
                new_nodes    = sum(1 for t in fresh if t.source_new or t.target_new)

            added_triples  = [(t.source_id, t.relation, t.target_id, t.confidence) for t in fresh]
            skipped_list   = [(t.source_id, t.relation, t.target_id, "duplicate") for t in duplicates]
            skipped_list  += [(r.subject_text, "?", r.object_text, reason) for r, reason in skipped]

            entities_found  = len({r.subject_text for r in raw} | {r.object_text for r in raw})
            entities_linked = len({t.source_id for t in linked if not t.source_new}
                                  | {t.target_id for t in linked if not t.target_new})
            entities_new_n  = len({t.source_id for t in fresh if t.source_new}
                                  | {t.target_id for t in fresh if t.target_new})

            report = IngestReport(
                text_length=len(text),
                sentences_processed=len(_split_sentences(text)),
                entities_found=entities_found,
                entities_linked=entities_linked,
                entities_new=entities_new_n,
                triples_extracted=len(raw),
                triples_accepted=len(fresh),
                triples_skipped_duplicate=len(duplicates),
                triples_skipped_low_confidence=low_conf,
                edges_added=added_count,
                nodes_added=new_nodes,
                added_triples=added_triples,
                skipped_triples=skipped_list,
                duration_seconds=time.time() - t0,
                provenance=prov,
                dry_run=dry_run,
            )
            self._last_report = report
            return report

    def ingest_file(self, path: str, dry_run: bool = False) -> IngestReport:
        """Read a text file and ingest it."""
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        return self.ingest_text(text, dry_run=dry_run)

    def rollback(self) -> int:
        """
        Remove all edges (and orphaned nodes) added by the last non-dry run.

        Returns
        -------
        int
            Number of edges removed.

        Raises
        ------
        RuntimeError
            If no prior non-dry-run exists.
        """
        with self._lock:
            if self._snapshot is None:
                raise RuntimeError(
                    "No prior ingest run to roll back.  "
                    "Call ingest_text(dry_run=False) first."
                )
            G = self._adapter.to_networkx()
            removed = 0
            for entry in self._snapshot:
                u, v = entry["source"], entry["target"]
                if G.has_edge(u, v):
                    G.remove_edge(u, v)
                    removed += 1
            # Remove newly added orphan nodes
            for entry in self._snapshot:
                for node in (entry["source"], entry["target"]):
                    if node in G and G.degree(node) == 0 and entry.get("node_new"):
                        G.remove_node(node)
            self._snapshot = None
            return removed

    @property
    def last_report(self) -> Optional[IngestReport]:
        return self._last_report

    @property
    def can_rollback(self) -> bool:
        return self._snapshot is not None

    # ------------------------------------------------------------------
    # Triple extraction
    # ------------------------------------------------------------------

    def _extract_triples(self, text: str) -> List[RawTriple]:
        """Run all available extractors, merge results, deduplicate."""
        triples: List[RawTriple] = []

        # Primary: entity-anchored extractor (no dependencies)
        triples.extend(self._extract_anchored(text))

        # Secondary: capitalization-based extractor for new entities
        if self._create_new:
            triples.extend(self._extract_capitalized(text))

        # Tertiary: spaCy SVO extractor (optional)
        if self._use_spacy:
            triples.extend(self._extract_spacy(text))

        # Deduplicate by (subject_lower, verb_lower, object_lower)
        seen: Set[Tuple[str, str, str]] = set()
        unique: List[RawTriple] = []
        for t in triples:
            key = (t.subject_text.lower(), t.verb_text.lower(), t.object_text.lower())
            if key not in seen:
                seen.add(key)
                unique.append(t)

        return unique

    def _extract_anchored(self, text: str) -> List[RawTriple]:
        """
        Anchor extraction: find entity labels (known or capitalized) in
        sentences, then find verbs between consecutive entity mentions.

        Uses both known graph entities (high confidence) and capitalized
        noun phrases (lower confidence) so that (known, new) pairs are
        captured as well as (known, known) pairs.
        """
        if not self._entity_index:
            return []

        triples: List[RawTriple] = []

        for sentence in _split_sentences(text):
            # Merge known entity mentions with capitalized noun candidates
            known   = _find_entity_mentions(sentence, self._entity_index)
            cap     = _find_cap_mentions(sentence, self._entity_index) \
                      if self._create_new else []
            mentions = _merge_mentions(known, cap)

            if len(mentions) < 2:
                continue

            for i in range(len(mentions) - 1):
                m1 = mentions[i]
                m2 = mentions[i + 1]

                # Text between the two entity mentions
                between = sentence[m1["end"]:m2["start"]].strip()
                if not between:
                    continue

                verb, verb_conf = _extract_verb_phrase(between)
                if not verb:
                    continue

                # Confidence: lower when either endpoint is a new (capitalized) entity
                both_known = m1.get("known", False) and m2.get("known", False)
                base = 0.85 if both_known else 0.65

                triples.append(RawTriple(
                    subject_text=m1["text"],
                    verb_text=verb,
                    object_text=m2["text"],
                    sentence=sentence,
                    confidence=base * verb_conf,
                    method="anchored",
                ))

        return triples

    def _extract_capitalized(self, text: str) -> List[RawTriple]:
        """
        Detect Capitalized Noun Sequences as potential new entities.

        Only runs when create_new_entities=True.  Lower confidence than
        anchored extraction since entity identity is uncertain.
        """
        triples: List[RawTriple] = []

        # Pattern: one or more capitalized words (not start of sentence)
        cap_pattern = re.compile(r'(?<!\.\s)(?<![.?!]\s)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)')

        for sentence in _split_sentences(text):
            # Strip leading word (often sentence-start capital, not a proper noun)
            sent_body = sentence[sentence.index(' ') + 1:] if ' ' in sentence else sentence

            caps = list(cap_pattern.finditer(sent_body))
            if len(caps) < 2:
                continue

            for i in range(len(caps) - 1):
                m1 = caps[i]
                m2 = caps[i + 1]

                # Skip if either span is already a known entity (anchored extractor handles those)
                t1 = m1.group(1).strip()
                t2 = m2.group(1).strip()
                if (t1.lower() in (self._entity_index or {}) or
                        t2.lower() in (self._entity_index or {})):
                    continue

                between = sent_body[m1.end():m2.start()].strip()
                verb, verb_conf = _extract_verb_phrase(between)
                if not verb:
                    continue

                triples.append(RawTriple(
                    subject_text=t1,
                    verb_text=verb,
                    object_text=t2,
                    sentence=sentence,
                    confidence=0.60 * verb_conf,
                    method="capitalized",
                ))

        return triples

    def _extract_spacy(self, text: str) -> List[RawTriple]:
        """
        Enhanced extraction using spaCy dependency parsing.

        Returns empty list silently if spaCy is not installed.
        """
        try:
            import spacy  # noqa: F401
        except ImportError:
            return []

        nlp = _get_spacy_model()
        if nlp is None:
            return []

        triples: List[RawTriple] = []
        try:
            doc = nlp(text)
            for sent in doc.sents:
                for token in sent:
                    if token.pos_ not in ("VERB",) or token.dep_ != "ROOT":
                        continue
                    subjects = [t for t in token.lefts
                                if t.dep_ in ("nsubj", "nsubjpass")]
                    objects = [t for t in token.rights
                               if t.dep_ in ("dobj", "attr")]
                    # Follow prep chains
                    for right in token.rights:
                        if right.dep_ == "prep":
                            objects.extend(
                                t for t in right.rights if t.dep_ == "pobj"
                            )
                    for subj in subjects:
                        for obj in objects:
                            s_text = _subtree_text(subj)
                            o_text = _subtree_text(obj)
                            if len(s_text) < _MIN_ENTITY_LEN or len(o_text) < _MIN_ENTITY_LEN:
                                continue
                            triples.append(RawTriple(
                                subject_text=s_text,
                                verb_text=token.lemma_,
                                object_text=o_text,
                                sentence=sent.text,
                                confidence=0.80,
                                method="spacy",
                            ))
        except Exception:
            pass

        return triples

    # ------------------------------------------------------------------
    # Entity linking and relation mapping
    # ------------------------------------------------------------------

    def _link_triples(
        self,
        raw: List[RawTriple],
    ) -> Tuple[List[LinkedTriple], List[Tuple[RawTriple, str]]]:
        """
        Resolve entity texts to graph IDs and map verbs to relation types.

        Returns (linked_triples, [(raw_triple, skip_reason), ...]).
        """
        linked: List[LinkedTriple] = []
        skipped: List[Tuple[RawTriple, str]] = []

        for rt in raw:
            # Map relation
            rel, rel_conf = _map_verb_to_relation(rt.verb_text, self._adapter)
            if not rel:
                skipped.append((rt, "no_relation_match"))
                continue

            # Link source
            src_id, src_label, src_conf, src_new = self._link_entity(
                rt.subject_text, create_if_missing=self._create_new
            )
            if src_id is None:
                skipped.append((rt, "source_entity_unresolvable"))
                continue

            # Link target
            tgt_id, tgt_label, tgt_conf, tgt_new = self._link_entity(
                rt.object_text, create_if_missing=self._create_new
            )
            if tgt_id is None:
                skipped.append((rt, "target_entity_unresolvable"))
                continue

            # Skip self-loops
            if src_id == tgt_id:
                skipped.append((rt, "self_loop"))
                continue

            # Extraction quality × verb mapping quality.
            # Entity-link confidence is already factored into rt.confidence
            # (anchored=0.85 for two known, 0.65 for one new); compounding it
            # again would make valid extractions fall below the threshold.
            combined_conf = rt.confidence * rel_conf

            linked.append(LinkedTriple(
                source_id=src_id,
                source_label=src_label,
                source_new=src_new,
                relation=rel,
                target_id=tgt_id,
                target_label=tgt_label,
                target_new=tgt_new,
                confidence=round(combined_conf, 6),
                sentence=rt.sentence,
                verb_text=rt.verb_text,
            ))

        return linked, skipped

    def _link_entity(
        self,
        text: str,
        create_if_missing: bool = True,
    ) -> Tuple[Optional[str], str, float, bool]:
        """
        Match an entity text span to a graph entity.

        Returns (entity_id, label, confidence, is_new).
        Returns (None, '', 0, False) if unresolvable and create_if_missing=False.
        """
        text_clean = text.strip()
        text_lower = text_clean.lower()
        idx = self._entity_index or {}

        # 1. Exact match
        if text_lower in idx:
            eid = idx[text_lower]
            return eid, text_clean, 1.0, False

        # 2. Substring: graph label contained in text
        for lbl_lower, eid in idx.items():
            if lbl_lower in text_lower and len(lbl_lower) >= _MIN_ENTITY_LEN:
                return eid, lbl_lower, 0.85, False

        # 3. Substring: text contained in graph label
        for lbl_lower, eid in idx.items():
            if text_lower in lbl_lower and len(text_lower) >= _MIN_ENTITY_LEN:
                return eid, lbl_lower, 0.80, False

        # 4. Try adapter's find_entities for fuzzy match
        try:
            results = self._adapter.find_entities(text_clean, top_k=3)
            for ent in results:
                lbl = ent.label.lower()
                if lbl == text_lower or text_lower in lbl or lbl in text_lower:
                    return ent.id, ent.label, 0.75, False
        except Exception:
            pass

        # 5. Create new entity
        if create_if_missing and len(text_clean) >= _MIN_ENTITY_LEN:
            new_id = text_clean.lower().replace(" ", "_")
            # Register in local index for this run
            if idx is not None:
                idx[text_lower] = new_id
            return new_id, text_clean, 0.65, True

        return None, "", 0.0, False

    # ------------------------------------------------------------------
    # Graph mutation
    # ------------------------------------------------------------------

    def _dedup(
        self,
        G,
        triples: List[LinkedTriple],
    ) -> Tuple[List[LinkedTriple], List[LinkedTriple]]:
        """Separate triples into fresh and duplicate sets."""
        fresh: List[LinkedTriple] = []
        dupes: List[LinkedTriple] = []
        seen: Set[Tuple[str, str, str]] = set()

        for t in triples:
            key = (t.source_id, t.relation, t.target_id)
            if key in seen:
                dupes.append(t)
                continue
            seen.add(key)
            # Check graph
            if G.has_edge(t.source_id, t.target_id):
                existing_rel = (
                    G[t.source_id][t.target_id].get("relation_type")
                    or G[t.source_id][t.target_id].get("relation", "")
                ).upper()
                if existing_rel == t.relation:
                    dupes.append(t)
                    continue
            fresh.append(t)

        return fresh, dupes

    def _materialize(
        self,
        G,
        triples: List[LinkedTriple],
        provenance: str,
    ) -> Tuple[int, int]:
        """
        Add triples to graph as edges (and nodes if new).

        Returns (edges_added, nodes_added).
        """
        edges_added = 0
        nodes_added = 0

        for t in triples:
            # Add new nodes if needed
            if t.source_new and t.source_id not in G:
                G.add_node(t.source_id, label=t.source_label)
                nodes_added += 1
            if t.target_new and t.target_id not in G:
                G.add_node(t.target_id, label=t.target_label)
                nodes_added += 1

            G.add_edge(
                t.source_id,
                t.target_id,
                relation_type=t.relation,
                relation=t.relation,
                confidence=t.confidence,
                weight=t.confidence,
                provenance=provenance,
                source_text=t.sentence[:200],  # truncate long sentences
                inferred_at=time.time(),
            )
            edges_added += 1

            self._snapshot.append({
                "source": t.source_id,
                "target": t.target_id,
                "relation": t.relation,
                "source_new": t.source_new,
                "target_new": t.target_new,
                "node_new": t.source_new or t.target_new,
            })

        return edges_added, nodes_added

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_entity_index(self) -> Dict[str, str]:
        """Build lower_label → entity_id dict from current graph."""
        idx: Dict[str, str] = {}
        G = self._adapter.to_networkx()
        for node in G.nodes():
            # Index by node ID
            idx[node.lower()] = node
            # Index by label if different
            try:
                ent = self._adapter.get_entity(node)
                if ent and ent.label and ent.label.lower() != node.lower():
                    idx[ent.label.lower()] = node
            except Exception:
                pass
        return idx

    def _empty_report(self, dry_run: bool, t0: float) -> IngestReport:
        return IngestReport(
            text_length=0, sentences_processed=0,
            entities_found=0, entities_linked=0, entities_new=0,
            triples_extracted=0, triples_accepted=0,
            triples_skipped_duplicate=0, triples_skipped_low_confidence=0,
            edges_added=0, nodes_added=0,
            added_triples=[], skipped_triples=[],
            duration_seconds=time.time() - t0,
            provenance="text_ingest:empty",
            dry_run=dry_run,
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> List[str]:
    """Split text into sentences using punctuation heuristic."""
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in raw if s.strip()]


def _find_entity_mentions(
    sentence: str,
    entity_index: Dict[str, str],
) -> List[Dict]:
    """
    Find all known entity mentions in *sentence*, in order of position.

    Returns list of {text, start, end, entity_id, known=True} dicts.
    """
    mentions: List[Dict] = []
    sent_lower = sentence.lower()

    # Sort by length descending so longer labels match first
    sorted_labels = sorted(entity_index.keys(), key=len, reverse=True)

    covered: Set[int] = set()

    for lbl in sorted_labels:
        if len(lbl) < _MIN_ENTITY_LEN:
            continue
        pos = 0
        while True:
            idx = sent_lower.find(lbl, pos)
            if idx == -1:
                break
            end = idx + len(lbl)
            before_ok = idx == 0 or not sent_lower[idx - 1].isalpha()
            after_ok  = end == len(sent_lower) or not sent_lower[end].isalpha()
            if before_ok and after_ok:
                span = set(range(idx, end))
                if not span & covered:
                    covered |= span
                    mentions.append({
                        "text": sentence[idx:end],
                        "start": idx,
                        "end": end,
                        "entity_id": entity_index[lbl],
                        "known": True,
                    })
            pos = end

    mentions.sort(key=lambda m: m["start"])
    return mentions


def _find_cap_mentions(
    sentence: str,
    entity_index: Dict[str, str],
) -> List[Dict]:
    """
    Find capitalized noun sequences that are NOT already known graph entities.

    Returns list of {text, start, end, entity_id, known=False} dicts.
    """
    cap_pattern = re.compile(r'\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*)\b')
    mentions: List[Dict] = []
    for m in cap_pattern.finditer(sentence):
        text = m.group(1)
        if text.lower() in entity_index:
            continue   # already a known entity — handled by _find_entity_mentions
        if len(text) < _MIN_ENTITY_LEN:
            continue
        new_id = text.lower().replace(" ", "_")
        mentions.append({
            "text": text,
            "start": m.start(),
            "end": m.end(),
            "entity_id": new_id,
            "known": False,
        })
    return mentions


def _merge_mentions(known: List[Dict], cap: List[Dict]) -> List[Dict]:
    """
    Merge known and capitalized entity mentions, giving known mentions priority.
    Capitalized mentions that overlap a known mention are dropped.
    Result is sorted by start position.
    """
    covered: Set[int] = set()
    for m in known:
        covered.update(range(m["start"], m["end"]))

    fresh_caps = [
        m for m in cap
        if not set(range(m["start"], m["end"])) & covered
    ]

    merged = known + fresh_caps
    merged.sort(key=lambda m: m["start"])
    return merged


def _extract_verb_phrase(between: str) -> Tuple[str, float]:
    """
    Extract the main verb (lemma or surface form) from the text between
    two entity mentions.

    Returns (verb_str, confidence) or ("", 0.0) if no verb found.
    """
    if not between:
        return "", 0.0

    tokens = between.lower().split()
    # Strip leading/trailing punctuation from tokens
    tokens = [t.strip(".,;:\"'()[]") for t in tokens]
    tokens = [t for t in tokens if t]

    # Skip auxiliary verbs and prepositions to find the main verb
    for token in tokens:
        if token in _AUX_VERBS or token in _PREPS:
            continue
        # Check if this token matches a known verb stem
        for stem in sorted(_VERB_MAP.keys(), key=len, reverse=True):
            if token == stem or token.startswith(stem[:5]):
                _, conf = _VERB_MAP[stem]
                return token, conf
        # Also accept any token that looks like a verb (ends in common suffixes)
        if (token.endswith(("ed", "es", "ing", "tion", "ions")) and
                len(token) >= 4 and token not in _PREPS):
            return token, 0.55

    return "", 0.0


def _map_verb_to_relation(
    verb: str,
    adapter=None,
) -> Tuple[str, float]:
    """
    Map a verb string to a graph relation type.

    Checks:
    1. Direct lookup in _VERB_MAP by stem prefix
    2. Graph's own relation vocabulary (if adapter provided)
    3. Returns ("", 0.0) if no match
    """
    v = verb.lower().strip()

    # Direct stem match
    for stem in sorted(_VERB_MAP.keys(), key=len, reverse=True):
        if v == stem or v.startswith(stem[:min(len(stem), 6)]):
            rel, conf = _VERB_MAP[stem]
            return rel, conf

    # Check graph vocabulary for novel relation types
    if adapter is not None:
        try:
            G = adapter.to_networkx()
            for _, _, data in G.edges(data=True):
                rt = (data.get("relation_type") or data.get("relation", "")).upper()
                if rt and v in rt.lower():
                    return rt, 0.65
        except Exception:
            pass

    return "", 0.0


# spaCy model singleton (loaded once, cached)
_spacy_nlp = None
_spacy_tried = False


def _get_spacy_model():
    """Load spaCy model lazily.  Returns None if unavailable."""
    global _spacy_nlp, _spacy_tried
    if _spacy_tried:
        return _spacy_nlp
    _spacy_tried = True
    try:
        import spacy
        for model in ("en_core_web_sm", "en_core_web_md", "en_core_web_lg"):
            try:
                _spacy_nlp = spacy.load(model)
                return _spacy_nlp
            except OSError:
                continue
    except ImportError:
        pass
    return None


def _subtree_text(token) -> str:
    """Get the full text of a spaCy token's subtree, cleaned up."""
    return " ".join(t.text for t in token.subtree if not t.is_punct).strip()
