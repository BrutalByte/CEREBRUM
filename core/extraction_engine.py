"""
ExtractionEngine — Phase 21: Unstructured Text → Knowledge Graph.

Bridges the gap between LLM-scale implicit knowledge and CEREBRUM's explicit
graph representation by converting free text into confidence-scored, provenance-
tagged (subject, relation, object) triples ready for IngestionPipeline.

Three backends — composable and combinable
------------------------------------------

``"local"``  (default — zero dependencies beyond spaCy)
    Extends TextIngestor with full spaCy NER + dependency-tree SVO extraction,
    entity linking against the existing graph, and simple pronoun coreference.
    Fast.  Best for structured/semi-structured text with known entities.

``"transformer"``  (pip install transformers torch)
    Uses REBEL (Babelscape/rebel-large) — a seq2seq model trained end-to-end
    on Wikipedia relation extraction.  Extracts triples directly from arbitrary
    sentences.  No relation schema required.  Medium quality, no API cost.

``"llm"``  (requires LLM Bridge + API key)
    Sends text chunks to any LLM via the LLM Bridge with a structured JSON
    extraction prompt.  Highest quality — handles implicit, contextual, and
    multi-sentence relationships that pattern-based systems miss.

All backends return ``List[ExtractedTriple]``.  Use ``ExtractionPipeline`` for
full document processing with chunking, deduplication, and batch ingest.

Usage
-----
    from adapters.networkx_adapter import NetworkXAdapter
    from core.extraction_engine import ExtractionEngine, ExtractionConfig

    adapter = NetworkXAdapter.from_csv("kb.csv")
    engine  = ExtractionEngine(adapter)

    triples = engine.extract("Newton discovered gravity in 1666.")
    for t in triples:
        print(t.subject, "→", t.predicate, "→", t.object, f"[{t.confidence:.2f}]")

    # Ingest directly into the graph
    report = engine.ingest_text("Aspirin inhibits COX-2 which promotes inflammation.")
    print(f"Added {report.edges_added} edges, {report.entities_added} new entities.")

    # LLM-assisted (highest quality)
    from llm_bridge.adapters import AnthropicAdapter
    engine_llm = ExtractionEngine(
        adapter,
        config=ExtractionConfig(backend="llm"),
        llm_fn=AnthropicAdapter(),
    )
    report = engine_llm.ingest_file("paper.txt")

    # Ensemble: combine all three backends
    engine_all = ExtractionEngine(
        adapter,
        config=ExtractionConfig(backend="ensemble"),
        llm_fn=AnthropicAdapter(),
    )
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

log = logging.getLogger("cerebrum.extraction")

# ---------------------------------------------------------------------------
# Optional heavy imports — graceful degradation
# ---------------------------------------------------------------------------

try:
    import spacy as _spacy
    _SPACY_AVAILABLE = True
except ImportError:
    _spacy = None  # type: ignore
    _SPACY_AVAILABLE = False

try:
    from transformers import pipeline as _hf_pipeline, Text2TextGenerationPipeline
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _hf_pipeline = None
    Text2TextGenerationPipeline = Any # type: ignore
    _TRANSFORMERS_AVAILABLE = False


# ---------------------------------------------------------------------------
# ExtractedTriple — the output unit of all backends
# ---------------------------------------------------------------------------

@dataclass
class ExtractedTriple:
    """
    A single extracted (subject, predicate, object) triple with provenance.

    Attributes
    ----------
    subject           : Normalised subject entity string.
    predicate         : Relation type string (e.g. "INHIBITS", "FOUNDED").
    object            : Normalised object entity string.
    confidence        : Extraction confidence in [0.0, 1.0].
    source_text       : The sentence or span this triple was extracted from.
    source_doc        : Document identifier (filename, URL, or hash).
    extraction_method : "local_verb", "local_spacy", "transformer", "llm", "ensemble".
    subject_type      : NER type of subject (PERSON, ORG, LOC, CHEM, GENE …).
    object_type       : NER type of object.
    subject_linked    : Whether subject was linked to an existing graph entity.
    object_linked     : Whether object was linked to an existing graph entity.
    """
    subject:           str
    predicate:         str
    object:            str
    confidence:        float
    source_text:       str        = ""
    source_doc:        str        = ""
    extraction_method: str        = "local_verb"
    subject_type:      str        = "UNKNOWN"
    object_type:       str        = "UNKNOWN"
    subject_linked:    bool       = False
    object_linked:     bool       = False

    def __post_init__(self) -> None:
        self.confidence = max(0.0, min(1.0, self.confidence))


# ---------------------------------------------------------------------------
# ExtractionConfig
# ---------------------------------------------------------------------------

@dataclass
class ExtractionConfig:
    """
    Configuration for ExtractionEngine.

    Attributes
    ----------
    backend             : "local" | "transformer" | "llm" | "ensemble".
    min_confidence      : Discard triples below this confidence.
    chunk_size          : Characters per text chunk for document processing.
    chunk_overlap       : Character overlap between consecutive chunks.
    max_workers         : Thread pool size for parallel chunk processing.
    link_to_graph       : Attempt to link extracted entities to existing graph nodes.
    link_threshold      : Minimum fuzzy-match similarity for entity linking.
    new_entity_threshold: Confidence required to add a completely new entity.
    relation_schema     : Optional {raw_verb → canonical_relation} override map.
    coref_resolution    : Attempt simple pronoun coreference resolution.
    rebel_model         : HuggingFace model ID for the transformer backend.
    llm_extract_prompt  : Custom extraction prompt (None = use built-in).
    dedup_triples       : Remove duplicate (subject, predicate, object) within a doc.
    namespace           : Entity namespace prefix ("" = no prefix).
    """
    backend:              str                     = "local"
    min_confidence:       float                   = 0.55
    chunk_size:           int                     = 1000
    chunk_overlap:        int                     = 100
    max_workers:          int                     = 4
    link_to_graph:        bool                    = True
    link_threshold:       float                   = 0.75
    new_entity_threshold: float                   = 0.70
    relation_schema:      Optional[Dict[str,str]] = None
    coref_resolution:     bool                    = True
    rebel_model:          str                     = "Babelscape/rebel-large"
    llm_extract_prompt:   Optional[str]           = None
    dedup_triples:        bool                    = True
    namespace:            str                     = ""


# ---------------------------------------------------------------------------
# ExtractionReport
# ---------------------------------------------------------------------------

@dataclass
class ExtractionReport:
    """Summary returned by ingest_text() and ingest_file()."""
    triples_extracted:   int   = 0
    triples_accepted:    int   = 0
    triples_rejected:    int   = 0
    edges_added:         int   = 0
    entities_added:      int   = 0
    sentences_processed: int   = 0
    chunks_processed:    int   = 0
    wall_ms:             float = 0.0
    source_doc:          str   = ""
    triples:             List[ExtractedTriple] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"ExtractionReport({self.source_doc}): "
            f"{self.triples_extracted} extracted → "
            f"{self.triples_accepted} accepted → "
            f"{self.edges_added} edges added, "
            f"{self.entities_added} new entities | "
            f"{self.wall_ms:.1f}ms"
        )


# ---------------------------------------------------------------------------
# ExtractionEngine
# ---------------------------------------------------------------------------

class ExtractionEngine:
    """
    Converts unstructured text into Knowledge Graph triples.

    Parameters
    ----------
    adapter : GraphAdapter — the graph to link against and optionally ingest into.
    config  : ExtractionConfig — backend and quality settings.
    llm_fn  : Any callable (str) → str used by the "llm" backend.
              Pass an adapter from llm_bridge.adapters (AnthropicAdapter, etc.).
    """

    def __init__(
        self,
        adapter,                              # GraphAdapter
        config: Optional[ExtractionConfig]   = None,
        llm_fn: Optional[Callable[[str],str]] = None,
    ):
        self._adapter  = adapter
        self._config   = config or ExtractionConfig()
        self._llm_fn   = llm_fn
        self._nlp      = None        # spaCy model — lazy-loaded
        self._rebel    = None        # REBEL pipeline — lazy-loaded
        self._lock     = threading.Lock()

        # Build known-entity set for entity linking
        self._entity_set: Set[str] = set()
        self._refresh_entity_set()

    # ------------------------------------------------------------------
    # Primary public API
    # ------------------------------------------------------------------

    def extract(self, text: str, doc_id: str = "") -> List[ExtractedTriple]:
        """
        Extract triples from a single text string.

        Returns List[ExtractedTriple] filtered to min_confidence.
        """
        cfg = self._config
        if not text or not text.strip():
            return []

        if cfg.coref_resolution:
            text = self._resolve_coreference(text)

        backend = cfg.backend.lower()
        if backend == "local":
            triples = self._extract_local(text, doc_id)
        elif backend == "transformer":
            triples = self._extract_transformer(text, doc_id)
        elif backend == "llm":
            triples = self._extract_llm(text, doc_id)
        elif backend == "ensemble":
            triples = self._extract_ensemble(text, doc_id)
        else:
            raise ValueError(f"Unknown backend: {backend!r}. Use 'local', 'transformer', 'llm', or 'ensemble'.")

        # Link entities to graph
        if cfg.link_to_graph:
            triples = [self._link_entities(t) for t in triples]

        # Apply relation schema override
        if cfg.relation_schema:
            triples = [self._apply_schema(t, cfg.relation_schema) for t in triples]

        # Filter by confidence and deduplicate
        triples = [t for t in triples if t.confidence >= cfg.min_confidence]
        if cfg.dedup_triples:
            triples = _dedup(triples)

        return triples

    def ingest_text(
        self,
        text: str,
        doc_id: str = "",
        dry_run: bool = False,
    ) -> ExtractionReport:
        """
        Extract triples and optionally add them to the graph.

        Parameters
        ----------
        text    : Free-text input.
        doc_id  : Document identifier for provenance.
        dry_run : If True, extract but do not modify the graph.

        Returns
        -------
        ExtractionReport with full statistics.
        """
        t0 = time.perf_counter()
        if not doc_id:
            doc_id = "text:" + hashlib.sha256(text.encode()).hexdigest()[:8]

        triples = self.extract(text, doc_id=doc_id)
        report  = self._build_report(triples, doc_id, text)

        if not dry_run:
            self._commit_triples(triples, report)

        report.wall_ms = (time.perf_counter() - t0) * 1000
        return report

    def ingest_file(
        self,
        path: str,
        dry_run: bool = False,
        encoding: str = "utf-8",
    ) -> ExtractionReport:
        """
        Extract triples from a text file (plain text, .txt, .md).

        Large files are automatically chunked per ExtractionConfig.chunk_size.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")

        text   = p.read_text(encoding=encoding, errors="replace")
        doc_id = p.name
        return self._ingest_chunked(text, doc_id, dry_run)

    def ingest_documents(
        self,
        paths: List[str],
        dry_run: bool = False,
    ) -> List[ExtractionReport]:
        """Ingest multiple files in parallel. Returns one report per file."""
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=self._config.max_workers) as ex:
            futures = {ex.submit(self.ingest_file, p, dry_run): p for p in paths}
            return [f.result() for f in concurrent.futures.as_completed(futures)]

    def rollback(self, report: ExtractionReport) -> int:
        """Remove all edges that were added during an ingest_text() call."""
        removed = 0
        G = self._adapter.to_networkx()
        for triple in report.triples:
            try:
                if G.has_edge(triple.subject, triple.object):
                    G.remove_edge(triple.subject, triple.object)
                    removed += 1
            except Exception as exc:
                log.debug("ExtractionEngine.rollback error: %s", exc)
        log.info("ExtractionEngine.rollback: removed %d edges.", removed)
        return removed

    # ------------------------------------------------------------------
    # Backends
    # ------------------------------------------------------------------

    def _extract_local(self, text: str, doc_id: str) -> List[ExtractedTriple]:
        """
        Local extraction: entity anchoring + spaCy SVO (if available).
        Extends and delegates to TextIngestor for verb-map extraction.
        """
        from core.text_ingestor import TextIngestor

        ingestor = TextIngestor(self._adapter, min_confidence=self._config.min_confidence)
        report   = ingestor.ingest_text(text, dry_run=True)

        triples: List[ExtractedTriple] = []
        for src, rel, dst, conf in report.added_triples:
            triples.append(ExtractedTriple(
                subject=src,
                predicate=rel,
                object=dst,
                confidence=conf,
                source_text=text[:200] + "...", # Best effort as sentence isn't in report
                source_doc=doc_id,
                extraction_method="local_verb",
            ))

        # Augment with spaCy if available
        if _SPACY_AVAILABLE:
            triples.extend(self._extract_spacy_svo(text, doc_id))

        return triples

    def _extract_spacy_svo(self, text: str, doc_id: str) -> List[ExtractedTriple]:
        """spaCy dependency-tree SVO extraction with NER typing."""
        nlp = self._get_spacy()
        if nlp is None:
            return []

        doc      = nlp(text[:10_000])   # cap at 10 KB to avoid memory issues
        triples: List[ExtractedTriple] = []

        for sent in doc.sents:
            for token in sent:
                if token.dep_ not in ("ROOT",) and token.pos_ not in ("VERB",):
                    continue

                # Find subject(s)
                subjects   = [c for c in token.children if c.dep_ in ("nsubj", "nsubjpass")]
                # Find object(s)
                objects    = [c for c in token.children if c.dep_ in ("dobj", "pobj", "attr", "nmod")]

                if not subjects or not objects:
                    continue

                verb_lemma = token.lemma_.lower()
                rel        = self._verb_to_relation(verb_lemma)

                for subj in subjects:
                    for obj in objects:
                        subj_np  = _get_noun_phrase(subj, doc)
                        obj_np   = _get_noun_phrase(obj, doc)
                        if not subj_np or not obj_np or subj_np == obj_np:
                            continue

                        subj_ent = _find_ent(subj, doc)
                        obj_ent  = _find_ent(obj, doc)

                        triples.append(ExtractedTriple(
                            subject=subj_np,
                            predicate=rel,
                            object=obj_np,
                            confidence=0.70,
                            source_text=sent.text,
                            source_doc=doc_id,
                            extraction_method="local_spacy",
                            subject_type=subj_ent.label_ if subj_ent else "UNKNOWN",
                            object_type=obj_ent.label_ if obj_ent else "UNKNOWN",
                        ))

        return triples

    def _extract_transformer(self, text: str, doc_id: str) -> List[ExtractedTriple]:
        """
        REBEL end-to-end relation extraction.

        Model: Babelscape/rebel-large (config.rebel_model).
        Outputs triplets in a special linearised format which this method parses.
        """
        if not _TRANSFORMERS_AVAILABLE:
            log.warning("transformers not installed — falling back to local backend.")
            return self._extract_local(text, doc_id)

        rebel = self._get_rebel()
        if rebel is None:
            return self._extract_local(text, doc_id)

        triples: List[ExtractedTriple] = []
        chunks = _sentence_chunks(text, max_chars=512)

        for chunk in chunks:
            try:
                outputs = rebel(chunk, return_tensors=True, return_text=False)
                for output in outputs:
                    decoded = rebel.tokenizer.batch_decode(
                        [output["generated_token_ids"]], skip_special_tokens=False
                    )
                    parsed  = _parse_rebel_output(decoded[0] if decoded else "")
                    for subj, rel, obj in parsed:
                        triples.append(ExtractedTriple(
                            subject=subj,
                            predicate=rel.upper().replace(" ", "_"),
                            object=obj,
                            confidence=0.82,
                            source_text=chunk[:200],
                            source_doc=doc_id,
                            extraction_method="transformer",
                        ))
            except Exception as exc:
                log.debug("REBEL extraction error on chunk: %s", exc)

        return triples

    def _extract_llm(self, text: str, doc_id: str) -> List[ExtractedTriple]:
        """
        LLM-assisted structured extraction.

        Sends text + extraction prompt to llm_fn, parses JSON response.
        Falls back to local backend if llm_fn is not set.
        """
        if self._llm_fn is None:
            log.warning("ExtractionEngine: llm_fn not set — falling back to local backend.")
            return self._extract_local(text, doc_id)

        prompt  = self._build_llm_prompt(text)
        triples: List[ExtractedTriple] = []

        try:
            response = self._llm_fn(prompt)
            triples  = self._parse_llm_response(response, text, doc_id)
        except Exception as exc:
            log.warning("LLM extraction failed: %s — falling back to local.", exc)
            return self._extract_local(text, doc_id)

        return triples

    def _extract_ensemble(self, text: str, doc_id: str) -> List[ExtractedTriple]:
        """
        Combine all available backends and merge results.

        Triples found by multiple backends receive a confidence boost.
        """
        all_triples: List[ExtractedTriple] = []

        all_triples.extend(self._extract_local(text, doc_id))

        if _TRANSFORMERS_AVAILABLE and self._get_rebel() is not None:
            all_triples.extend(self._extract_transformer(text, doc_id))

        if self._llm_fn is not None:
            all_triples.extend(self._extract_llm(text, doc_id))

        return _merge_ensemble(all_triples)

    # ------------------------------------------------------------------
    # LLM prompt building and response parsing
    # ------------------------------------------------------------------

    _DEFAULT_EXTRACTION_PROMPT = """Extract all factual relationships from the text below.
Return a JSON array of objects with exactly these fields:
  "subject"    : the entity performing or being described
  "predicate"  : the relationship type in UPPER_SNAKE_CASE (e.g. INHIBITS, FOUNDED, CAUSES)
  "object"     : the entity receiving the relationship
  "confidence" : your confidence in this extraction, 0.0 to 1.0
  "subject_type": entity type (PERSON, ORG, CHEMICAL, GENE, LOCATION, CONCEPT, OTHER)
  "object_type" : entity type (same options)

Rules:
- Only include relationships explicitly stated or strongly implied in the text.
- Use canonical entity names (full names, not pronouns).
- Output ONLY the JSON array, no other text.

Text:
---
{text}
---
JSON:"""

    def _build_llm_prompt(self, text: str) -> str:
        template = self._config.llm_extract_prompt or self._DEFAULT_EXTRACTION_PROMPT
        return template.format(text=text[:4000])

    def _parse_llm_response(
        self,
        response: str,
        source_text: str,
        doc_id: str,
    ) -> List[ExtractedTriple]:
        """Parse JSON array from LLM response into ExtractedTriple objects."""
        triples: List[ExtractedTriple] = []
        response = response.strip()

        # Extract JSON array even if wrapped in markdown fences
        match = re.search(r"\[.*\]", response, re.DOTALL)
        if not match:
            log.debug("LLM response contained no JSON array.")
            return triples

        try:
            items: List[Dict[str, Any]] = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            log.debug("LLM JSON parse error: %s", exc)
            return triples

        for item in items:
            if not isinstance(item, dict):
                continue
            subj = str(item.get("subject", "")).strip()
            pred = str(item.get("predicate", "RELATED_TO")).strip().upper().replace(" ", "_")
            obj  = str(item.get("object", "")).strip()
            if not subj or not obj:
                continue
            triples.append(ExtractedTriple(
                subject=subj,
                predicate=pred,
                object=obj,
                confidence=float(item.get("confidence", 0.75)),
                source_text=source_text[:200],
                source_doc=doc_id,
                extraction_method="llm",
                subject_type=str(item.get("subject_type", "UNKNOWN")),
                object_type=str(item.get("object_type", "UNKNOWN")),
            ))

        return triples

    # ------------------------------------------------------------------
    # Entity linking
    # ------------------------------------------------------------------

    def _refresh_entity_set(self) -> None:
        """Rebuild the set of known entity IDs from the adapter."""
        try:
            G = self._adapter.to_networkx()
            with self._lock:
                self._entity_set = {str(n).lower() for n in G.nodes()}
        except Exception:
            pass

    def _link_entities(self, triple: ExtractedTriple) -> ExtractedTriple:
        """
        Attempt fuzzy linking of subject/object to existing graph entities.
        Updates subject/object strings to canonical graph IDs when a match
        exceeds config.link_threshold.
        """
        cfg       = self._config
        subj_norm = triple.subject.lower().strip()
        obj_norm  = triple.object.lower().strip()

        with self._lock:
            eset = self._entity_set

        # Exact match
        if subj_norm in eset:
            triple.subject_linked = True
        else:
            best = _fuzzy_best(subj_norm, eset, cfg.link_threshold)
            if best:
                triple.subject        = best
                triple.subject_linked = True

        if obj_norm in eset:
            triple.object_linked = True
        else:
            best = _fuzzy_best(obj_norm, eset, cfg.link_threshold)
            if best:
                triple.object        = best
                triple.object_linked = True

        # Namespace prefix
        if cfg.namespace:
            if not triple.subject_linked:
                triple.subject = f"{cfg.namespace}:{triple.subject}"
            if not triple.object_linked:
                triple.object = f"{cfg.namespace}:{triple.object}"

        return triple

    def _apply_schema(
        self,
        triple: ExtractedTriple,
        schema: Dict[str, str],
    ) -> ExtractedTriple:
        """Map raw predicate to canonical relation via schema."""
        triple.predicate = schema.get(triple.predicate, triple.predicate)
        return triple

    # ------------------------------------------------------------------
    # Coreference resolution (simple rule-based)
    # ------------------------------------------------------------------

    def _resolve_coreference(self, text: str) -> str:
        """
        Simple rule-based pronoun coreference resolution.

        Replaces pronouns with their most recent named antecedent.
        Handles: he/she/it/they/his/her/its/their + subject/object positions.

        For production quality, use neuralcoref or spaCy's coref component.
        """
        # Map of pronouns to their gender/number category
        pronoun_map = {
            "he": "M_SG", "him": "M_SG", "his": "M_SG",
            "she": "F_SG", "her": "F_SG", "hers": "F_SG",
            "it": "N_SG", "its": "N_SG",
            "they": "PL",  "them": "PL", "their": "PL", "theirs": "PL",
        }

        # Collect candidate antecedents from NER (if spaCy available)
        antecedents: Dict[str, str] = {}  # {category: last_entity}

        if _SPACY_AVAILABLE:
            nlp = self._get_spacy()
            if nlp:
                doc = nlp(text[:5000])
                for ent in doc.ents:
                    if ent.label_ in ("PERSON",):
                        antecedents["M_SG"] = ent.text
                        antecedents["F_SG"] = ent.text
                    elif ent.label_ in ("ORG", "GPE", "LOC", "PRODUCT"):
                        antecedents["N_SG"] = ent.text
                        antecedents["PL"]   = ent.text

        if not antecedents:
            return text

        # Simple token-by-token replacement
        tokens = text.split()
        result = []
        for tok in tokens:
            clean = tok.lower().rstrip(".,;:!?)")
            cat   = pronoun_map.get(clean)
            if cat and cat in antecedents:
                # Preserve trailing punctuation
                suffix = tok[len(clean):]
                result.append(antecedents[cat] + suffix)
            else:
                result.append(tok)
        return " ".join(result)

    # ------------------------------------------------------------------
    # Verb → relation mapping
    # ------------------------------------------------------------------

    def _verb_to_relation(self, verb_lemma: str) -> str:
        """Map a verb lemma to a canonical relation type."""
        from core.text_ingestor import _VERB_MAP
        key = verb_lemma[:6].lower()
        if key in _VERB_MAP:
            return _VERB_MAP[key][0]
        # Default: UPPER_SNAKE of the lemma
        return verb_lemma.upper().replace(" ", "_")

    # ------------------------------------------------------------------
    # Chunked document processing
    # ------------------------------------------------------------------

    def _ingest_chunked(self, text: str, doc_id: str, dry_run: bool) -> ExtractionReport:
        """Split large text into overlapping chunks and aggregate reports."""
        cfg         = self._config
        chunks      = _chunk_text(text, cfg.chunk_size, cfg.chunk_overlap)
        all_triples: List[ExtractedTriple] = []

        import concurrent.futures
        def _process(chunk: str) -> List[ExtractedTriple]:
            return self.extract(chunk, doc_id=doc_id)

        with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.max_workers) as ex:
            for result in ex.map(_process, chunks):
                all_triples.extend(result)

        if cfg.dedup_triples:
            all_triples = _dedup(all_triples)

        report = self._build_report(all_triples, doc_id, text)
        report.chunks_processed = len(chunks)

        if not dry_run:
            self._commit_triples(all_triples, report)

        return report

    # ------------------------------------------------------------------
    # Graph commit
    # ------------------------------------------------------------------

    def _commit_triples(
        self,
        triples: List[ExtractedTriple],
        report: ExtractionReport,
    ) -> None:
        """Commit accepted triples to the graph via IngestionPipeline."""
        from core.thalamus import IngestionPipeline

        # IngestionPipeline handles normalization and confidence clamping.
        # We don't pass namespace here because _link_entities (called in extract())
        # has already applied it to the triple subject/object.
        pipeline = IngestionPipeline()
        G = self._adapter.to_networkx()
        known_before = set(G.nodes())

        for triple in triples:
            if triple.confidence < self._config.min_confidence:
                continue
            try:
                processed = pipeline.process(
                    source=triple.subject,
                    target=triple.object,
                    relation=triple.predicate,
                    metadata={
                        "confidence": triple.confidence,
                        "provenance": f"extraction:{triple.extraction_method}",
                        "source_text": triple.source_text[:500],
                    }
                )
                G.add_edge(
                    processed.source,
                    processed.target,
                    relation=processed.relation,
                    confidence=processed.confidence,
                    provenance=processed.provenance,
                    weight=processed.weight,
                    **processed.properties,
                )
                report.edges_added += 1
            except Exception as exc:
                log.debug("ExtractionEngine commit error: %s", exc)

        known_after = set(G.nodes())
        report.entities_added = len(known_after - known_before)
        self._refresh_entity_set()

    # ------------------------------------------------------------------
    # Report builder
    # ------------------------------------------------------------------

    def _build_report(
        self,
        triples: List[ExtractedTriple],
        doc_id: str,
        text: str,
    ) -> ExtractionReport:
        accepted = [t for t in triples if t.confidence >= self._config.min_confidence]
        return ExtractionReport(
            triples_extracted=len(triples),
            triples_accepted=len(accepted),
            triples_rejected=len(triples) - len(accepted),
            sentences_processed=len(re.split(r"[.!?]+", text)),
            source_doc=doc_id,
            triples=accepted,
        )

    # ------------------------------------------------------------------
    # Lazy model loaders
    # ------------------------------------------------------------------

    def _get_spacy(self):
        if self._nlp is not None:
            return self._nlp
        if not _SPACY_AVAILABLE:
            return None
        with self._lock:
            if self._nlp is None:
                for model in ("en_core_web_md", "en_core_web_sm", "en_core_web_lg"):
                    try:
                        self._nlp = _spacy.load(model)
                        log.info("ExtractionEngine: loaded spaCy model '%s'.", model)
                        break
                    except OSError:
                        continue
                if self._nlp is None:
                    log.warning(
                        "ExtractionEngine: no spaCy model found. "
                        "Run: python -m spacy download en_core_web_sm"
                    )
        return self._nlp

    def _get_rebel(self):
        if self._rebel is not None:
            return self._rebel
        if not _TRANSFORMERS_AVAILABLE:
            return None
        with self._lock:
            if self._rebel is None:
                try:
                    self._rebel = _hf_pipeline(
                        "text2text-generation",
                        model=self._config.rebel_model,
                        tokenizer=self._config.rebel_model,
                        pipeline_class=Text2TextGenerationPipeline, # type: ignore
                    )
                    log.info("ExtractionEngine: loaded REBEL model '%s'.", self._config.rebel_model)
                except Exception as exc:
                    log.warning("ExtractionEngine: could not load REBEL: %s", exc)
                return self._rebel
# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _sentence_chunks(text: str, max_chars: int = 512) -> List[str]:
    """Split text into sentence-bounded chunks of at most max_chars."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks    = []
    buf       = ""
    for sent in sentences:
        if len(buf) + len(sent) > max_chars and buf:
            chunks.append(buf.strip())
            buf = sent
        else:
            buf += (" " if buf else "") + sent
    if buf.strip():
        chunks.append(buf.strip())
    return chunks


def _chunk_text(text: str, size: int, overlap: int) -> List[str]:
    """Sliding-window text chunker."""
    chunks = []
    step   = max(1, size - overlap)
    for start in range(0, len(text), step):
        chunk = text[start:start + size]
        if chunk.strip():
            chunks.append(chunk)
        if start + size >= len(text):
            break
    return chunks or [text]


def _dedup(triples: List[ExtractedTriple]) -> List[ExtractedTriple]:
    """Remove duplicate (subject, predicate, object), keeping highest confidence."""
    seen: Dict[Tuple[str,str,str], ExtractedTriple] = {}
    for t in triples:
        key = (t.subject.lower(), t.predicate.lower(), t.object.lower())
        if key not in seen or t.confidence > seen[key].confidence:
            seen[key] = t
    return list(seen.values())


def _merge_ensemble(triples: List[ExtractedTriple]) -> List[ExtractedTriple]:
    """
    Merge triples from multiple backends. Triples confirmed by ≥2 backends
    receive a +0.10 confidence boost (capped at 1.0).
    """
    key_to_methods: Dict[Tuple[str,str,str], Set[str]] = {}
    key_to_triple:  Dict[Tuple[str,str,str], ExtractedTriple] = {}

    for t in triples:
        key = (t.subject.lower(), t.predicate.lower(), t.object.lower())
        key_to_methods.setdefault(key, set()).add(t.extraction_method)
        if key not in key_to_triple or t.confidence > key_to_triple[key].confidence:
            key_to_triple[key] = t

    result = []
    for key, triple in key_to_triple.items():
        n_methods = len(key_to_methods[key])
        if n_methods >= 2:
            triple.confidence = min(1.0, triple.confidence + 0.10)
            triple.extraction_method = "ensemble"
        result.append(triple)

    return result


def _fuzzy_best(query: str, candidates: Set[str], threshold: float) -> Optional[str]:
    """
    Simple character-level similarity (Jaccard on bigrams).
    Returns the best match above threshold, or None.
    """
    if not candidates:
        return None
    q_bg = _bigrams(query)
    if not q_bg:
        return None
    best_score = 0.0
    best_cand  = None
    for cand in candidates:
        c_bg  = _bigrams(cand)
        inter = len(q_bg & c_bg)
        union = len(q_bg | c_bg)
        score = inter / union if union > 0 else 0.0
        if score > best_score:
            best_score = score
            best_cand  = cand
    return best_cand if best_score >= threshold else None


def _bigrams(s: str) -> Set[str]:
    return {s[i:i+2] for i in range(len(s) - 1)}


def _get_noun_phrase(token, doc) -> str:
    """Expand a token to its full noun phrase span."""
    start = token.i
    end   = token.i + 1
    for child in token.subtree:
        if child.i < start:
            start = child.i
        if child.i + 1 > end:
            end = child.i + 1
    span = doc[start:end]
    return span.text.strip()


def _find_ent(token, doc):
    """Find the NER entity that covers a token."""
    for ent in doc.ents:
        if ent.start <= token.i < ent.end:
            return ent
    return None


def _parse_rebel_output(text: str) -> List[Tuple[str, str, str]]:
    """
    Parse REBEL's linearised triple output.
    Format: <triplet> subject <subj> relation <obj> object
    """
    triples = []
    triplets = [t.strip() for t in text.split("<triplet>") if t.strip()]
    for triplet in triplets:
        try:
            parts = re.split(r"<subj>|<obj>", triplet)
            if len(parts) >= 3:
                subj = parts[0].strip()
                rel  = parts[1].strip()
                obj  = parts[2].strip()
                if subj and rel and obj:
                    triples.append((subj, rel, obj))
        except Exception:
            continue
    return triples
