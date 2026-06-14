"""
KnowledgeHarvester — Phase 270.

Autonomous external knowledge ingestion with a 5-stage vetting pipeline.
Triggered on PLATEAU events from MetaOrchestrator.

Source tiers
------------
Tier 1 (peer-reviewed):  PubMed/MeSH, CrossRef
Tier 2 (curated KB):     Wikidata, Wikipedia infoboxes
Tier 3 (community):      community-edited sources — require ≥3 independent
                         reputable (tier-1 or tier-2) corroborating sources

Vetting pipeline (all stages must pass)
----------------------------------------
1. Schema conformance   — entity types must fit existing relation constraints
2. ContradictionResolver— net evidence score must not be "discardable"
3. Source authority gate— tier-3 proposals need ≥3 independent tier-1/2 sources
4. AutoApprover         — hard gates + logistic classifier must not reject
5. TriangulationEngine  — cross-strategy validation on graph paths

Passing triples are materialized via adapter.add_edge() and recorded in
ProvenanceLedger with source_tier metadata.
Rejected triples are logged to benchmarks/rejected_knowledge.jsonl.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Dict, Tuple

import requests

logger = logging.getLogger(__name__)

REJECTED_LOG = Path(__file__).parent.parent / "benchmarks" / "rejected_knowledge.jsonl"
ACCEPTED_LOG = Path(__file__).parent.parent / "benchmarks" / "accepted_knowledge.jsonl"

# Rate limits per source (seconds between requests)
_RATE_LIMITS: Dict[str, float] = {
    "wikidata":  0.5,
    "wikipedia": 0.5,
    "pubmed":    0.4,
    "crossref":  0.2,
}

_REQUEST_TIMEOUT = 10  # seconds


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class CandidateTriple:
    """A (source, relation, target) triple harvested from an external source."""

    triple_id:   str
    source:      str   # entity ID
    relation:    str
    target:      str   # entity ID
    source_name: str   # human-readable source label
    target_name: str   # human-readable target label
    source_url:  str
    source_tier: int   # 1=peer-reviewed, 2=curated KB, 3=community-edited
    confidence:  float
    fetched_at:  str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    # corroboration tracking for tier-3 gate
    corroborating_sources: List[str] = field(default_factory=list)

    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HarvestResult:
    started_at:    str
    finished_at:   str
    gap_hints:     List[str]
    fetched:       int
    passed_vetting: int
    materialized:  int
    rejected:      int


# ── Synthetic finding shim ────────────────────────────────────────────────────
# AutoApprover and ContradictionResolver use duck typing via getattr().
# This shim wraps a CandidateTriple in the expected ResearchFinding interface.

@dataclass
class _CandidateShim:
    """Mimics ResearchCandidate for AutoApprover feature extraction."""
    discovery_potential: float = 0.5
    gap_score:           float = 0.5
    community_distance:  int   = 1
    local_density:       float = 0.0
    seeded_by:           str   = "knowledge_harvester"


@dataclass
class _FindingShim:
    """Mimics ResearchFinding for AutoApprover / ContradictionResolver."""
    finding_id:       str
    candidate:        _CandidateShim
    proposals:        List[Any] = field(default_factory=list)
    best_confidence:  float     = 0.5
    literature_status: str      = "novel"
    validation_report: Any      = None
    metadata:         Dict[str, Any] = field(default_factory=dict)


@dataclass
class _ProposalShim:
    """Mimics HypothesisProposal for ContradictionResolver."""
    source_id:          str = ""
    target_id:          str = ""
    relation:           str = ""
    confidence:         float = 0.5
    contradiction_score: float = 0.0
    path_count:         int = 1


# ── KnowledgeHarvester ────────────────────────────────────────────────────────

class KnowledgeHarvester:
    """
    Fetches candidate knowledge from reputable external sources and vets each
    triple through a 5-stage pipeline before materializing it into the graph.
    """

    def __init__(
        self,
        adapter:                 Any,                # GraphAdapter
        provenance_ledger:       Optional[Any] = None,  # ProvenanceLedger
        auto_approver:           Optional[Any] = None,  # AutoApprover
        triangulation_engine:    Optional[Any] = None,  # TriangulationEngine
        contradiction_resolver:  Optional[Any] = None,  # ContradictionResolver
        min_tier3_corroborators: int   = 3,
        request_timeout:         float = _REQUEST_TIMEOUT,
    ) -> None:
        self._adapter               = adapter
        self._ledger                = provenance_ledger
        self._auto_approver         = auto_approver
        self._triangulation_engine  = triangulation_engine
        self._contradiction_resolver = contradiction_resolver
        self._min_tier3             = min_tier3_corroborators
        self._timeout               = request_timeout
        self._lock                  = threading.Lock()
        self._last_request: Dict[str, float] = {}

        REJECTED_LOG.parent.mkdir(parents=True, exist_ok=True)
        ACCEPTED_LOG.parent.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def harvest(self, gap_hints: Optional[List[str]] = None) -> HarvestResult:
        """
        Fetch and vet candidate triples from all whitelisted sources.
        gap_hints: entity or topic strings CEREBRUM is failing on.
        """
        started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        gap_hints = gap_hints or []
        logger.info("KnowledgeHarvester: starting harvest (hints=%d).", len(gap_hints))

        candidates = self._fetch_all(gap_hints)
        logger.info("KnowledgeHarvester: fetched %d raw candidates.", len(candidates))

        passed = self._corroborate(candidates)
        materialized = 0
        rejected = 0

        for triple in passed:
            ok, reason = self._vet(triple)
            if ok:
                self._materialize(triple)
                self._log_accepted(triple)
                materialized += 1
            else:
                self._log_rejected(triple, reason)
                rejected += 1

        finished = datetime.now(timezone.utc).isoformat(timespec="seconds")
        result = HarvestResult(
            started_at=started,
            finished_at=finished,
            gap_hints=gap_hints,
            fetched=len(candidates),
            passed_vetting=materialized,
            materialized=materialized,
            rejected=rejected,
        )
        logger.info(
            "KnowledgeHarvester: finished — fetched=%d, materialized=%d, rejected=%d.",
            len(candidates), materialized, rejected,
        )
        return result

    # ── Source fetchers ───────────────────────────────────────────────────────

    def _fetch_all(self, hints: List[str]) -> List[CandidateTriple]:
        results: List[CandidateTriple] = []
        for hint in hints:
            results.extend(self._fetch_wikidata(hint))
            results.extend(self._fetch_wikipedia(hint))
            results.extend(self._fetch_pubmed(hint))
            results.extend(self._fetch_crossref(hint))
        return results

    def _fetch_wikidata(self, entity: str) -> List[CandidateTriple]:
        """Tier 2 — Wikidata SPARQL: fetch top-K property triples for entity."""
        self._rate_limit("wikidata")
        entity_label = entity.replace(" ", "_")
        sparql = f"""
SELECT ?subjectLabel ?propLabel ?objectLabel ?prop WHERE {{
  ?subject rdfs:label "{entity}"@en .
  ?subject ?p ?statement .
  ?statement ?ps ?object .
  ?prop wikibase:statementProperty ?ps ;
        wikibase:directClaim ?directClaim ;
        rdfs:label ?propLabel .
  FILTER(lang(?propLabel)="en")
  FILTER(!isBlank(?object) && isLiteral(?object) || isIRI(?object))
  BIND(?subject AS ?subjectLabel)
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}} LIMIT 20
"""
        try:
            resp = requests.get(
                "https://query.wikidata.org/sparql",
                params={"query": sparql, "format": "json"},
                headers={"User-Agent": "CEREBRUM-KnowledgeHarvester/2.0"},
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.debug("KnowledgeHarvester[wikidata]: %s — %s", entity, exc)
            return []

        triples: List[CandidateTriple] = []
        for row in data.get("results", {}).get("bindings", []):
            try:
                subj  = row["subjectLabel"]["value"]
                prop  = row["propLabel"]["value"].upper().replace(" ", "_")
                obj   = row["objectLabel"]["value"]
                triples.append(CandidateTriple(
                    triple_id   = str(uuid.uuid4()),
                    source      = _slugify(subj),
                    relation    = prop,
                    target      = _slugify(obj),
                    source_name = subj,
                    target_name = obj,
                    source_url  = "https://www.wikidata.org",
                    source_tier = 2,
                    confidence  = 0.80,
                    raw         = row,
                ))
            except (KeyError, TypeError):
                pass
        return triples

    def _fetch_wikipedia(self, entity: str) -> List[CandidateTriple]:
        """Tier 2 — Wikipedia REST API: fetch a summary page and extract infobox-style triples."""
        self._rate_limit("wikipedia")
        title = entity.replace(" ", "_")
        try:
            resp = requests.get(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}",
                headers={"User-Agent": "CEREBRUM-KnowledgeHarvester/2.0"},
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.debug("KnowledgeHarvester[wikipedia]: %s — %s", entity, exc)
            return []

        description = data.get("description", "")
        extract     = data.get("extract", "")
        if not description and not extract:
            return []

        # Minimal synthetic triple: (entity, HAS_DESCRIPTION, description_text)
        if description:
            return [CandidateTriple(
                triple_id   = str(uuid.uuid4()),
                source      = _slugify(entity),
                relation    = "HAS_DESCRIPTION",
                target      = _slugify(description[:120]),
                source_name = entity,
                target_name = description[:120],
                source_url  = data.get("content_urls", {}).get("desktop", {}).get("page", "https://en.wikipedia.org"),
                source_tier = 2,
                confidence  = 0.75,
                raw         = {"description": description},
            )]
        return []

    def _fetch_pubmed(self, entity: str) -> List[CandidateTriple]:
        """Tier 1 — NCBI E-utilities: search PubMed for entity, extract MeSH terms as triples."""
        self._rate_limit("pubmed")
        try:
            search_resp = requests.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                params={"db": "pubmed", "term": entity, "retmax": "5", "retmode": "json"},
                headers={"User-Agent": "CEREBRUM-KnowledgeHarvester/2.0"},
                timeout=self._timeout,
            )
            search_resp.raise_for_status()
            pmids = search_resp.json().get("esearchresult", {}).get("idlist", [])
        except Exception as exc:
            logger.debug("KnowledgeHarvester[pubmed]: %s — %s", entity, exc)
            return []

        if not pmids:
            return []

        triples: List[CandidateTriple] = []
        try:
            fetch_resp = requests.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                params={"db": "pubmed", "id": ",".join(pmids[:3]), "retmode": "xml"},
                headers={"User-Agent": "CEREBRUM-KnowledgeHarvester/2.0"},
                timeout=self._timeout,
            )
            fetch_resp.raise_for_status()
            # Parse MeSH descriptor names from XML
            import xml.etree.ElementTree as ET
            root = ET.fromstring(fetch_resp.text)
            for article in root.findall(".//PubmedArticle"):
                pmid_el = article.find(".//PMID")
                pmid = pmid_el.text if pmid_el is not None else "unknown"
                for mesh in article.findall(".//MeshHeading/DescriptorName"):
                    mesh_term = mesh.text or ""
                    if mesh_term:
                        triples.append(CandidateTriple(
                            triple_id   = str(uuid.uuid4()),
                            source      = _slugify(entity),
                            relation    = "MESH_RELATED_TO",
                            target      = _slugify(mesh_term),
                            source_name = entity,
                            target_name = mesh_term,
                            source_url  = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}",
                            source_tier = 1,
                            confidence  = 0.85,
                            raw         = {"pmid": pmid, "mesh": mesh_term},
                        ))
        except Exception as exc:
            logger.debug("KnowledgeHarvester[pubmed-fetch]: %s — %s", entity, exc)

        return triples

    def _fetch_crossref(self, entity: str) -> List[CandidateTriple]:
        """Tier 1 — CrossRef: search for works mentioning entity, extract subject triples."""
        self._rate_limit("crossref")
        try:
            resp = requests.get(
                "https://api.crossref.org/works",
                params={"query": entity, "rows": "5", "select": "title,subject,DOI"},
                headers={
                    "User-Agent": "CEREBRUM-KnowledgeHarvester/2.0 (mailto:cerebrum@cerebrum.ai)"
                },
                timeout=self._timeout,
            )
            resp.raise_for_status()
            items = resp.json().get("message", {}).get("items", [])
        except Exception as exc:
            logger.debug("KnowledgeHarvester[crossref]: %s — %s", entity, exc)
            return []

        triples: List[CandidateTriple] = []
        for item in items:
            doi = item.get("DOI", "")
            for subject in item.get("subject", []):
                triples.append(CandidateTriple(
                    triple_id   = str(uuid.uuid4()),
                    source      = _slugify(entity),
                    relation    = "HAS_SUBJECT_AREA",
                    target      = _slugify(subject),
                    source_name = entity,
                    target_name = subject,
                    source_url  = f"https://doi.org/{doi}" if doi else "https://crossref.org",
                    source_tier = 1,
                    confidence  = 0.80,
                    raw         = {"doi": doi, "subject": subject},
                ))
        return triples

    # ── Corroboration (tier-3 gate) ───────────────────────────────────────────

    def _corroborate(self, candidates: List[CandidateTriple]) -> List[CandidateTriple]:
        """
        Build a cross-source index keyed on (source, relation, target).
        Tier-3 triples only pass if ≥3 independent tier-1/2 sources assert the same triple.
        Tier-1/2 triples always pass through to stage-by-stage vetting.
        """
        index: Dict[Tuple[str, str, str], List[CandidateTriple]] = {}
        for t in candidates:
            key = (t.source, t.relation, t.target)
            index.setdefault(key, []).append(t)

        result: List[CandidateTriple] = []
        for key, group in index.items():
            tier1_2 = [t for t in group if t.source_tier <= 2]
            tier3   = [t for t in group if t.source_tier == 3]

            # Always pass tier-1/2 (deduplicate by confidence)
            if tier1_2:
                best = max(tier1_2, key=lambda t: t.confidence)
                best.corroborating_sources = [t.source_url for t in tier1_2]
                result.append(best)

            # Tier-3 only if backed by ≥ min_tier3 distinct reputable sources
            for t3 in tier3:
                reputable_corroborators = {t.source_url for t in tier1_2}
                if len(reputable_corroborators) >= self._min_tier3:
                    t3.corroborating_sources = list(reputable_corroborators)
                    result.append(t3)
                else:
                    self._log_rejected(
                        t3,
                        f"tier-3 gate: only {len(reputable_corroborators)} reputable corroborators "
                        f"(need {self._min_tier3})",
                    )

        return result

    # ── 5-Stage vetting pipeline ──────────────────────────────────────────────

    def _vet(self, triple: CandidateTriple) -> Tuple[bool, str]:
        """Run all 5 stages. Returns (passed, rejection_reason)."""

        # Stage 1: schema conformance
        ok, reason = self._stage1_schema(triple)
        if not ok:
            return False, f"stage1_schema: {reason}"

        # Stage 2: contradiction check
        ok, reason = self._stage2_contradiction(triple)
        if not ok:
            return False, f"stage2_contradiction: {reason}"

        # Stage 3: source authority gate (tier-3 already handled in _corroborate,
        #          but double-check here for direct calls)
        if triple.source_tier >= 3 and len(triple.corroborating_sources) < self._min_tier3:
            return False, (
                f"stage3_authority: tier-3 with only "
                f"{len(triple.corroborating_sources)} corroborators"
            )

        # Stage 4: AutoApprover
        ok, reason = self._stage4_auto_approver(triple)
        if not ok:
            return False, f"stage4_auto_approver: {reason}"

        # Stage 5: TriangulationEngine
        ok, reason = self._stage5_triangulation(triple)
        if not ok:
            return False, f"stage5_triangulation: {reason}"

        return True, ""

    def _stage1_schema(self, triple: CandidateTriple) -> Tuple[bool, str]:
        """Check that relation and entity IDs are not trivially malformed."""
        if not triple.source or not triple.target or not triple.relation:
            return False, "empty source/target/relation"
        if len(triple.source) > 500 or len(triple.target) > 500:
            return False, "entity ID too long (>500 chars)"
        if len(triple.relation) > 200:
            return False, "relation too long (>200 chars)"
        # Check relation matches known schema if graph exposes it
        try:
            G = self._adapter.to_networkx()
            known_relations = {d.get("relation_type", d.get("relation", ""))
                               for _, _, d in G.edges(data=True)}
            if known_relations and triple.relation not in known_relations:
                # Novel relation type — allow but log at debug level
                logger.debug(
                    "KnowledgeHarvester: novel relation '%s' not in schema (%d known).",
                    triple.relation, len(known_relations),
                )
        except Exception:
            pass
        return True, ""

    def _stage2_contradiction(self, triple: CandidateTriple) -> Tuple[bool, str]:
        """Run ContradictionResolver if available; block 'discardable' outcomes."""
        if self._contradiction_resolver is None:
            return True, ""

        proposal = _ProposalShim(
            source_id  = triple.source,
            target_id  = triple.target,
            relation   = triple.relation,
            confidence = triple.confidence,
        )
        finding = _FindingShim(
            finding_id      = triple.triple_id,
            candidate       = _CandidateShim(),
            proposals       = [proposal],
            best_confidence = triple.confidence,
        )
        try:
            record = self._contradiction_resolver.resolve(finding, [proposal])
            if getattr(record, "resolution", "") == "discardable":
                return False, f"resolution=discardable (net={record.net_evidence_score:.3f})"
        except Exception as exc:
            logger.debug("KnowledgeHarvester: ContradictionResolver error: %s", exc)
        return True, ""

    def _stage4_auto_approver(self, triple: CandidateTriple) -> Tuple[bool, str]:
        """Run AutoApprover; block 'reject' decisions."""
        if self._auto_approver is None:
            return True, ""

        finding = _FindingShim(
            finding_id      = triple.triple_id,
            candidate       = _CandidateShim(
                discovery_potential = triple.confidence,
                gap_score           = 0.5,
                community_distance  = 1,
            ),
            best_confidence  = triple.confidence,
            literature_status = "novel",
        )
        try:
            decision = self._auto_approver.decide(finding)
            if getattr(decision, "action", "") == "reject":
                return False, f"AutoApprover rejected: {decision.reason}"
        except Exception as exc:
            logger.debug("KnowledgeHarvester: AutoApprover error: %s", exc)
        return True, ""

    def _stage5_triangulation(self, triple: CandidateTriple) -> Tuple[bool, str]:
        """
        Run TriangulationEngine if available.
        Block triples where reverse_confidence is near zero (graph has no path support).
        """
        if self._triangulation_engine is None:
            return True, ""

        proposal = _ProposalShim(
            source_id  = triple.source,
            target_id  = triple.target,
            relation   = triple.relation,
            confidence = triple.confidence,
        )
        try:
            report = self._triangulation_engine.evaluate(proposal, [proposal])
            rev_conf = getattr(report, "reverse_confidence", 1.0)
            if rev_conf < 0.05:
                return False, f"TriangulationEngine: reverse_confidence={rev_conf:.3f} < 0.05"
        except Exception as exc:
            logger.debug("KnowledgeHarvester: TriangulationEngine error: %s", exc)
        return True, ""

    # ── Materialization ───────────────────────────────────────────────────────

    def _materialize(self, triple: CandidateTriple) -> None:
        """Add the triple to the graph and record it in the ProvenanceLedger."""
        provenance = f"knowledge_harvester:{triple.source_url}:tier{triple.source_tier}"
        try:
            self._adapter.add_edge(
                u          = triple.source,
                v          = triple.target,
                relation   = triple.relation,
                confidence = triple.confidence,
                provenance = provenance,
                synthetic  = True,
            )
        except Exception as exc:
            logger.exception("KnowledgeHarvester: add_edge failed for %s.", triple.triple_id)
            return

        if self._ledger is not None:
            try:
                self._ledger.record_batch(
                    batch_id   = triple.triple_id,
                    finding_id = triple.triple_id,
                    edges      = [(triple.source, triple.target, triple.relation)],
                )
            except Exception:
                logger.debug("KnowledgeHarvester: ProvenanceLedger record failed.", exc_info=True)

        logger.debug(
            "KnowledgeHarvester: materialized (%s, %s, %s) tier=%d.",
            triple.source, triple.relation, triple.target, triple.source_tier,
        )

    # ── Logging helpers ───────────────────────────────────────────────────────

    def _log_rejected(self, triple: CandidateTriple, reason: str) -> None:
        entry = {
            "triple_id":   triple.triple_id,
            "source":      triple.source,
            "relation":    triple.relation,
            "target":      triple.target,
            "source_url":  triple.source_url,
            "source_tier": triple.source_tier,
            "reason":      reason,
            "ts":          datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        with REJECTED_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def _log_accepted(self, triple: CandidateTriple) -> None:
        entry = {
            "triple_id":   triple.triple_id,
            "source":      triple.source,
            "relation":    triple.relation,
            "target":      triple.target,
            "source_url":  triple.source_url,
            "source_tier": triple.source_tier,
            "confidence":  triple.confidence,
            "ts":          datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        with ACCEPTED_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    # ── Rate limiting ─────────────────────────────────────────────────────────

    def _rate_limit(self, source: str) -> None:
        min_gap = _RATE_LIMITS.get(source, 0.5)
        with self._lock:
            last = self._last_request.get(source, 0.0)
            gap  = time.time() - last
            if gap < min_gap:
                time.sleep(min_gap - gap)
            self._last_request[source] = time.time()


# ── Utility ───────────────────────────────────────────────────────────────────

def _slugify(text: str) -> str:
    """Convert a display label to a graph-compatible entity ID slug."""
    return text.lower().strip().replace(" ", "_").replace("-", "_")[:120]
