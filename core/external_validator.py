"""
ExternalValidator — Literature Search for Hypothesis Validation (Phase 52).

Queries free public research databases to assess whether a proposed knowledge
graph edge is already known, actively being investigated, or genuinely novel.

Supported adapters (all free, no API key required):
  PubMedAdapter         — NCBI E-utilities (biomedical literature)
  ClinicalTrialsAdapter — ClinicalTrials.gov v2 API (active clinical studies)
  ArXivAdapter          — arXiv.org Atom feed (preprints; good for physics/CS/math)
  OpenAlexAdapter       — OpenAlex REST API (all disciplines; comprehensive)

Literature status tags:
  "novel"            — Zero hits across all queried adapters.
  "active_research"  — ClinicalTrials has a recruiting/active trial, OR 1-9 hits.
  "established"      — ≥10 hits (well-studied connection).
  "contested"        — Hits found for both the proposed relation AND its opposing relation.
  "unvalidated"      — No adapters were queried (offline or all failed).

Usage
-----
    from core.external_validator import ExternalValidator, PubMedAdapter
    from core.hypothesis_engine import HypothesisEngine

    validator = ExternalValidator()          # auto-selects all adapters
    report = validator.validate(proposal)    # HypothesisProposal from Phase 50
    print(report.literature_status, report.novelty_score)
"""
from __future__ import annotations

import json
import logging
import time
import threading
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Opposing relations (reuse from hypothesis_engine)
# ---------------------------------------------------------------------------

_OPPOSING_RELATIONS: Dict[str, str] = {
    "CAUSES":                "PREVENTS",
    "PREVENTS":              "CAUSES",
    "ACTIVATES":             "INHIBITS",
    "INHIBITS":              "ACTIVATES",
    "PROMOTES":              "SUPPRESSES",
    "SUPPRESSES":            "PROMOTES",
    "INCREASES":             "DECREASES",
    "DECREASES":             "INCREASES",
    "TREATS":                "WORSENS",
    "WORSENS":               "TREATS",
    "UPREGULATES":           "DOWNREGULATES",
    "DOWNREGULATES":         "UPREGULATES",
    "INDIRECTLY_CAUSES":     "INDIRECTLY_PREVENTS",
    "INDIRECTLY_PREVENTS":   "INDIRECTLY_CAUSES",
}

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LiteratureHit:
    """A single result from an external research database."""

    adapter: str
    """Source adapter name: ``"pubmed"`` | ``"clinical_trials"`` | ``"arxiv"`` | ``"openalex"``."""

    external_id: str
    """PMID, NCT number, arXiv ID, or OpenAlex work ID."""

    title: str
    year: Optional[int] = None
    relevance_score: float = 1.0
    """Normalized quality score [0, 1]; higher = more relevant."""


@dataclass
class ValidationReport:
    """Result of validating one HypothesisProposal against external literature."""

    hypothesis_id: str
    source_id: str
    target_id: str
    derived_relation: str

    literature_status: str
    """``"novel"`` | ``"active_research"`` | ``"established"`` | ``"contested"`` | ``"unvalidated"``."""

    novelty_score: float
    """1.0 = no hits (fully novel); 0.0 = heavily established (≥10 hits)."""

    hit_count: int
    hits: List[LiteratureHit] = field(default_factory=list)
    adapters_queried: List[str] = field(default_factory=list)
    checked_at: float = field(default_factory=time.time)
    error: Optional[str] = None
    """Set when one or more adapters raised an exception (partial results still returned)."""


# ---------------------------------------------------------------------------
# Adapter base class
# ---------------------------------------------------------------------------

class ExternalValidatorAdapter(ABC):
    """Abstract base for an external literature search adapter."""

    @abstractmethod
    def search(self, source: str, relation: str, target: str) -> List[LiteratureHit]:
        """
        Search for literature connecting *source* and *target* via *relation*.

        Parameters
        ----------
        source, target
            Entity names (plain string labels).
        relation
            Relation type string (e.g. ``"TREATS"``).

        Returns
        -------
        List[LiteratureHit]
            Found results; empty list = no hits.
        """

    @abstractmethod
    def name(self) -> str:
        """Short identifier used in ValidationReport.adapters_queried."""

    @staticmethod
    def _fetch(url: str, timeout: float = 8.0) -> str:
        """HTTP GET with timeout; returns response body as str."""
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "CEREBRUM-ResearchAgent/1.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# PubMed (NCBI E-utilities)
# ---------------------------------------------------------------------------

class PubMedAdapter(ExternalValidatorAdapter):
    """
    Queries PubMed via NCBI E-utilities (completely free, no API key).

    Uses the ``esearch`` endpoint to get hit counts and top PMIDs, then
    ``esummary`` to retrieve titles for display.
    """

    _ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    _ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

    def __init__(self, timeout: float = 8.0) -> None:
        self._timeout = timeout

    def name(self) -> str:
        return "pubmed"

    def _build_query(self, source: str, relation: str, target: str) -> str:
        """Build PubMed query URL for the entity pair."""
        query = urllib.parse.quote(f'"{source}" AND "{target}"')
        return (
            f"{self._ESEARCH}?db=pubmed&term={query}"
            f"&retmax=5&retmode=json&usehistory=n"
        )

    def search(self, source: str, relation: str, target: str) -> List[LiteratureHit]:
        try:
            url = self._build_query(source, relation, target)
            body = self._fetch(url, self._timeout)
            data = json.loads(body)
            result = data.get("esearchresult", {})
            count = int(result.get("count", 0))
            pmids = result.get("idlist", [])

            if count == 0:
                return []

            hits: List[LiteratureHit] = []
            # Fetch titles for the top PMIDs
            if pmids:
                ids_str = urllib.parse.quote(",".join(pmids[:5]))
                sum_url = f"{self._ESUMMARY}?db=pubmed&id={ids_str}&retmode=json"
                try:
                    sum_body = self._fetch(sum_url, self._timeout)
                    sum_data = json.loads(sum_body).get("result", {})
                    for pmid in pmids[:5]:
                        entry = sum_data.get(pmid, {})
                        title = entry.get("title", f"PubMed:{pmid}")
                        year_str = entry.get("pubdate", "")[:4]
                        year = int(year_str) if year_str.isdigit() else None
                        hits.append(LiteratureHit(
                            adapter="pubmed",
                            external_id=pmid,
                            title=title,
                            year=year,
                            relevance_score=1.0,
                        ))
                except Exception:
                    # Fall back: return count-only hits without titles
                    for pmid in pmids[:5]:
                        hits.append(LiteratureHit(
                            adapter="pubmed",
                            external_id=pmid,
                            title=f"PubMed:{pmid}",
                            relevance_score=1.0,
                        ))

            # If more hits than returned IDs, add synthetic records for the count
            for _ in range(min(count, 5) - len(hits)):
                hits.append(LiteratureHit(
                    adapter="pubmed",
                    external_id=f"count_{count}",
                    title=f"[{count} PubMed results]",
                    relevance_score=0.5,
                ))

            return hits[:5]

        except (OSError, TimeoutError) as exc:
            raise
        except Exception as exc:
            logger.debug("PubMedAdapter error for (%s, %s): %s", source, target, exc)
            return []


# ---------------------------------------------------------------------------
# ClinicalTrials.gov
# ---------------------------------------------------------------------------

class ClinicalTrialsAdapter(ExternalValidatorAdapter):
    """
    Queries ClinicalTrials.gov v2 API (completely free).

    Tags results as ``"active_research"`` when recruiting/active trials are found.
    """

    _BASE = "https://clinicaltrials.gov/api/v2/studies"
    _ACTIVE_STATUSES = {"RECRUITING", "ACTIVE_NOT_RECRUITING", "NOT_YET_RECRUITING", "ENROLLING_BY_INVITATION"}

    def __init__(self, timeout: float = 8.0) -> None:
        self._timeout = timeout

    def name(self) -> str:
        return "clinical_trials"

    def search(self, source: str, relation: str, target: str) -> List[LiteratureHit]:
        try:
            query = urllib.parse.quote(f"{source} {target}")
            url = f"{self._BASE}?query.term={query}&pageSize=5&format=json"
            body = self._fetch(url, self._timeout)
            data = json.loads(body)
            studies = data.get("studies", [])

            hits: List[LiteratureHit] = []
            for study in studies[:5]:
                proto = study.get("protocolSection", {})
                id_mod = proto.get("identificationModule", {})
                status_mod = proto.get("statusModule", {})
                nct = id_mod.get("nctId", "")
                title = id_mod.get("briefTitle", nct)
                status = status_mod.get("overallStatus", "")
                year_str = status_mod.get("startDateStruct", {}).get("date", "")[:4]
                year = int(year_str) if year_str.isdigit() else None
                is_active = status.upper() in self._ACTIVE_STATUSES
                hits.append(LiteratureHit(
                    adapter="clinical_trials",
                    external_id=nct,
                    title=title,
                    year=year,
                    relevance_score=1.0 if is_active else 0.6,
                ))
            return hits

        except (OSError, TimeoutError):
            raise
        except Exception as exc:
            logger.debug("ClinicalTrialsAdapter error for (%s, %s): %s", source, target, exc)
            return []


# ---------------------------------------------------------------------------
# arXiv
# ---------------------------------------------------------------------------

class ArXivAdapter(ExternalValidatorAdapter):
    """
    Queries the arXiv API (completely free, uses Atom/XML; stdlib only).

    Good for non-biomedical domains: physics, CS, math, economics.
    """

    _BASE = "http://export.arxiv.org/api/query"
    _NS = "http://www.w3.org/2005/Atom"

    def __init__(self, timeout: float = 8.0) -> None:
        self._timeout = timeout

    def name(self) -> str:
        return "arxiv"

    def search(self, source: str, relation: str, target: str) -> List[LiteratureHit]:
        try:
            query = urllib.parse.quote(f'all:"{source}" AND all:"{target}"')
            url = f"{self._BASE}?search_query={query}&max_results=5"
            body = self._fetch(url, self._timeout)
            root = ET.fromstring(body)
            ns = {"a": self._NS}
            hits: List[LiteratureHit] = []
            for entry in root.findall("a:entry", ns)[:5]:
                arxiv_id = (entry.findtext("a:id", "", ns) or "").split("/abs/")[-1]
                title = (entry.findtext("a:title", "", ns) or "").strip()
                published = (entry.findtext("a:published", "", ns) or "")[:4]
                year = int(published) if published.isdigit() else None
                hits.append(LiteratureHit(
                    adapter="arxiv",
                    external_id=arxiv_id,
                    title=title,
                    year=year,
                    relevance_score=1.0,
                ))
            return hits

        except (OSError, TimeoutError):
            raise
        except Exception as exc:
            logger.debug("ArXivAdapter error for (%s, %s): %s", source, target, exc)
            return []


# ---------------------------------------------------------------------------
# OpenAlex
# ---------------------------------------------------------------------------

class OpenAlexAdapter(ExternalValidatorAdapter):
    """
    Queries OpenAlex (completely free, all disciplines, no API key).

    Best general-purpose fallback covering biomedical, social science,
    humanities, engineering, and more.
    """

    _BASE = "https://api.openalex.org/works"

    def __init__(self, timeout: float = 8.0) -> None:
        self._timeout = timeout

    def name(self) -> str:
        return "openalex"

    def search(self, source: str, relation: str, target: str) -> List[LiteratureHit]:
        try:
            query = urllib.parse.quote(f"{source} {target}")
            url = f"{self._BASE}?search={query}&per-page=5&mailto=cerebrum@noreply.local"
            body = self._fetch(url, self._timeout)
            data = json.loads(body)
            results = data.get("results", [])

            hits: List[LiteratureHit] = []
            for work in results[:5]:
                oa_id = work.get("id", "").split("/")[-1]  # strip URL prefix
                title = work.get("title", oa_id) or oa_id
                year = work.get("publication_year")
                hits.append(LiteratureHit(
                    adapter="openalex",
                    external_id=oa_id,
                    title=title,
                    year=year,
                    relevance_score=1.0,
                ))
            return hits

        except (OSError, TimeoutError):
            raise
        except Exception as exc:
            logger.debug("OpenAlexAdapter error for (%s, %s): %s", source, target, exc)
            return []


# ---------------------------------------------------------------------------
# ExternalValidator
# ---------------------------------------------------------------------------

class ExternalValidator:
    """
    Orchestrates multiple literature adapters to validate a HypothesisProposal.

    Parameters
    ----------
    adapters
        List of ExternalValidatorAdapter instances to query.  If None, all
        four built-in adapters are used (PubMed, ClinicalTrials, arXiv, OpenAlex).
    timeout
        Per-adapter query timeout in seconds.
    cache_ttl
        Cache TTL in seconds (default 1 h).  Cached results skip network calls.
    """

    def __init__(
        self,
        adapters: Optional[List[ExternalValidatorAdapter]] = None,
        timeout: float = 10.0,
        cache_ttl: float = 3600.0,
    ) -> None:
        self._adapters: List[ExternalValidatorAdapter] = adapters or [
            PubMedAdapter(timeout=timeout),
            ClinicalTrialsAdapter(timeout=timeout),
            ArXivAdapter(timeout=timeout),
            OpenAlexAdapter(timeout=timeout),
        ]
        self._timeout  = timeout
        self._cache_ttl = cache_ttl
        self._cache: Dict[Tuple[str, str, str], Tuple[float, ValidationReport]] = {}
        self._lock = threading.Lock()

    @property
    def cache_size(self) -> int:
        """Number of cached validation results."""
        with self._lock:
            return len(self._cache)

    def validate(self, proposal) -> ValidationReport:
        """
        Validate one HypothesisProposal against all configured adapters.

        Results are cached by (source, relation, target) for cache_ttl seconds.

        Parameters
        ----------
        proposal
            HypothesisProposal from HypothesisEngine.generate().

        Returns
        -------
        ValidationReport
        """
        key = (proposal.source, proposal.derived_relation, proposal.target)

        # Check cache
        with self._lock:
            if key in self._cache:
                ts, report = self._cache[key]
                if time.time() - ts < self._cache_ttl:
                    return report

        # Query adapters in parallel
        all_hits: List[LiteratureHit] = []
        errors: List[str] = []
        queried: List[str] = []

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                pool.submit(adapter.search, proposal.source,
                            proposal.derived_relation, proposal.target): adapter
                for adapter in self._adapters
            }
            for future in as_completed(futures, timeout=self._timeout + 2):
                adapter = futures[future]
                queried.append(adapter.name())
                try:
                    hits = future.result(timeout=self._timeout)
                    all_hits.extend(hits)
                except (OSError, TimeoutError, FuturesTimeout) as exc:
                    errors.append(f"{adapter.name()}: network_unavailable")
                    logger.debug("%s timeout/network error: %s", adapter.name(), exc)
                except Exception as exc:
                    errors.append(f"{adapter.name()}: {exc}")
                    logger.debug("%s error: %s", adapter.name(), exc)

        # Check for contradicting evidence only when there are primary hits
        opposing_hits: List[LiteratureHit] = []
        if all_hits:
            opposing_hits = self._search_opposing(proposal, queried)
        contested = bool(all_hits) and len(opposing_hits) > 0

        # Determine literature_status
        hit_count = len(all_hits)
        clinical_active = any(
            h.adapter == "clinical_trials" and h.relevance_score >= 1.0
            for h in all_hits
        )
        if not queried:
            status = "unvalidated"
        elif contested:
            status = "contested"
        elif hit_count == 0:
            status = "novel"
        elif clinical_active or 1 <= hit_count <= 9:
            status = "active_research"
        else:
            status = "established"

        novelty_score = max(0.0, 1.0 - hit_count / 10.0)

        report = ValidationReport(
            hypothesis_id=proposal.hypothesis_id,
            source_id=proposal.source,
            target_id=proposal.target,
            derived_relation=proposal.derived_relation,
            literature_status=status,
            novelty_score=novelty_score,
            hit_count=hit_count,
            hits=all_hits[:10],  # cap serialized hits
            adapters_queried=queried,
            error="; ".join(errors) if errors else None,
        )

        # Cache result
        with self._lock:
            self._cache[key] = (time.time(), report)

        return report

    def validate_batch(self, proposals: List[Any]) -> List[ValidationReport]:
        """
        Validate a list of proposals; cached entries skip network calls.

        Returns
        -------
        List[ValidationReport]
            One report per proposal, in the same order.
        """
        reports: List[ValidationReport] = []
        for proposal in proposals:
            reports.append(self.validate(proposal))
        return reports

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _search_opposing(self, proposal, queried: List[str]) -> List[LiteratureHit]:
        """
        Search for evidence of the opposing relation to detect contested proposals.

        Returns hits for the opposing relation (e.g. PREVENTS when proposal is TREATS).
        """
        opposing_rel = _OPPOSING_RELATIONS.get(proposal.derived_relation.upper())
        if not opposing_rel:
            return []

        # Build a minimal proxy proposal with swapped relation
        class _Proxy:
            def __init__(self, src, tgt, rel, hyp_id):
                self.source = src
                self.target = tgt
                self.derived_relation = rel
                self.hypothesis_id = hyp_id

        proxy = _Proxy(
            proposal.source,
            proposal.target,
            opposing_rel,
            proposal.hypothesis_id + "_opposing",
        )

        all_hits: List[LiteratureHit] = []
        for adapter in self._adapters:
            try:
                hits = adapter.search(proxy.source, proxy.derived_relation, proxy.target)
                all_hits.extend(hits)
            except Exception:
                pass
        return all_hits
