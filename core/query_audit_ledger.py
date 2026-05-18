"""
QueryAuditLedger — compliance-mode query logging for CEREBRUM.

In compliance mode (`cerebrum serve --compliance`), every query is logged
with timestamp, user/client identity, query string, answer, confidence,
and the full hop-by-hop reasoning trace.  The ledger is in-memory with
optional file persistence and CSV/JSON export for regulatory review.

    from core.query_audit_ledger import QueryAuditLedger

    ledger = QueryAuditLedger(log_file="audit.jsonl")
    ledger.record(
        query="Who directed Inception?",
        answer="Christopher_Nolan",
        confidence=0.923,
        trace_path=[("Inception", "directed_by"), ("Christopher_Nolan", "")],
        elapsed_ms=14.2,
        client_id="user@example.com",
    )
    ledger.export_csv("audit_export.csv")
    ledger.export_json("audit_export.json")
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Deque, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class QueryAuditRecord:
    """One logged query-answer pair with full reasoning provenance."""
    record_id:    int
    timestamp:    float
    iso_time:     str
    client_id:    str
    query:        str
    answer:       str
    confidence:   float
    hop_depth:    int
    trace_path:   List[Tuple[str, str]]  # [(entity, relation), ...]
    elapsed_ms:   float
    extra:        dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["trace_path"] = [{"entity": e, "relation": r} for e, r in self.trace_path]
        return d


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------

class QueryAuditLedger:
    """
    Thread-safe in-memory audit ledger with optional JSONL persistence.

    Parameters
    ----------
    max_records : Maximum records to keep in memory (FIFO; oldest dropped first).
    log_file    : If set, every record is also appended to this JSONL file.
    """

    def __init__(self, max_records: int = 10_000, log_file: Optional[str] = None) -> None:
        self._records: Deque[QueryAuditRecord] = deque(maxlen=max_records)
        self._lock = threading.Lock()
        self._counter = 0
        self._log_file = Path(log_file) if log_file else None
        if self._log_file:
            self._log_file.parent.mkdir(parents=True, exist_ok=True)
            logger.info("QueryAuditLedger: logging to %s", self._log_file)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        query: str,
        answer: str,
        confidence: float,
        trace_path: list,
        elapsed_ms: float,
        *,
        client_id: str = "anonymous",
        extra: dict = None,
    ) -> QueryAuditRecord:
        """
        Log one query-answer event.

        Parameters
        ----------
        trace_path : List of (entity, relation) tuples OR TraceStep objects.
        """
        import datetime

        # Normalise trace_path
        pairs = []
        for step in (trace_path or []):
            if isinstance(step, tuple):
                pairs.append(step)
            elif hasattr(step, "entity") and hasattr(step, "relation"):
                pairs.append((step.entity, step.relation))
            else:
                pairs.append((str(step), ""))

        ts = time.time()
        with self._lock:
            self._counter += 1
            rec = QueryAuditRecord(
                record_id  = self._counter,
                timestamp  = ts,
                iso_time   = datetime.datetime.utcfromtimestamp(ts).isoformat() + "Z",
                client_id  = client_id,
                query      = query,
                answer     = answer,
                confidence = round(float(confidence), 6),
                hop_depth  = len(pairs),
                trace_path = pairs,
                elapsed_ms = round(float(elapsed_ms), 2),
                extra      = extra or {},
            )
            self._records.append(rec)

        if self._log_file:
            try:
                with open(self._log_file, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec.to_dict()) + "\n")
            except Exception as exc:
                logger.warning("QueryAuditLedger: failed to write to %s: %s", self._log_file, exc)

        return rec

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def recent(self, n: int = 100) -> List[QueryAuditRecord]:
        """Return the last *n* records, newest first."""
        with self._lock:
            return list(reversed(list(self._records)))[:n]

    def all_records(self) -> List[QueryAuditRecord]:
        """Return all in-memory records (oldest first)."""
        with self._lock:
            return list(self._records)

    def __len__(self) -> int:
        return len(self._records)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_json(self, path: str, *, n: Optional[int] = None) -> Path:
        """Export audit log as JSON array."""
        out = Path(path)
        records = self.all_records() if n is None else self.recent(n)
        out.write_text(
            json.dumps([r.to_dict() for r in records], indent=2),
            encoding="utf-8",
        )
        logger.info("QueryAuditLedger: exported %d records to %s", len(records), out)
        return out

    def export_csv(self, path: str, *, n: Optional[int] = None) -> Path:
        """Export audit log as CSV (trace_path serialised as JSON string)."""
        import csv as _csv
        out = Path(path)
        records = self.all_records() if n is None else self.recent(n)
        if not records:
            out.write_text("record_id,timestamp,iso_time,client_id,query,answer,confidence,hop_depth,elapsed_ms,trace_path\n")
            return out
        with open(out, "w", newline="", encoding="utf-8") as fh:
            writer = _csv.writer(fh)
            writer.writerow(["record_id", "timestamp", "iso_time", "client_id",
                             "query", "answer", "confidence", "hop_depth",
                             "elapsed_ms", "trace_path"])
            for r in records:
                trace_json = json.dumps([{"entity": e, "relation": rel} for e, rel in r.trace_path])
                writer.writerow([r.record_id, r.timestamp, r.iso_time, r.client_id,
                                 r.query, r.answer, r.confidence, r.hop_depth,
                                 r.elapsed_ms, trace_json])
        logger.info("QueryAuditLedger: exported %d records to %s", len(records), out)
        return out

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Summary statistics for the audit log."""
        records = self.all_records()
        if not records:
            return {"total": 0}
        confs = [r.confidence for r in records]
        return {
            "total": len(records),
            "first_at": records[0].iso_time,
            "last_at": records[-1].iso_time,
            "avg_confidence": round(sum(confs) / len(confs), 4),
            "avg_elapsed_ms": round(sum(r.elapsed_ms for r in records) / len(records), 2),
            "unique_clients": len({r.client_id for r in records}),
        }
