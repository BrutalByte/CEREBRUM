"""
Centralized logging configuration for CEREBRUM.

Sets up the ``cerebrum.*`` logger hierarchy with:
  - StreamHandler (console)
  - RotatingFileHandler (optional, enabled via log_file parameter)
  - RingBufferHandler (always active — feeds the ``GET /logs`` dashboard endpoint)

Usage
-----
Early in startup (cli or server)::

    from core.log_config import setup_logging
    setup_logging(level="DEBUG", log_file="cerebrum.log")

Anywhere else::

    import logging
    log = logging.getLogger("cerebrum.mymodule")
    log.info("something happened")

The ring buffer is queryable via ``get_ring_handler().get_entries(...)``.
"""
import logging
import logging.handlers
from collections import deque
from threading import Lock
from typing import Dict, List, Optional, Any

# ── Constants ────────────────────────────────────────────────────────
LOG_BUFFER_MAXLEN = 5000
_LOG_FMT  = "%(asctime)s [%(levelname)8s] %(name)s — %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

# Loggers to capture in addition to the cerebrum hierarchy
_EXTRA_LOGGERS = ("uvicorn", "uvicorn.access", "uvicorn.error")


# ── Ring-buffer handler ───────────────────────────────────────────────
class RingBufferHandler(logging.Handler):
    """Thread-safe in-memory ring buffer consumed by the dashboard /logs endpoint."""

    def __init__(self, maxlen: int = LOG_BUFFER_MAXLEN) -> None:
        super().__init__()
        self._buffer: deque = deque(maxlen=maxlen)
        self._lock = Lock()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        entry: Dict[str, Any] = {
            "ts":     record.created,
            "level":  record.levelname,
            "logger": record.name,
            "module": record.module,
            "msg":    msg,
        }
        with self._lock:
            self._buffer.append(entry)

    def get_entries(
        self,
        level: Optional[str] = None,
        limit: int = 500,
        since: float = 0.0,
        search: str = "",
    ) -> List[Dict[str, Any]]:
        """Return recent log entries with optional filtering."""
        with self._lock:
            entries = list(self._buffer)
        if since:
            entries = [e for e in entries if e["ts"] > since]
        if level:
            entries = [e for e in entries if e["level"] == level.upper()]
        if search:
            lo = search.lower()
            entries = [e for e in entries if lo in e["msg"].lower() or lo in e["logger"].lower()]
        return entries[-limit:]

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()

    @property
    def entry_count(self) -> int:
        with self._lock:
            return len(self._buffer)


# ── Module-level singleton ────────────────────────────────────────────
_ring_handler: Optional[RingBufferHandler] = None


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> RingBufferHandler:
    """
    Configure the cerebrum logging hierarchy.

    Parameters
    ----------
    level        : Minimum log level for console and file output (DEBUG/INFO/WARNING/ERROR).
    log_file     : Optional path for a rotating log file.  ``None`` = no file output.
    max_bytes    : Max size of each log file before rotation (default 10 MB).
    backup_count : Number of rotated files to keep (default 5).

    Returns
    -------
    RingBufferHandler
        The in-memory ring buffer instance (always captures DEBUG+).
    """
    global _ring_handler

    level_int = getattr(logging, level.upper(), logging.INFO)
    fmt = logging.Formatter(_LOG_FMT, datefmt=_DATE_FMT)

    # ── Root cerebrum logger ─────────────────────────────────────────
    root = logging.getLogger("cerebrum")
    root.setLevel(logging.DEBUG)   # handlers govern effective level
    root.handlers.clear()
    root.propagate = False

    # ── Capture uvicorn loggers too ──────────────────────────────────
    for name in _EXTRA_LOGGERS:
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = False
        lg.setLevel(level_int)

    # ── Console handler ──────────────────────────────────────────────
    ch = logging.StreamHandler()
    ch.setLevel(level_int)
    ch.setFormatter(fmt)
    root.addHandler(ch)
    for name in _EXTRA_LOGGERS:
        logging.getLogger(name).addHandler(ch)

    # ── Rotating file handler (optional) ────────────────────────────
    if log_file:
        fh = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        fh.setLevel(level_int)
        fh.setFormatter(fmt)
        root.addHandler(fh)
        for name in _EXTRA_LOGGERS:
            logging.getLogger(name).addHandler(fh)

    # ── Ring buffer (always DEBUG so dashboard captures everything) ──
    _ring_handler = RingBufferHandler()
    _ring_handler.setLevel(logging.DEBUG)
    _ring_handler.setFormatter(fmt)
    root.addHandler(_ring_handler)
    for name in _EXTRA_LOGGERS:
        logging.getLogger(name).addHandler(_ring_handler)

    logging.getLogger("cerebrum").info(
        "Logging configured — level=%s  file=%s  buffer_maxlen=%d",
        level.upper(), log_file or "none", LOG_BUFFER_MAXLEN,
    )
    return _ring_handler


def get_ring_handler() -> RingBufferHandler:
    """Return the active ring buffer handler, initialising with defaults if needed."""
    global _ring_handler
    if _ring_handler is None:
        setup_logging()
    return _ring_handler  # type: ignore[return-value]
