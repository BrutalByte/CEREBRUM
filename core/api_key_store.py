"""
API Key Store — dynamic key management with per-key usage metering.

Keys are hashed (SHA-256) before storage so raw secrets never persist to disk.
Supports tenant namespacing: each key maps to a tenant_id that can be used
to route queries to a tenant-specific graph.
"""
import hashlib
import json
import logging
import os
import secrets
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("cerebrum.api_key_store")


def _hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


@dataclass
class ApiKeyUsage:
    queries_today: int = 0
    total_queries: int = 0
    total_latency_ms: float = 0.0
    last_used_at: Optional[float] = None
    day_epoch: int = 0  # UTC day number, used to reset queries_today

    @property
    def avg_latency_ms(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return round(self.total_latency_ms / self.total_queries, 2)

    def record(self, elapsed_ms: float) -> None:
        today = int(time.time() // 86400)
        if today != self.day_epoch:
            self.queries_today = 0
            self.day_epoch = today
        self.queries_today += 1
        self.total_queries += 1
        self.total_latency_ms += elapsed_ms
        self.last_used_at = time.time()


@dataclass
class ApiKey:
    key_id: str
    key_hash: str
    tenant_id: str
    label: str
    created_at: float
    is_active: bool = True
    usage: ApiKeyUsage = field(default_factory=ApiKeyUsage)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ApiKey":
        usage_data = d.pop("usage", {})
        obj = cls(**d)
        obj.usage = ApiKeyUsage(**usage_data)
        return obj

    def public_info(self) -> dict:
        """Safe subset for API responses — never exposes key_hash."""
        return {
            "key_id": self.key_id,
            "tenant_id": self.tenant_id,
            "label": self.label,
            "created_at": self.created_at,
            "is_active": self.is_active,
        }


class ApiKeyStore:
    """
    Thread-safe store for API keys with usage metering and JSON persistence.

    Keys are identified by a random key_id (shown to admin) and a raw secret
    that is only returned once on creation. The store only keeps the SHA-256
    hash of the raw secret.
    """

    def __init__(self, store_file: Optional[str] = None):
        self._lock = threading.Lock()
        self._keys: Dict[str, ApiKey] = {}          # key_id → ApiKey
        self._hash_index: Dict[str, str] = {}        # key_hash → key_id
        self._store_file = store_file
        if store_file and Path(store_file).exists():
            self._load()

    # ── CRUD ─────────────────────────────────────────────────────────────────

    def generate(self, label: str, tenant_id: str = "default") -> tuple[str, str]:
        """
        Create a new API key.
        Returns (key_id, raw_secret). raw_secret is shown ONCE and never stored.
        """
        raw_secret = secrets.token_urlsafe(32)
        key_id = secrets.token_hex(8)
        key = ApiKey(
            key_id=key_id,
            key_hash=_hash_key(raw_secret),
            tenant_id=tenant_id,
            label=label,
            created_at=time.time(),
        )
        with self._lock:
            self._keys[key_id] = key
            self._hash_index[key.key_hash] = key_id
        self._persist()
        logger.info("Generated API key %s for tenant=%s label=%r", key_id, tenant_id, label)
        return key_id, raw_secret

    def revoke(self, key_id: str) -> bool:
        """Mark a key as inactive. Returns False if key not found."""
        with self._lock:
            key = self._keys.get(key_id)
            if key is None:
                return False
            key.is_active = False
        self._persist()
        logger.info("Revoked API key %s", key_id)
        return True

    def list_keys(self, tenant_id: Optional[str] = None) -> List[dict]:
        with self._lock:
            keys = list(self._keys.values())
        if tenant_id is not None:
            keys = [k for k in keys if k.tenant_id == tenant_id]
        return [k.public_info() for k in keys]

    def get(self, key_id: str) -> Optional[ApiKey]:
        with self._lock:
            return self._keys.get(key_id)

    # ── Validation ───────────────────────────────────────────────────────────

    def validate(self, raw_key: str) -> Optional[ApiKey]:
        """Return ApiKey if valid and active, else None."""
        h = _hash_key(raw_key)
        with self._lock:
            key_id = self._hash_index.get(h)
            if key_id is None:
                return None
            key = self._keys[key_id]
        return key if key.is_active else None

    # ── Usage metering ────────────────────────────────────────────────────────

    def record_usage(self, key_id: str, elapsed_ms: float) -> None:
        with self._lock:
            key = self._keys.get(key_id)
            if key is not None:
                key.usage.record(elapsed_ms)

    def get_usage(self, key_id: str) -> Optional[dict]:
        with self._lock:
            key = self._keys.get(key_id)
            if key is None:
                return None
            usage = key.usage
            return {
                "key_id": key_id,
                "tenant_id": key.tenant_id,
                "label": key.label,
                "queries_today": usage.queries_today,
                "total_queries": usage.total_queries,
                "avg_latency_ms": usage.avg_latency_ms,
                "last_used_at": usage.last_used_at,
            }

    # ── Persistence ──────────────────────────────────────────────────────────

    def _persist(self) -> None:
        if not self._store_file:
            return
        try:
            Path(self._store_file).parent.mkdir(parents=True, exist_ok=True)
            tmp = self._store_file + ".tmp"
            with self._lock:
                data = [k.to_dict() for k in self._keys.values()]
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, self._store_file)
        except Exception as exc:
            logger.error("ApiKeyStore persist failed: %s", exc)

    def _load(self) -> None:
        try:
            with open(self._store_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                key = ApiKey.from_dict(item)
                self._keys[key.key_id] = key
                self._hash_index[key.key_hash] = key.key_id
            logger.info("Loaded %d API keys from %s", len(self._keys), self._store_file)
        except Exception as exc:
            logger.error("ApiKeyStore load failed: %s", exc)
