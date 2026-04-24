"""
TruthCache — Phase 118: Immutable Proof Ledger.

Stores verified reasoning paths and their accompanying symbolic proofs.
Extended in Phase 120 to support typed CausalProof storage.
"""
import hashlib
import json
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.causal_engine import CausalProof

class TruthCache:
    """Immutable ledger of proven facts."""
    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._causal_cache: Dict[str, "CausalProof"] = {}

    def store_proof(self, path: List[Any], proof: Dict[str, Any]):
        path_hash = hashlib.sha256(json.dumps(path, sort_keys=True).encode()).hexdigest()
        self._cache[path_hash] = proof

    def get_proof(self, path: List[Any]) -> Optional[Dict[str, Any]]:
        path_hash = hashlib.sha256(json.dumps(path, sort_keys=True).encode()).hexdigest()
        return self._cache.get(path_hash)

    def store_causal_proof(self, source: str, target: str, proof: "CausalProof") -> None:
        """Store a CausalProof keyed by (source, target)."""
        key = f"{source}||{target}"
        self._causal_cache[key] = proof

    def get_causal_proof(self, source: str, target: str) -> Optional["CausalProof"]:
        """Retrieve a cached CausalProof, or None if not present."""
        key = f"{source}||{target}"
        return self._causal_cache.get(key)

    def clear_causal(self) -> None:
        """Clear the causal proof cache."""
        self._causal_cache.clear()
