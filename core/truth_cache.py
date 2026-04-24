"""
TruthCache — Phase 118: Immutable Proof Ledger.

Stores verified reasoning paths and their accompanying symbolic proofs.
"""
import hashlib
import json
from typing import Dict, Any

class TruthCache:
    """Immutable ledger of proven facts."""
    def __init__(self):
        self._cache: Dict[str, Any] = {}

    def store_proof(self, path: List[Any], proof: Dict[str, Any]):
        path_hash = hashlib.sha256(json.dumps(path, sort_keys=True).encode()).hexdigest()
        self._cache[path_hash] = proof

    def get_proof(self, path: List[Any]) -> Optional[Dict[str, Any]]:
        path_hash = hashlib.sha256(json.dumps(path, sort_keys=True).encode()).hexdigest()
        return self._cache.get(path_hash)
