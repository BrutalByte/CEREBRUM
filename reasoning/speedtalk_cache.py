"""
SpeedTalk-Compressed Engram Cache (Phase 58).

A Heinlein-inspired encoding layer over the Engram relation-pattern cache.
Each relation type is assigned a single-character "phoneme" (a character from
a 62-symbol alphabet), so relation sequences are stored as compact strings
rather than verbose tuples.

Analogy to Heinlein's SpeedTalk ("Gulf", 1949 / "Friday"):
    - Every primitive concept → single phoneme
    - Sequences of concepts → concatenated phoneme strings
    - Most-common concepts get the shortest symbols (frequency-order assignment)

New capabilities over Engram
--------------------------------
    prefix_query(*rels)
        Find all cached patterns whose relation sequence *starts with* the
        given relations.  Works because encoded string prefix ↔ relation
        sequence prefix — each character encodes exactly one relation.

    alphabet()
        Expose the full {relation_type → phoneme_char} mapping so callers
        can inspect or visualise the learned vocabulary.

    compression_stats()
        Report key-length reduction, vocabulary size, and estimated memory
        savings vs. plain tuple string storage.

Classes
-------
    SpeedTalkEncoder           — relation-type ↔ phoneme bijection
    SpeedTalkEngram            — compressed cache with prefix-query API
    SpeedTalkEngramTraversal   — BeamTraversal variant using the above cache

Usage
-----
    from reasoning.speedtalk_cache import SpeedTalkEngramTraversal, SpeedTalkEngram

    cache = SpeedTalkEngram()
    traversal = SpeedTalkEngramTraversal(adapter, csa, cache=cache, engram_strength=0.3)

    paths = traversal.traverse(seeds)
    answers = extract(paths, ...)
    traversal.record_answers(answers, min_score=0.5)

    # Prefix query — new capability not present in Engram
    patterns = cache.prefix_query("CAUSES")
    # → [(('CAUSES', 'TREATS'), 5), (('CAUSES', 'INHIBITS', 'PREVENTS'), 3)]

    stats = cache.compression_stats()
    # → {'vocab_size': 8, 'compression_ratio': 11.2, ...}
"""
from __future__ import annotations

import json
import logging
import threading
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_log = logging.getLogger("cerebrum.speedtalk")

# ---------------------------------------------------------------------------
# Base alphabet — 62 single-char slots before overflow
# ---------------------------------------------------------------------------

_BASE_ALPHABET = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
)


# ---------------------------------------------------------------------------
# SpeedTalkEncoder
# ---------------------------------------------------------------------------

class SpeedTalkEncoder:
    """
    Maps relation-type strings to single-character phonemes and back.

    Design (Heinlein-inspired)
    --------------------------
    Each unique relation type is assigned one character from _BASE_ALPHABET.
    For KGs with more than 62 relation types, overflow symbols use a
    null-delimited integer notation ``\\x00N\\x00`` — transparent to users
    of encode()/decode().

    Key properties
    --------------
    - encode() · decode() are inverses for any relation sequence
    - Prefix of encoded string ↔ prefix of original sequence
      (each char encodes exactly one relation, so ``s[:k]`` decodes to first k)
    - Thread-safe reads and writes
    - Frequency-order rebalancing: call build_frequency_order() at startup
      to assign shortest symbols to most-used relations (true SpeedTalk spirit)
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rel_to_sym: Dict[str, str] = {}
        self._sym_to_rel: Dict[str, str] = {}
        self._counter: int = 0

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------

    def encode(self, rels: Sequence[str]) -> str:
        """Encode a sequence of relation types to a compact phoneme string."""
        return "".join(self._intern(r) for r in rels)

    def decode(self, encoded: str) -> Tuple[str, ...]:
        """
        Decode a phoneme string back to a tuple of relation types.

        Raises KeyError if *encoded* contains unregistered phonemes.
        """
        if not encoded:
            return ()
        parts = self._split_encoded(encoded)
        with self._lock:
            return tuple(self._sym_to_rel[p] for p in parts)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def vocabulary(self) -> Dict[str, str]:
        """Return a copy of the {relation → symbol} mapping."""
        with self._lock:
            return dict(self._rel_to_sym)

    @property
    def size(self) -> int:
        """Number of distinct relation types registered."""
        with self._lock:
            return self._counter

    # ------------------------------------------------------------------
    # Frequency-order rebalancing (Tier-2 SpeedTalk)
    # ------------------------------------------------------------------

    def build_frequency_order(self, freq: Dict[str, int]) -> None:
        """
        Rebuild symbol assignments so most-common relations get shortest symbols.

        This implements the true Heinlein SpeedTalk principle: the concepts
        used most often in practice receive the most economical representation.

        Call at startup with accumulated relation-frequency data before the
        cache is populated.  Any previously encoded strings are invalidated
        and the cache must be rebuilt.

        Parameters
        ----------
        freq : {relation_type: occurrence_count}
        """
        sorted_rels = sorted(freq.keys(), key=lambda r: freq[r], reverse=True)
        with self._lock:
            self._rel_to_sym.clear()
            self._sym_to_rel.clear()
            self._counter = 0
            for rel in sorted_rels:
                self._assign_next(rel)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise the encoder state for JSON persistence."""
        with self._lock:
            return {"version": 1, "rel_to_sym": dict(self._rel_to_sym)}

    @classmethod
    def from_dict(cls, d: dict) -> "SpeedTalkEncoder":
        """Restore an encoder from a previously serialised dict."""
        enc = cls()
        for rel, sym in d.get("rel_to_sym", {}).items():
            enc._rel_to_sym[rel] = sym
            enc._sym_to_rel[sym] = rel
            enc._counter += 1
        return enc

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _intern(self, rel: str) -> str:
        """Return the symbol for *rel*, assigning a new one if needed."""
        with self._lock:
            if rel not in self._rel_to_sym:
                self._assign_next(rel)
            return self._rel_to_sym[rel]

    def _assign_next(self, rel: str) -> None:
        """Assign the next available symbol to *rel*. Caller holds the lock."""
        if self._counter < len(_BASE_ALPHABET):
            sym = _BASE_ALPHABET[self._counter]
        else:
            sym = f"\x00{self._counter}\x00"
        self._rel_to_sym[rel] = sym
        self._sym_to_rel[sym] = rel
        self._counter += 1

    def _split_encoded(self, encoded: str) -> List[str]:
        """
        Split an encoded string into individual symbol tokens.

        Handles single-char tokens and overflow ``\\x00N\\x00`` tokens.
        """
        tokens: List[str] = []
        i = 0
        while i < len(encoded):
            if encoded[i] == "\x00":
                try:
                    end = encoded.index("\x00", i + 1)
                except ValueError:
                    # Malformed/truncated overflow token — skip remaining bytes
                    break
                tokens.append(encoded[i : end + 1])
                i = end + 1
            else:
                tokens.append(encoded[i])
                i += 1
        return tokens


# ---------------------------------------------------------------------------
# SpeedTalkEngram
# ---------------------------------------------------------------------------

class SpeedTalkEngram:
    """
    Engram relation-pattern cache with Heinlein SpeedTalk phonemic encoding.

    Stores relation sequences as compact phoneme strings rather than verbose
    Python tuples.  The key insight: because each character encodes exactly
    one relation type, a string prefix corresponds exactly to a relation-
    sequence prefix — enabling O(P) prefix queries over all cached patterns.

    Interface mirrors Engram so it can act as a drop-in replacement.

    New methods vs Engram
    ---------------------
    prefix_query(*rels)
        All cached patterns whose sequence starts with the given relation(s).

    alphabet()
        The full {relation_type → phoneme_char} mapping.

    compression_stats()
        Memory-efficiency metrics for the current cache state.

    Parameters
    ----------
    max_patterns : eviction cap (same semantics as Engram)
    encoder      : pre-built SpeedTalkEncoder, or None to create a new one
    """

    def __init__(
        self,
        max_patterns: int = 1000,
        encoder: Optional[SpeedTalkEncoder] = None,
    ) -> None:
        self._lock = threading.Lock()
        self._encoder = encoder or SpeedTalkEncoder()
        self._counts: Dict[str, int] = defaultdict(int)   # encoded_str → count
        self._prefix: Dict[str, int] = defaultdict(int)   # encoded_prefix → count
        self._max_patterns = max_patterns
        self._max_count: int = 1

    # ------------------------------------------------------------------
    # Core operations (mirrors Engram API)
    # ------------------------------------------------------------------

    def record(self, rel_sequence: Sequence[str], weight: int = 1) -> None:
        """
        Record a successful relation sequence (raw relation-type strings).

        The sequence is encoded to phonemes before storage.
        All prefixes of the encoded string are also indexed for fast
        affinity lookup.

        Parameters
        ----------
        rel_sequence : raw relation-type strings, e.g. ("CAUSES", "TREATS")
        weight       : success-count to credit (default 1)
        """
        if not rel_sequence:
            return
        encoded = self._encoder.encode(rel_sequence)
        with self._lock:
            if len(self._counts) >= self._max_patterns:
                min_key = min(self._counts, key=self._counts.__getitem__)
                del self._counts[min_key]
                # Rebuild prefix map on eviction (rare)
                self._prefix.clear()
                for enc, cnt in self._counts.items():
                    for k in range(1, len(enc) + 1):
                        self._prefix[enc[:k]] += cnt

            self._counts[encoded] += weight
            self._max_count = max(self._max_count, self._counts[encoded])
            for k in range(1, len(encoded) + 1):
                self._prefix[encoded[:k]] += weight

    def affinity(self, rel_prefix: Sequence[str]) -> float:
        """
        Return a score in [0, 1] for how well *rel_prefix* matches cached patterns.

        Looks up the longest matching prefix in the index; returns 0.0 if no
        match, 1.0 for the single highest-count pattern.  Identical semantics
        to Engram.affinity() but operates on phoneme-encoded keys.
        """
        if not rel_prefix or self._max_count == 0:
            return 0.0
        encoded = self._encoder.encode(rel_prefix)
        with self._lock:
            for k in range(len(encoded), 0, -1):
                cnt = self._prefix.get(encoded[:k], 0)
                if cnt > 0:
                    depth_bonus = k / len(encoded)
                    return min(1.0, (cnt / self._max_count) * depth_bonus)
        return 0.0

    def size(self) -> int:
        """Return the number of distinct full sequences in the cache."""
        with self._lock:
            return len(self._counts)

    def top_patterns(self, n: int = 10) -> List[Tuple[Tuple[str, ...], int]]:
        """Return the n most frequent patterns as [(decoded_sequence, count)]."""
        with self._lock:
            items = sorted(self._counts.items(), key=lambda x: x[1], reverse=True)
            top = items[:n]
        return [(self._encoder.decode(enc), cnt) for enc, cnt in top]

    def clear(self) -> None:
        with self._lock:
            self._counts.clear()
            self._prefix.clear()
            self._max_count = 1

    # ------------------------------------------------------------------
    # SpeedTalk-specific capabilities
    # ------------------------------------------------------------------

    def prefix_query(self, *rels: str) -> List[Tuple[Tuple[str, ...], int]]:
        """
        Return all cached patterns that start with the given relation sequence.

        The phonemic encoding makes this natural: encoded string prefixes
        correspond exactly to relation-sequence prefixes, so prefix filtering
        is a simple ``str.startswith()`` scan.

        Parameters
        ----------
        *rels : one or more raw relation-type strings forming the query prefix

        Returns
        -------
        list of (decoded_sequence, count) sorted by count descending

        Example
        -------
        >>> cache.record(("CAUSES", "TREATS"), weight=5)
        >>> cache.record(("CAUSES", "INHIBITS", "PREVENTS"), weight=3)
        >>> cache.prefix_query("CAUSES")
        [(('CAUSES', 'TREATS'), 5), (('CAUSES', 'INHIBITS', 'PREVENTS'), 3)]
        """
        if not rels:
            return []
        prefix_enc = self._encoder.encode(rels)
        with self._lock:
            matches = [
                (enc, cnt)
                for enc, cnt in self._counts.items()
                if enc.startswith(prefix_enc)
            ]
        matches.sort(key=lambda x: x[1], reverse=True)
        return [(self._encoder.decode(enc), cnt) for enc, cnt in matches]

    def alphabet(self) -> Dict[str, str]:
        """Return the full {relation_type → phoneme_char} mapping."""
        return self._encoder.vocabulary

    def compression_stats(self) -> dict:
        """
        Return memory-efficiency metrics for the current cache state.

        Keys in the returned dict
        -------------------------
        vocab_size        : number of distinct relation types in the encoder
        total_patterns    : number of cached full sequences
        avg_encoded_len   : mean length of encoded key strings (chars)
        avg_tuple_len     : estimated mean length of equivalent tuple repr (chars)
        compression_ratio : avg_tuple_len / avg_encoded_len
        """
        with self._lock:
            if not self._counts:
                return {
                    "vocab_size": self._encoder.size,
                    "total_patterns": 0,
                    "avg_encoded_len": 0.0,
                    "avg_tuple_len": 0.0,
                    "compression_ratio": 1.0,
                }
            keys = list(self._counts.keys())

        avg_enc = sum(len(k) for k in keys) / len(keys)
        vocab = self._encoder.vocabulary
        avg_rel_name_len = (
            sum(len(r) for r in vocab) / max(1, len(vocab))
        )
        # Tuple repr overhead: quotes + ", " separator + outer parens
        avg_tuple = avg_enc * (avg_rel_name_len + 4)
        return {
            "vocab_size": self._encoder.size,
            "total_patterns": len(keys),
            "avg_encoded_len": round(avg_enc, 2),
            "avg_tuple_len": round(avg_tuple, 2),
            "compression_ratio": round(avg_tuple / max(avg_enc, 0.01), 2),
        }

    # ------------------------------------------------------------------
    # Graph-adaptive encoding
    # ------------------------------------------------------------------

    def adapt_to_graph(self, edge_type_counts: Dict[str, int]) -> None:
        """
        Retune the SpeedTalk alphabet to the relation-type distribution of the
        currently loaded knowledge graph.

        The most-frequent relation types in *this* graph are assigned the shortest
        phoneme symbols, so the cache is maximally compressed for the vocabulary
        that actually appears in the active KG.

        If the cache already contains stored patterns (e.g. loaded from disk),
        they are transparently re-encoded to the new symbol assignments — no
        patterns are lost.

        Call this immediately after loading a new graph, before any queries run.

        Parameters
        ----------
        edge_type_counts : {relation_type: occurrence_count}
            Typically obtained from :meth:`count_edge_types` or by iterating
            the graph adapter directly.

        Example
        -------
        >>> counts = SpeedTalkEngram.count_edge_types(adapter)
        >>> cache.adapt_to_graph(counts)
        # Now cache.alphabet() reflects the graph's relation distribution.
        """
        if not edge_type_counts:
            return

        # Snapshot current patterns (decoded, so they survive re-encoding)
        with self._lock:
            snapshot = [
                (self._encoder.decode(enc), cnt)
                for enc, cnt in self._counts.items()
            ]

        # Rebuild encoder in frequency order for this graph
        self._encoder.build_frequency_order(edge_type_counts)
        _log.info(
            "SpeedTalkEngram: alphabet rebuilt for %d relation types "
            "(top 3: %s)",
            len(edge_type_counts),
            ", ".join(
                f"{r}→'{self._encoder.encode([r])}'"
                for r in sorted(
                    edge_type_counts, key=edge_type_counts.__getitem__, reverse=True
                )[:3]
                if r in self._encoder.vocabulary
            ),
        )

        # Re-encode all existing patterns under the new alphabet
        with self._lock:
            self._counts.clear()
            self._prefix.clear()
            self._max_count = 1

        for seq, cnt in snapshot:
            # seq may contain relation types not in the new graph's vocab —
            # they will be assigned new symbols on-demand via _intern()
            self.record(seq, weight=cnt)

    @classmethod
    def from_graph_adapter(
        cls,
        adapter,
        max_patterns: int = 1000,
    ) -> "SpeedTalkEngram":
        """
        Create a new cache whose alphabet is pre-tuned to *adapter*'s edge types.

        Reads all edges from the adapter, counts relation-type frequencies, and
        builds a frequency-ordered encoder so the most-traversed relation types
        get the shortest symbols.

        Parameters
        ----------
        adapter     : any GraphAdapter implementation
        max_patterns: eviction cap passed through to the new cache

        Returns
        -------
        A fresh SpeedTalkEngram with an adapter-tuned encoder and no stored
        patterns (ready to accumulate patterns from live queries).
        """
        freq = cls.count_edge_types(adapter)
        encoder = SpeedTalkEncoder()
        if freq:
            encoder.build_frequency_order(freq)
        cache = cls(max_patterns=max_patterns, encoder=encoder)
        _log.info(
            "SpeedTalkEngram.from_graph_adapter: %d relation types indexed",
            len(freq),
        )
        return cache

    @staticmethod
    def count_edge_types(adapter) -> Dict[str, int]:
        """
        Count relation-type frequencies from a graph adapter.

        Iterates all edges returned by ``adapter.get_all_edges()`` (if available)
        or falls back to ``adapter.edges`` attribute.  Relation type is read from
        ``Edge.relation_type``.

        Parameters
        ----------
        adapter : GraphAdapter — must expose edges via get_all_edges() or .edges

        Returns
        -------
        {relation_type: count}  — empty dict if the adapter exposes no edges
        """
        from collections import Counter
        counts: Counter = Counter()

        # Try get_all_edges() first (standard adapter interface)
        get_all = getattr(adapter, "get_all_edges", None)
        if callable(get_all):
            for edge in get_all():
                counts[edge.relation_type] += 1
            return dict(counts)

        # Fallback: .edges attribute (some adapters expose this)
        edges_attr = getattr(adapter, "edges", None)
        if edges_attr is not None:
            for edge in edges_attr:
                counts[edge.relation_type] += 1
            return dict(counts)

        _log.warning(
            "SpeedTalkEngram.count_edge_types: adapter has no get_all_edges() "
            "or .edges — returning empty frequency map"
        )
        return {}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Persist the cache (counts + encoder) to a JSON file.

        The prefix index is derived and not stored — it is rebuilt on load.

        Parameters
        ----------
        path : file path to write (parent directories are created if needed)
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            data = {
                "version": 2,
                "max_patterns": self._max_patterns,
                "encoder": self._encoder.to_dict(),
                "counts": [[enc, cnt] for enc, cnt in self._counts.items()],
            }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        _log.info(
            "SpeedTalkEngram saved: %d patterns → %s", len(data["counts"]), p
        )

    @classmethod
    def load(cls, path: str) -> "SpeedTalkEngram":
        """
        Load a previously saved cache.

        Returns an empty cache if the file does not exist.
        The prefix index is rebuilt from the stored counts.
        """
        p = Path(path)
        if not p.exists():
            _log.info("SpeedTalkEngram: no file at %s — starting empty", p)
            return cls()
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        encoder = SpeedTalkEncoder.from_dict(data.get("encoder", {}))
        cache = cls(max_patterns=data.get("max_patterns", 1000), encoder=encoder)
        for enc, cnt in data.get("counts", []):
            cache._counts[enc] = cnt
            cache._max_count = max(cache._max_count, cnt)
            for k in range(1, len(enc) + 1):
                cache._prefix[enc[:k]] += cnt
        _log.info(
            "SpeedTalkEngram loaded: %d patterns ← %s", len(cache._counts), p
        )
        return cache

    def save_if_path(self, path: Optional[str]) -> None:
        """Convenience: call save(path) only if path is not None."""
        if path:
            self.save(path)


# ---------------------------------------------------------------------------
# Phase 218-A: Cross-KB Engram Transfer Registry
# ---------------------------------------------------------------------------

class EngramTransferRegistry:
    """
    Phase 218-A: Carry successful reasoning patterns across knowledge bases.

    When switching to a new KB, the registry re-encodes patterns from
    previously seen KBs into the new KB's relation vocabulary and merges
    them into the initial Engram with a configurable count decay so that
    transferred patterns are trusted less than locally observed ones.

    This closes the "every KB starts from zero" gap — CEREBRUM can seed
    its Engram cache with structural priors from prior KBs on first launch.

    Usage
    -----
    registry = EngramTransferRegistry()
    # After finishing queries on KB-A:
    registry.register("metaqa", engram_metaqa)
    # When loading KB-B:
    new_engram = registry.transfer_to(target_encoder=encoder_kb_b, decay=0.5)
    """

    def __init__(self) -> None:
        self._engrams: Dict[str, "SpeedTalkEngram"] = {}

    def register(self, kb_id: str, engram: "SpeedTalkEngram") -> None:
        """Register an Engram trained on kb_id for later transfer."""
        self._engrams[kb_id] = engram
        _log.info("EngramTransferRegistry: registered '%s' (%d patterns)", kb_id, engram.size())

    def transfer_to(
        self,
        target_encoder: "SpeedTalkEncoder",
        source_kb_ids: Optional[list] = None,
        decay: float = 0.5,
        max_patterns: int = 1000,
    ) -> "SpeedTalkEngram":
        """
        Build a new SpeedTalkEngram pre-populated with patterns from registered KBs,
        re-encoded for the target KB's vocabulary and decayed by *decay*.

        Parameters
        ----------
        target_encoder : SpeedTalkEncoder already adapted to the target KB
        source_kb_ids  : which KB ids to pull from (None = all registered)
        decay          : multiply transferred counts by this factor [0, 1]
                         (default 0.5: transferred patterns start at half weight)
        max_patterns   : maximum patterns in the merged Engram
        """
        merged = SpeedTalkEngram(max_patterns=max_patterns, encoder=target_encoder)
        sources = source_kb_ids or list(self._engrams.keys())
        total_transferred = 0
        for kb_id in sources:
            src = self._engrams.get(kb_id)
            if src is None:
                _log.warning("EngramTransferRegistry: unknown kb_id '%s'", kb_id)
                continue
            # Decode all patterns from source, re-encode in target vocab, apply decay
            for enc_seq, cnt in src._counts.items():
                rel_seq = src._encoder.decode(enc_seq)
                decayed_count = max(1, int(cnt * decay))
                merged.record(rel_seq, weight=decayed_count)
                total_transferred += 1

        _log.info(
            "EngramTransferRegistry: transferred %d patterns from %s (decay=%.2f)",
            total_transferred, sources, decay,
        )
        return merged

    def save(self, path: str) -> None:
        """Persist all registered engrams to a directory."""
        import os
        os.makedirs(path, exist_ok=True)
        for kb_id, engram in self._engrams.items():
            safe_id = kb_id.replace("/", "_").replace("\\", "_")
            engram.save(f"{path}/{safe_id}.json")
        _log.info("EngramTransferRegistry: saved %d engrams to %s", len(self._engrams), path)

    @classmethod
    def load(cls, path: str) -> "EngramTransferRegistry":
        """Load all engrams from a previously saved directory."""
        import os
        registry = cls()
        if not os.path.isdir(path):
            return registry
        for fname in os.listdir(path):
            if fname.endswith(".json"):
                kb_id = fname[:-5]
                fpath = os.path.join(path, fname)
                try:
                    registry._engrams[kb_id] = SpeedTalkEngram.load(fpath)
                    _log.info("EngramTransferRegistry: loaded '%s'", kb_id)
                except Exception as e:
                    _log.warning("EngramTransferRegistry: failed to load '%s': %s", fpath, e)
        return registry


# ---------------------------------------------------------------------------
# Phase 219-A: Fast Binding Engine (one-shot episodic encoding)
# ---------------------------------------------------------------------------

class FastBindingEngine:
    """
    Hippocampal one-shot episodic encoding analog.

    When a path is both novel (low existing affinity) and high-confidence
    (score > score_threshold), it is bound directly into the Engram with
    a higher initial count — equivalent to a single salient experience
    encoding into episodic memory.

    Parameters
    ----------
    engram          : The SpeedTalkEngram (or Engram) to bind patterns into.
    novelty_threshold : Affinity below which a path is considered novel.
    score_threshold   : Minimum traversal score to qualify for fast binding.
    fast_weight       : Initial count for fast-bound patterns (equiv. to N observations).
    """

    def __init__(
        self,
        engram: "SpeedTalkEngram",
        novelty_threshold: float = 0.1,
        score_threshold: float = 0.7,
        fast_weight: int = 5,
    ) -> None:
        self.engram = engram
        self.novelty_threshold = novelty_threshold
        self.score_threshold = score_threshold
        self.fast_weight = fast_weight

    def evaluate(self, rel_seq: Tuple[str, ...], existing_affinity: float, path_score: float) -> bool:
        """Return True if this path qualifies for fast binding."""
        return (
            len(rel_seq) > 0
            and existing_affinity < self.novelty_threshold
            and path_score >= self.score_threshold
        )

    def bind(self, rel_seq: Tuple[str, ...]) -> None:
        """Directly encode rel_seq into the Engram at fast_weight count."""
        self.engram.record(rel_seq, weight=self.fast_weight)
        _log.debug("FastBinding: encoded %s at weight=%d", rel_seq, self.fast_weight)


# ---------------------------------------------------------------------------
# Path helper — raw relation extraction (no Engram shorthand)
# ---------------------------------------------------------------------------

def _raw_rel_sequence(path) -> Tuple[str, ...]:
    """
    Extract raw relation-type strings from a TraversalPath.

    TraversalPath.nodes is [entity, rel, entity, rel, entity, ...].
    Odd-indexed elements are relation types.

    Unlike _path_rel_sequence() in engram_traversal.py, this does NOT
    apply the Engram shorthand map — SpeedTalkEncoder handles its own encoding.
    """
    nodes = path.nodes
    return tuple(nodes[i] for i in range(1, len(nodes), 2))


# ---------------------------------------------------------------------------
# SpeedTalkEngramTraversal
# ---------------------------------------------------------------------------

# Deferred imports to avoid circular dependency
from reasoning.traversal import BeamTraversal, TraversalPath  # noqa: E402
from core.graph_adapter import GraphAdapter  # noqa: E402
from core.attention_engine import CSAEngine  # noqa: E402


class SpeedTalkEngramTraversal(BeamTraversal):
    """
    Beam traversal using SpeedTalkEngram for relation-pattern guidance.

    Identical beam-search logic to EngramTraversal but:
    - Stores relation sequences using SpeedTalk phonemic encoding
    - Exposes prefix_query() on the cache for post-traversal analysis
    - Uses raw relation-type names (not Engram shorthands) as cache keys

    Boost formula (same as EngramTraversal):
        effective_score = path.score * (1 + engram_strength * affinity)

    Parameters
    ----------
    cache            : SpeedTalkEngram instance (shared across queries)
    engram_strength  : multiplicative boost ceiling (default 0.3 → max 30% boost)
    All other parameters are forwarded to BeamTraversal.
    """

    def __init__(
        self,
        *args,
        cache: Optional[SpeedTalkEngram] = None,
        engram_strength: float = 0.3,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.cache = cache or SpeedTalkEngram()
        self.engram_strength = engram_strength

    def _boosted_score(self, path: TraversalPath) -> float:
        """Compute the SpeedTalk-affinity-boosted effective score."""
        rel_seq = _raw_rel_sequence(path)
        if not rel_seq:
            return path.score
        aff = self.cache.affinity(rel_seq)
        return path.score * (1.0 + self.engram_strength * aff)

    def _prune_candidates(
        self,
        candidates: List[TraversalPath],
        hop: int,
    ) -> List[TraversalPath]:
        """
        SpeedTalk-steered pruning: rank by boosted score instead of raw score.

        Candidates whose relation-sequence prefix matches a cached successful
        pattern receive a multiplicative boost; the rest are pruned on raw CSA
        score.  At the terminal hop all candidates are returned (sorted) so the
        answer extractor sees the full frontier.
        """
        import heapq
        hop_bw = self._beam_widths.get(hop, self.beam_width)
        if hop < self.max_hop and len(candidates) > hop_bw:
            return heapq.nlargest(hop_bw, candidates, key=self._boosted_score)
        return sorted(candidates, key=self._boosted_score, reverse=True)

    def record_answers(self, answers, min_score: float = 0.3) -> None:
        """
        Feed successful answers back into the SpeedTalkEngram.

        Call after each query to accumulate relation patterns.  Higher-scoring
        paths receive proportionally larger weight credits.

        Parameters
        ----------
        answers   : iterable of Answer objects (from reasoning.answer_extractor)
        min_score : only record answers with score >= this threshold
        """
        for ans in answers:
            if ans.score < min_score:
                continue
            path = getattr(ans, "best_path", None)
            if path is None:
                continue
            rel_seq = _raw_rel_sequence(path)
            if rel_seq:
                weight = max(1, int(ans.score * 10))
                self.cache.record(rel_seq, weight=weight)


