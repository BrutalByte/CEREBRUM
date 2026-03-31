"""
Relation Path Frequency Prior — Phase 27B

Learns which relation-sequence patterns are productive for multi-hop QA by
counting how often each pattern appears in correct beam paths.  At query time,
paths whose relation sequence matches a high-frequency pattern receive a score
bonus — equivalent to what a trained RL policy learns implicitly, but achieved
with a single counting pass over training data and zero gradient descent.

Two variants:

  RelationPathPrior
    Learns from (paths, correct_answer_set) pairs — requires a training split
    or the current question's answers (for online accumulation).

  GraphRelationPrior
    Structural fallback: scores relation sequences by the co-occurrence
    frequency of each relation type in the graph.  No QA labels needed.
    Weaker signal but always available.

Usage
-----
  # Build from training questions (offline):
  prior = RelationPathPrior()
  for seed, answers in train_qa:
      paths = traversal.traverse([seed])
      prior.update(paths, set(answers))
  prior.freeze()

  # Score a path at query time:
  bonus = prior.score(path)   # float in [0, 1]

  # Pass to extract() for re-ranking:
  answers = extract(paths, relation_prior=prior, ...)
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _rel_sequence(path) -> Tuple[str, ...]:
    """
    Extract the relation sequence from a TraversalPath.

    TraversalPath.nodes alternates [entity, relation, entity, ...].
    Odd indices (1, 3, 5, ...) are relation labels.
    """
    nodes = getattr(path, "nodes", [])
    return tuple(nodes[i] for i in range(1, len(nodes), 2))


# ---------------------------------------------------------------------------
# RelationPathPrior — learned from QA (path, correct_answers) pairs
# ---------------------------------------------------------------------------

class RelationPathPrior:
    """
    Frequency-based relation path prior learned from QA traversal data.

    Tracks how often each relation sequence appears in paths that reach a
    correct answer vs. any path.  Score = smoothed success rate for the
    sequence, normalised to [0, 1].

    Parameters
    ----------
    smoothing : float
        Laplace smoothing constant (default 1.0).
    max_len : int
        Maximum relation sequence length to track.  Longer sequences are
        truncated to this length for generalisation.  Default 3.
    min_count : int
        Minimum total observations before a sequence gets a non-trivial
        score.  Sequences seen fewer times fall back to the global base rate.
        Default 3.
    """

    def __init__(
        self,
        smoothing: float = 1.0,
        max_len: int = 3,
        min_count: int = 3,
    ) -> None:
        self.smoothing  = smoothing
        self.max_len    = max_len
        self.min_count  = min_count
        self._hits:  Counter = Counter()   # seq -> correct-path count
        self._total: Counter = Counter()   # seq -> all-path count
        self._global_hits  = 0
        self._global_total = 0
        self._frozen = False

    # ------------------------------------------------------------------
    # Building the prior
    # ------------------------------------------------------------------

    def update(self, paths: List[Any], correct_entities: Set[str]) -> None:
        """
        Update counts from one query's traversal results.

        Parameters
        ----------
        paths            : list of TraversalPath from BeamTraversal.traverse()
        correct_entities : set of ground-truth answer entity IDs
        """
        if self._frozen:
            raise RuntimeError("RelationPathPrior is frozen — call unfreeze() first.")
        for path in paths:
            if path.hop_depth < 1:
                continue
            seq = _rel_sequence(path)[:self.max_len]
            if not seq:
                continue
            is_hit = path.tail in correct_entities
            self._total[seq] += 1
            self._global_total += 1
            if is_hit:
                self._hits[seq] += 1
                self._global_hits += 1

    def freeze(self) -> "RelationPathPrior":
        """Lock counts; pre-compute normalisation stats."""
        self._frozen = True
        # Pre-compute max smoothed score for normalisation
        if self._global_total > 0:
            self._base_rate = (self._global_hits + self.smoothing) / (
                self._global_total + 2 * self.smoothing
            )
        else:
            self._base_rate = 0.5
        return self

    def unfreeze(self) -> "RelationPathPrior":
        self._frozen = False
        return self

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, path: Any) -> float:
        """
        Score a path by the success rate of its relation sequence.

        Returns float in [0, 1].  Unseen or low-count sequences return the
        global base rate, providing a smooth floor.
        """
        seq = _rel_sequence(path)[:self.max_len]
        if not seq:
            return self._base_rate if self._frozen else 0.5

        total = self._total[seq]
        if total < self.min_count:
            return self._base_rate if self._frozen else 0.5

        hits = self._hits[seq]
        smoothed = (hits + self.smoothing) / (total + 2 * self.smoothing)
        return float(np.clip(smoothed, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Prefix generalisation — score by longest matching prefix
    # ------------------------------------------------------------------

    def score_with_prefix(self, path: Any) -> float:
        """
        Like score(), but falls back to shorter prefix sequences if the full
        sequence is unseen.  Improves generalisation on unseen relation paths.
        """
        seq = _rel_sequence(path)[:self.max_len]
        for length in range(len(seq), 0, -1):
            prefix = seq[:length]
            if self._total[prefix] >= self.min_count:
                hits  = self._hits[prefix]
                total = self._total[prefix]
                return float(np.clip(
                    (hits + self.smoothing) / (total + 2 * self.smoothing),
                    0.0, 1.0,
                ))
        return self._base_rate if self._frozen else 0.5

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def top_sequences(self, n: int = 20) -> List[Tuple[Tuple, float, int]]:
        """
        Return the top-N sequences by smoothed success rate.

        Returns list of (sequence, success_rate, total_count).
        Only includes sequences with total >= min_count.
        """
        results = []
        for seq, total in self._total.items():
            if total < self.min_count:
                continue
            hits = self._hits[seq]
            rate = (hits + self.smoothing) / (total + 2 * self.smoothing)
            results.append((seq, rate, total))
        results.sort(key=lambda x: -x[1])
        return results[:n]

    def __repr__(self) -> str:
        n_seqs = sum(1 for t in self._total.values() if t >= self.min_count)
        return (
            f"RelationPathPrior("
            f"sequences={n_seqs}, "
            f"total_paths={self._global_total}, "
            f"base_rate={getattr(self, '_base_rate', '?'):.3f}, "
            f"frozen={self._frozen})"
        )


# ---------------------------------------------------------------------------
# GraphRelationPrior — structural fallback (no QA labels needed)
# ---------------------------------------------------------------------------

class GraphRelationPrior:
    """
    Structural prior built from relation co-occurrence in the graph.

    Scores a relation sequence by the log-normalised frequency of each
    relation type in the graph.  High-frequency relations are assumed to be
    more topically relevant (they tend to connect semantically close entities).

    No QA training data required.  Weaker signal than RelationPathPrior but
    always available as a baseline.

    Parameters
    ----------
    decay : float
        Multiplicative decay applied at each hop (default 0.8).
        Later hops contribute less to avoid over-weighting long paths.
    """

    def __init__(self, decay: float = 0.8) -> None:
        self.decay = decay
        self._rel_score: Dict[str, float] = {}

    def fit(self, adapter) -> "GraphRelationPrior":
        """
        Build relation scores from a graph adapter.

        Scores each relation type by log-normalised edge count.
        """
        counts: Counter = Counter()
        G = getattr(adapter, "_G", None) or getattr(adapter, "G", None)
        if G is None:
            return self

        for _, _, data in G.edges(data=True):
            rel = data.get("relation", "")
            if rel:
                counts[rel] += 1

        if not counts:
            return self

        max_count = max(counts.values())
        log_max   = math.log1p(max_count)
        self._rel_score = {
            rel: math.log1p(cnt) / log_max
            for rel, cnt in counts.items()
        }
        return self

    def score(self, path: Any) -> float:
        """Score a path by the geometric mean of its relation frequencies."""
        seq = _rel_sequence(path)
        if not seq:
            return 0.5
        scores = []
        decay  = 1.0
        for rel in seq:
            s = self._rel_score.get(rel, 0.1)
            scores.append(s * decay)
            decay *= self.decay

        return float(np.clip(
            sum(scores) / max(len(scores), 1),
            0.0, 1.0,
        ))

    def __repr__(self) -> str:
        return f"GraphRelationPrior(relations={len(self._rel_score)}, decay={self.decay})"
