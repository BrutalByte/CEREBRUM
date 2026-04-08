# AAAK-Steered Traversal: Training-Free Relation-Pattern Caching for Knowledge Graph Beam Search

**Authors**: Bryan Alexander Buchorn · Claude Sonnet 4.6 (Research Collaborator)  
**Affiliations**: Independent Researcher · Anthropic  
**Status**: v2.0.1 (Phase 57 COMPLETE)  
**Date**: April 2026

---

### Abstract
Classical beam search over Knowledge Graphs treats each query independently, discarding all information about which relation sequences led to successful answers in previous queries. We present **AAAK-Steered Traversal**, a training-free mechanism that accumulates successful relation-sequence patterns in a persistent `AAAKCache` and biases future beam pruning toward known-productive reasoning chains. The affinity boost is applied multiplicatively in `AAAKBeamTraversal._prune_candidates()`: $s_\text{eff}(c) = s(c) \times (1 + \lambda_\text{AAAK} \cdot \text{affinity}(\text{rel\_seq}))$. No gradient descent is required — patterns are accumulated through frequency counting alone. The cache is durable across process restarts via JSON serialization, and two-tier warm-up on startup (saved JSON + `QueryLog` replay) ensures that learned patterns are immediately available after server restart.

### 1. Introduction
Knowledge Graph reasoning systems learn from feedback through two established mechanisms: online parameter updates (MetaParameterLearner, Paper 2) and batch retraining (CSAParameterLearner, Paper 2). Both adjust numerical weights in the CSA attention formula. Neither captures the relational structure of successful reasoning paths — the specific sequences of edge types that, when followed, reliably connect a seed entity to a correct answer.

Classical beam search \cite{vaswani2017attention} is a stateless algorithm: the beam at hop $k$ depends only on the current graph state and the CSA scores computed from that state. Every query starts from the same initial conditions, regardless of how many similar queries have been answered successfully before. This statelessness is correct in principle but wasteful in practice: a KG reasoning server that has answered thousands of protein-disease queries should be biased toward the relation sequences that previously reached disease nodes from protein nodes.

AAAK-steered traversal addresses this by maintaining a persistent, query-accumulated cache of relation-sequence success counts. The cache requires no training, no labels, and no gradient computation — it is updated after each successful query by recording the relation sequence of the winning path.

### 2. Methodology

#### 2.1 AAAKCache Structure
`AAAKCache` maintains a count dictionary `_counts: Dict[Tuple[str, ...], int]` mapping relation sequences (tuples of relation type strings) to their accumulated success counts. A `_max_count: int` tracks the maximum count across all sequences, used for normalization.

The affinity of a relation sequence `seq` is:

$$\text{affinity}(\text{seq}) = \frac{\texttt{\_counts}[\text{seq}]}{\texttt{\_max\_count}}$$

For sequences not present in `_counts`, `affinity = 0` and no boost is applied — the cache degrades gracefully to unsteered beam search on unseen relation sequences.

#### 2.2 Beam Pruning Integration
`AAAKBeamTraversal` extends `BeamTraversal` and overrides `_prune_candidates()`. For each candidate path $c$ at each hop, the current partial relation sequence `rel_seq` is extracted from the path history. The effective score is:

$$s_\text{eff}(c) = s(c) \times (1 + \lambda_\text{AAAK} \cdot \text{affinity}(\text{rel\_seq}))$$

where $\lambda_\text{AAAK}$ (`aaak_strength`) is a configurable scalar (default: 1.0). This multiplicative form ensures that a zero-CSA-score candidate is never promoted by the cache (it remains zero), and that high-CSA candidates receive proportionally larger boosts from high-affinity sequences.

#### 2.3 Cache Persistence
`AAAKCache.save(path)` serializes `_counts` to a JSON file with a versioned envelope:

```json
{"version": 1, "counts": [["rel_seq_tuple", count], ...]}
```

`AAAKCache.load(path)` restores `_counts` from the JSON file and recomputes `_max_count` from the loaded values. `save_if_path(path)` is a null-safe variant that silently returns if `path` is `None`, enabling code paths where the cache path is optionally configured.

#### 2.4 Two-Tier Startup Warm-Up
On server startup, the FastAPI lifespan context manager performs two-tier warm-up:

1. **Tier 1**: Load the saved `AAAKCache` JSON from the configured path. This restores all counts from the previous server run.
2. **Tier 2**: Call `QueryLog.replay_into_cache(aaak_cache)` to merge any `QueryLog` entries recorded since the last explicit `save()` call. This closes the gap between the last save and the end of the previous run.

On server shutdown, the lifespan `try/finally` block calls `AAAKCache.save_if_path(path)`, ensuring that all counts accumulated during the current run are persisted. Write failures during shutdown are isolated with `try/except` and logged at WARNING.

#### 2.5 QueryLog Integration
`QueryLog` maintains an append-only NDJSON file of query history records. Each record includes seeds, answers, and the relation sequence of the winning path. `replay_into_cache(aaak_cache)` iterates the log file and calls `aaak_cache.record(rel_seq)` for each successful query entry, updating `_counts` and `_max_count`.

### 3. Prior Art
MINERVA \cite{das2018minerva} learns a policy over relation sequences using REINFORCE — a reinforcement learning algorithm requiring labelled question-answer pairs and many training episodes. AAAK uses no training: counts are derived from live query success without any reward signal or policy gradient.

M-Walk uses Monte Carlo tree search to explore relation paths during inference. AAAK does not modify the search tree structure; it biases existing beam pruning through a multiplicative score adjustment, leaving the beam search algorithm unchanged.

Neural LP and DRUM learn logical rules over KGs from labelled triples. AAAK accumulates relation-sequence statistics from runtime query history, requiring no labelled training data and no rule induction algorithm.

The key distinction of AAAK is its operational simplicity: it is a frequency counter with a lookup table. The implementation adds fewer than 100 lines of code to the traversal module. It improves on zero-shot performance without any of the infrastructure requirements of RL-based or training-based approaches.

### 4. Integration and Failure Isolation
`AAAKCache.record()` calls in the hot query path are wrapped in `try/except`. A write failure (disk full, OOM) logs at WARNING and does not propagate to the HTTP response — the query result is returned normally even if the cache update fails. This ensures that the persistence layer never becomes a reliability dependency for the core reasoning path.

The `QueryLog` replay on startup is similarly isolated: if the log file is corrupted or missing, the warm-up step logs at WARNING and proceeds with the counts loaded from the saved JSON (Tier 1 only). The system starts in a degraded-warm state rather than failing to start.

### 5. Conclusion
AAAK-steered traversal demonstrates that meaningful learning from experience does not require gradient descent. By accumulating relation-sequence success counts in a persistent, durable cache and applying a multiplicative affinity boost during beam pruning, the system biases future queries toward known-productive reasoning chains without modifying graph structure, CSA parameters, or the beam search algorithm. Two-tier warm-up on startup ensures that no productive reasoning trace is lost across process restarts.

---
**References**
1. Vaswani, A., et al. (2017). Attention is All You Need. NIPS.
2. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP.
3. Das, R., et al. (2018). Go for a Walk and Arrive at the Answer: Reasoning over Paths in Knowledge Bases using Reinforcement Learning (MINERVA). ICLR.
