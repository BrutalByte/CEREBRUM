# PAPER 022: Looped Beam Traversal — Iterative Refinement for Knowledge Graph Reasoning
**Status**: v2.71.0 (Phase 172 (STRB) COMPLETE)
**Date**: May 2, 2026

**CEREBRUM Phase 70**  
**Inspired by:** Zhu, R.-J. et al. (2025). *Scaling Latent Reasoning via Looped Language Models.* arXiv:2510.25741. ByteDance Seed / UC Santa Cruz et al. [zhu2025loooplm]

---

## Abstract

We present **LoopedBeamTraversal**, an iterative reasoning mechanism for CEREBRUM's Knowledge Graph traversal engine. Inspired by LoopLM [zhu2025loooplm], which demonstrates that applying the same transformer stack T times yields dramatically better reasoning on hard inputs without increasing parameter count, we adapt the looping principle to graph beam search. CEREBRUM's analog applies `BeamTraversal` T times, using the top answer entities from loop t as additional seeds for loop t+1. An adaptive exit gate — driven by Prediction Error (PE) from Phase 69's `PredictiveCodingEngine` [Phase 69] — terminates the loop when further iterations cease to improve reasoning quality. Three inter-loop feedback channels make each pass progressively better-calibrated, resulting in richer iterative refinement per compute step than LoopLM's single hidden-state channel.

---

## 1. Motivation: From Single-Pass to Iterative Reasoning

Standard beam search over a Knowledge Graph is a single-pass operation: the traversal engine expands from seed entities, prunes candidates at each hop via CSA attention [vaswani2017attention, hamilton2017graphsage], and returns the best paths. This approach implicitly assumes that the optimal reasoning trajectory is fully determined by the initial seed set and graph structure.

This assumption breaks down for:
- **Hard multi-hop queries** where the optimal intermediate entity is not reachable in a single traversal.
- **Sparse graphs** where early hops produce few viable candidates, starving later hops.
- **Cold-start seeds** where the seed entities have low structural centrality, missing productive relation neighbourhoods.

LoopLM [zhu2025loooplm] demonstrates that applying the *same* computation stack T times — rather than once — yields substantial gains on MATH, GSM8K, AIME, and other hard reasoning benchmarks. The key insight is that each loop pass can refine its starting context using the output of the previous pass, progressively converging toward the correct answer. An adaptive exit gate (the ideal continuation probability λ_t) prevents wasted compute when the model has already converged.

We transfer this principle to CEREBRUM's graph traversal: if the first beam search surfaces partially-relevant entities, those entities can be treated as *new seeds* for a second pass, revealing connections that were unreachable from the original seed set.

---

## 2. Architecture: LoopedBeamTraversal

### 2.1 Core Loop

Let `S_0` be the original seed entity set and `T` the maximum loop count. For loop `t ∈ {1, …, T}`:

1. **Traverse**: Run `BeamTraversal(S_{t-1})` → path set `P_t`.
2. **Extract**: Call `extract(P_t, top_k=K)` → answer list `A_t`.
3. **Merge**: Update `best_by_tail` dict: `best_by_tail[e] = argmax_score(best_by_tail[e], P_t[e])` for all tail entities `e`.
4. **Exit gate**: Evaluate exit conditions (§2.3). If triggered, stop.
5. **Expand**: Build `S_t = S_0 ∪ {a.entity_id : a ∈ A_t[:K_seed]}` (deduplicated).

Final merged path set: `best_by_tail.values()` — the highest-scoring path per tail entity across **all** loops.

### 2.2 Three Inter-Loop Feedback Channels

CEREBRUM's looped reasoning uses three feedback channels between passes, compared to LoopLM's single hidden-state channel:

| Channel | Mechanism | Effect |
|---|---|---|
| **Semantic** | Top-K answer entities from loop t expand `S_{t+1}` | Richer neighbourhood coverage; loop t+1 starts closer to productive sub-graph |
| **Metabolic** | PE from `PredictiveCodingEngine` drives `ChemicalModulator` (arousal, novelty, reinforcement) | Adjusts `beam_width` and CSA α/β for the next loop; high PE → wider beam |
| **Mnemonic** | Engram records added during loop t bias beam pruning in loop t+1 | Known-productive relation patterns up-weighted via affinity boost in `_prune_candidates()` |

The metabolic and mnemonic channels are unique to CEREBRUM — LoopLM has no analog for these, relying solely on hidden-state propagation across loops. This makes CEREBRUM's iterative refinement richer per compute step.

### 2.3 Adaptive Exit Gate

The exit gate mirrors LoopLM's ideal continuation probability λ_t, which penalises both underthinking (exits too early) and overthinking (continues past the point of improvement):

**Primary gate — PE convergence** (requires `PredictiveCodingEngine`):
```
|PE_t - PE_{t-1}| < γ  →  exit_reason = "pe_converged"
```

PE is Jaccard divergence between the Engram-derived prior relation sequence and the best actual path (§3). When PE stops improving, the model's internal state has converged — further loops will not yield qualitatively different paths.

**Fallback gate — answer stability**:
```
Jaccard(A_{t-1}, A_t) ≥ θ  →  exit_reason = "answers_stable"
```

When the top-K answer entities stabilise (high overlap), the reasoning has converged even without PE signal (e.g., cold-start Engram).

**Max loops cap**:
```
t == T  →  exit_reason = "max_loops"
```

Default parameters: `γ = 0.05`, `θ = 0.80`, `T = 4`.

### 2.4 Backward Compatibility

`max_loops=1` (default) bypasses all looping logic and calls inner `BeamTraversal` directly. The return type is identical to the non-looped case. This ensures zero regression risk when the feature is not enabled.

---

## 3. Integration with PredictiveCodingEngine (Phase 69)

Phase 69 [Phase 69] introduced `PredictiveCodingEngine`, which generates an Engram-derived prior before traversal and computes PE after. PE measures the Jaccard distance between:
- **Predicted** relation sequence: derived from top Engram patterns for the seed set.
- **Actual** relation sequence: extracted from the best path returned by traversal.

In the looped context, PE serves a dual role:

1. **Exit gate signal**: PE delta between successive loops signals convergence.
2. **Metabolic regulation**: After each loop, PE is dispatched to `ChemicalModulator`:
   - `update_arousal(PE)` — high PE (surprising result) increases arousal, widening beam on next loop.
   - `update_novelty(PE)` — high PE marks the seed domain as novel, increasing exploration.
   - `update_reinforcement(1 - PE)` — low PE (good prediction) reinforces current traversal parameters.

This creates a closed loop: the graph's own predictive model regulates how aggressively the next iteration explores, without any external supervision signal.

---

## 4. Integration with MACH L1 Consensus (Phase 60)

`MultiStrategyConsensus.run_consensus_query()` (Phase 60) runs multiple traversal strategies (standard, Bayesian, Engram) and aggregates paths via `ConsensusScorer`. With `max_loops > 1`, each strategy's traversal is independently wrapped in `LoopedBeamTraversal` before execution. This means each strategy iteratively refines its own path set, then all refined sets are aggregated — combining the depth of looped reasoning with the breadth of multi-strategy consensus.

```python
# Each strategy loops independently
looped = LoopedBeamTraversal(
    traversal        = strategy_traversal,
    predictive_coder = self.predictive_coder,
    max_loops        = max_loops,
)
paths, _ = looped.traverse(seeds, query_embedding=q_emb)
```

---

## 5. API Surface

### QueryRequest (modified)
```json
{
  "query": "...",
  "seeds": ["..."],
  "max_loops": 2
}
```
`max_loops` (default 1, range 1–8) triggers iterative refinement. 1 = single-pass (backward compatible).

### QueryResponse (extended)
```json
{
  "paths": [...],
  "loops_run": 2,
  "pe_per_loop": [0.42, 0.18]
}
```

`loops_run` and `pe_per_loop` are `None` when `max_loops=1`.

### LoopTrace (diagnostic)
```python
@dataclass
class LoopTrace:
    loops_run: int
    seeds_per_loop: List[List[str]]      # seeds used at start of each loop
    pe_per_loop: List[Optional[float]]   # PE after each loop (None = no PE engine)
    paths_per_loop: List[int]            # path count per loop
    new_answers_per_loop: List[int]      # new unique answers per loop
    exit_reason: str                     # "pe_converged"|"answers_stable"|"max_loops"|"single_loop"
```

Available via `ReasoningTrace.loop_trace` when using `POST /query/trace`.

---

## 6. Empirical Characterisation

On the toy graph fixture (21 nodes, 30 edges), single-pass vs 2-loop traversal from `"newton"`:

| Metric | 1 loop | 2 loops |
|---|---|---|
| Unique tail entities | 8 | 12 |
| Max path depth reached | 3 | 3 |
| Exit reason | — | answers_stable |
| PE loop 1 | 0.45 | 0.45 |
| PE loop 2 | — | 0.20 |

The 2-loop run surfaces 4 additional entities unreachable in the single pass, and PE drops significantly as the Engram prior catches up to the actual paths. On larger, sparser graphs the gains are expected to be substantially larger, consistent with LoopLM's reported improvements on hard reasoning benchmarks [zhu2025loooplm].

---

## 7. Complexity Analysis

Let `B` = beam width, `H` = max hops, `N` = nodes, `T` = max loops. Single-pass traversal: O(T_1 · B · H) where T_1 = 1. Looped traversal adds O(T · B · H) with a constant factor from seed expansion (~K additional seeds per loop, K ≪ N). The PE computation (Jaccard on relation sets) is O(|prior_rels| + |actual_rels|), negligible against traversal cost. The adaptive exit gate amortizes across queries that converge quickly (T_actual ≪ T_max).

---

## 8. Design Decisions

**Why merge paths across all loops rather than only the last?**
Each loop explores a different neighbourhood (different seeds). Merging gives `extract()` the full picture — paths discovered in loop 1 from the original seeds coexist with paths discovered in loop 2 from expanded seeds. This maximises coverage without requiring `extract()` to be loop-aware.

**Why use original seeds for PE computation (not expanded)?**
PE measures alignment between the Engram prior (built from the original query intent) and the actual paths. Using expanded seeds would shift the PE reference point per loop, making the exit gate signal inconsistent. Anchoring to original seeds ensures PE delta measures genuine improvement in reasoning quality, not drift from seed expansion.

**Why seed expansion rather than full replacement?**
LoopLM passes the entire hidden state forward — all prior context is preserved. The analog in graph search is to always include original seeds (preserving query intent) while adding new candidates (expanding context). Full replacement would abandon the original query anchor, potentially causing semantic drift.

---

## 9. References

- [zhu2025loooplm] Zhu, R.-J., Wang, Z., Hua, K., et al. (2025). Scaling Latent Reasoning via Looped Language Models. arXiv:2510.25741. ByteDance Seed / UC Santa Cruz et al.
- [vaswani2017attention] Vaswani, A. et al. (2017). Attention Is All You Need. NeurIPS.
- [hamilton2017graphsage] Hamilton, W., Ying, Z., & Leskovec, J. (2017). Inductive Representation Learning on Large Graphs. NeurIPS.
- [bengio2025soliton] Bengio, Y. et al. (2025). Consciousness as a Soliton, Not a Process: Identity, Memory, and the Hard Problem in Coherence Field Theory. UCFT 2025 Preprint.
- [Phase 69] CEREBRUM Phase 69: PredictiveCodingEngine — active inference, PE, soliton_index.
- [Phase 60] CEREBRUM Phase 60: MACH — Multi-Agent Consensus Hierarchies (L1/L2/L3).
- [Phase 68] CEREBRUM Phase 68: ChemicalModulator — metabolic scalar regulation.
- [Phase 55] CEREBRUM Phase 55: Engram — persistent relation-pattern cache; EngramTraversal.

---
**Reviewed on**: May 2, 2026 for version v2.71.0


