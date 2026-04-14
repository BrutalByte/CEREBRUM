"""
LoopedBeamTraversal — Adaptive Iterative Reasoning for CEREBRUM (Phase 70).

Inspired by *Scaling Latent Reasoning via Looped Language Models* (Zhu et al.,
ByteDance Seed / arXiv:2510.25741, Nov 2025). LoopLM demonstrates that applying
the same computation stack T times — rather than once — yields dramatically better
reasoning on hard inputs without increasing parameter count. An adaptive exit gate
prevents both underthinking (exits too early) and overthinking (continues past the
point of improvement).

CEREBRUM's analog: apply BeamTraversal T times, using top answer entities from
loop t as additional seeds for loop t+1. Phase 69's Prediction Error (PE) serves
as the primary exit gate signal — when PE stops improving (|ΔPE| < γ), the loop
terminates. Three feedback channels operate between loops:

  1. **Semantic** — top answer entities expand the seed set, providing a richer
     starting neighbourhood for the next pass.
  2. **Metabolic** — the PE signal updates ChemicalModulator scalars (arousal,
     novelty, reinforcement), which adjust beam_width and CSA α/β for the next loop.
  3. **Mnemonic** — Engram records added during loop t bias beam pruning in loop
     t+1 toward known-productive relation patterns.

This gives CEREBRUM three inter-loop feedback channels vs LoopLM's single
hidden-state channel, making iterative refinement richer per compute step.

Reference
---------
Zhu, R.-J., Wang, Z., Hua, K., et al. (2025). Scaling Latent Reasoning via
Looped Language Models. arXiv:2510.25741. ByteDance Seed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from reasoning.answer_extractor import extract

logger = logging.getLogger("cerebrum.looped_traversal")


# ---------------------------------------------------------------------------
# LoopTrace — per-loop diagnostics
# ---------------------------------------------------------------------------

@dataclass
class LoopTrace:
    """
    Diagnostic record of a multi-loop traversal run.

    Fields mirror LoopLM's per-step exit metrics — loops_run is the depth
    actually reached, analogous to LoopLM's recurrent depth t.
    """
    loops_run: int
    """Number of loops actually executed (1 = single-pass, no looping)."""

    seeds_per_loop: List[List[str]] = field(default_factory=list)
    """Seed entity IDs used at the start of each loop (grows as answers expand)."""

    pe_per_loop: List[Optional[float]] = field(default_factory=list)
    """Prediction Error after each loop. None when no PredictiveCodingEngine attached."""

    paths_per_loop: List[int] = field(default_factory=list)
    """Raw path count returned by BeamTraversal at each loop."""

    new_answers_per_loop: List[int] = field(default_factory=list)
    """New unique answer entities discovered vs the previous loop."""

    exit_reason: str = "max_loops"
    """
    Why the loop terminated:
      'single_loop'    — max_loops=1, no looping attempted (backward compat).
      'pe_converged'   — |ΔPE| < pe_convergence_threshold (improvement stalled).
      'answers_stable' — Jaccard(prev_answers, curr_answers) ≥ threshold.
      'max_loops'      — reached the configured ceiling.
    """


# ---------------------------------------------------------------------------
# LoopedBeamTraversal
# ---------------------------------------------------------------------------

class LoopedBeamTraversal:
    """
    Wraps any BeamTraversal-compatible traversal engine and applies it T times,
    progressively refining the reasoning via seed expansion and adaptive exit.

    Parameters
    ----------
    traversal : BeamTraversal (or EngramTraversal / any subclass)
        The inner traversal engine to loop.
    predictive_coder : Optional[PredictiveCodingEngine]
        Phase 69 engine. When attached, PE becomes the primary exit gate signal.
        Without it, the engine falls back to answer-stability exit only.
    max_loops : int
        Maximum number of traversal iterations (default 4).
        Set to 1 for single-pass behaviour (fully backward compatible).
    pe_convergence_threshold : float
        Exit when |PE_t - PE_{t-1}| < this value (default 0.05).
        Mirrors LoopLM's γ threshold in the ideal continuation probability.
    answer_stability_threshold : float
        Exit when Jaccard(prev_answers, curr_answers) ≥ this value (default 0.8).
        Fallback gate when no PredictiveCodingEngine is present.
    top_k_seed_expansion : int
        Number of top answer entities added as new seeds each loop (default 2).
    """

    def __init__(
        self,
        traversal: Any,
        predictive_coder: Optional[Any] = None,
        max_loops: int = 4,
        pe_convergence_threshold: float = 0.05,
        answer_stability_threshold: float = 0.80,
        top_k_seed_expansion: int = 2,
    ) -> None:
        self.traversal                  = traversal
        self.predictive_coder           = predictive_coder
        self.max_loops                  = max_loops
        self.pe_convergence_threshold   = pe_convergence_threshold
        self.answer_stability_threshold = answer_stability_threshold
        self.top_k_seed_expansion       = top_k_seed_expansion

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def traverse(
        self,
        seeds: List[str],
        query_time: Optional[float] = None,
        query_embedding=None,
        trace_info=None,
    ) -> Tuple[List[Any], LoopTrace]:
        """
        Execute the looped traversal and return (merged_paths, loop_trace).

        Parameters
        ----------
        seeds         : initial seed entity IDs
        query_time    : optional temporal filter (passed through to inner traversal)
        query_embedding : optional query embedding (passed through)
        trace_info    : optional ReasoningTrace; loop_trace is attached if provided

        Returns
        -------
        merged_paths : List[TraversalPath] — deduplicated across all loops,
                       best-score per tail entity wins.
        loop_trace   : LoopTrace — full per-loop diagnostics.
        """
        # --- Backward-compat fast path ---
        if self.max_loops == 1:
            paths = self.traversal.traverse(
                seeds,
                query_time=query_time,
                query_embedding=query_embedding,
                trace_info=trace_info,
            )
            lt = LoopTrace(
                loops_run=1,
                seeds_per_loop=[list(seeds)],
                pe_per_loop=[None],
                paths_per_loop=[len(paths)],
                new_answers_per_loop=[len({p.tail for p in paths if p.hop_depth > 0})],
                exit_reason="single_loop",
            )
            if trace_info is not None:
                trace_info.loop_trace = lt
            return paths, lt

        # --- Multi-loop execution ---
        return self._multi_loop(seeds, query_time, query_embedding, trace_info)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _multi_loop(
        self,
        original_seeds: List[str],
        query_time: Optional[float],
        query_embedding,
        trace_info,
    ) -> Tuple[List[Any], LoopTrace]:

        original_seed_set: Set[str] = set(original_seeds)
        current_seeds: List[str]    = list(original_seeds)

        # Best path per tail entity across all loops
        best_by_tail: Dict[str, Any] = {}

        # Per-loop tracking
        seeds_log:       List[List[str]]         = []
        pe_log:          List[Optional[float]]   = []
        paths_count_log: List[int]               = []
        new_answers_log: List[int]               = []

        prev_answer_entities: Set[str] = set()
        prev_pe:              Optional[float] = None
        exit_reason: str = "max_loops"

        for t in range(1, self.max_loops + 1):
            logger.debug("Loop %d/%d — seeds: %s", t, self.max_loops, current_seeds[:5])

            # Run inner traversal (no trace_info on inner loops to avoid hop duplication)
            paths = self.traversal.traverse(
                current_seeds,
                query_time=query_time,
                query_embedding=query_embedding,
            )

            # Merge: keep highest-score path per tail entity
            for p in paths:
                if p.hop_depth > 0:
                    tail = p.tail
                    if tail not in best_by_tail or float(p.score) > float(best_by_tail[tail].score):
                        best_by_tail[tail] = p

            # Extract top answers from THIS loop for exit gate + seed expansion
            loop_answers = extract(
                paths,
                top_k=max(self.top_k_seed_expansion + 5, 10),
                min_hop=1,
            )
            curr_answer_entities = {a.entity_id for a in loop_answers}

            # Compute PE via PredictiveCodingEngine if available
            curr_pe: Optional[float] = None
            if self.predictive_coder is not None:
                try:
                    prior  = self.predictive_coder.predict(list(original_seeds))
                    result = self.predictive_coder.update(prior, paths)
                    curr_pe = result.prediction_error
                except Exception as exc:
                    logger.debug("PredictiveCodingEngine failed in loop %d: %s", t, exc)

            # Record loop stats
            new_count = len(curr_answer_entities - prev_answer_entities)
            seeds_log.append(list(current_seeds))
            pe_log.append(curr_pe)
            paths_count_log.append(len(paths))
            new_answers_log.append(new_count)

            logger.debug(
                "Loop %d: paths=%d  curr_answers=%d  new=%d  PE=%s",
                t, len(paths), len(curr_answer_entities), new_count,
                f"{curr_pe:.3f}" if curr_pe is not None else "N/A",
            )

            # --- Exit gate (LoopLM §3.4 analogue) ---
            should_exit, reason = self._should_exit(
                t, prev_answer_entities, curr_answer_entities, prev_pe, curr_pe,
            )
            if should_exit:
                exit_reason = reason
                logger.debug("Early exit at loop %d: %s", t, reason)
                break

            # Expand seeds for next loop:
            # original seeds always included; top-K new answer entities appended
            expansion = [
                a.entity_id
                for a in loop_answers[: self.top_k_seed_expansion]
                if a.entity_id not in original_seed_set
            ]
            next_seeds = list(original_seeds) + expansion
            # Deduplicate while preserving order
            seen: Set[str] = set()
            current_seeds = [s for s in next_seeds if not (s in seen or seen.add(s))]  # type: ignore[func-returns-value]

            prev_answer_entities = curr_answer_entities
            prev_pe = curr_pe

        # Build final merged path list
        merged_paths = list(best_by_tail.values())

        loop_trace = LoopTrace(
            loops_run=len(seeds_log),
            seeds_per_loop=seeds_log,
            pe_per_loop=pe_log,
            paths_per_loop=paths_count_log,
            new_answers_per_loop=new_answers_log,
            exit_reason=exit_reason,
        )

        # Attach trace to ERT if provided (final loop only)
        if trace_info is not None:
            trace_info.loop_trace = loop_trace

        logger.info(
            "LoopedBeamTraversal: %d loops, exit=%r, total_merged_paths=%d",
            loop_trace.loops_run, exit_reason, len(merged_paths),
        )
        return merged_paths, loop_trace

    def _should_exit(
        self,
        loop_num: int,
        prev_answers: Set[str],
        curr_answers: Set[str],
        prev_pe: Optional[float],
        curr_pe: Optional[float],
    ) -> Tuple[bool, str]:
        """
        Adaptive exit gate — mirrors LoopLM's ideal continuation probability.

        Penalises both underthinking (exits too early) and overthinking (runs
        past the point of improvement). PE-based exit takes priority when the
        PredictiveCodingEngine is attached; answer-stability is the fallback.
        """
        # Primary: PE convergence (Phase 69 integration)
        if prev_pe is not None and curr_pe is not None:
            if abs(curr_pe - prev_pe) < self.pe_convergence_threshold:
                return True, "pe_converged"

        # Secondary: answer-set stability (always available)
        if prev_answers:
            union = len(prev_answers | curr_answers)
            if union > 0:
                jaccard = len(prev_answers & curr_answers) / union
                if jaccard >= self.answer_stability_threshold:
                    return True, "answers_stable"

        return False, "continuing"
