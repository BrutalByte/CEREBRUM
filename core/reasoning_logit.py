"""
Core reasoning logit framework for CEREBRUM.

Consolidates all signals (semantic, community, grounding, etc.) into a unified structure
to ensure consistency across traversal, pruning, and ranking.
"""
from dataclasses import dataclass
from typing import Tuple
import numpy as np

@dataclass
class ReasoningLogit:
    """
    Unified signal vector representing a single reasoning step.
    Signals are normalized to [0, 1] before storage.
    """
    sim: float = 0.0          # Semantic Similarity
    cs: float = 0.5           # Community Score
    etw: float = 0.0          # Edge Type Weight
    nd: float = 0.0           # Normalized Distance
    hd: float = 0.0           # Hop Decay
    pr_v: float = 0.0         # PageRank Prior
    td: float = 0.0           # Temporal Decay (Edge)
    nr_v: float = 0.0         # Node Recency Prior
    sd: float = 0.0           # Synthesis Density (REM/SynapticBridge)
    grounding: float = 1.0    # Grounding/Confidence Score

    def to_vector(self) -> np.ndarray:
        """Convert to flat vector for parametric learning."""
        return np.array([
            self.sim, self.cs, self.etw, self.nd, self.hd, 
            self.pr_v, self.td, self.nr_v, self.sd, self.grounding
        ], dtype=np.float32)

    @classmethod
    def from_tuple(cls, t: Tuple[float, ...]) -> "ReasoningLogit":
        """Create from the legacy or updated tuple."""
        return cls(*t)

    def score(self, params: Tuple[float, ...]) -> float:
        """
        Apply learned weights to compute logit score.
        params: (alpha, beta, gamma, delta, epsilon, zeta, eta, iota, mu, theta)
        mu is the penalty for synthetic/synthesis density.
        """
        if len(params) == 9:
            # Backward compatibility
            a, b, g, d, e, z, eta, iota, theta = params
            mu = 0.050
        else:
            a, b, g, d, e, z, eta, iota, mu, theta = params

        raw = (
            a * self.sim
            + b * self.cs
            + g * self.etw
            - d * self.nd
            + e * self.hd
            + z * self.pr_v
            + eta * self.td
            + iota * self.nr_v
            - mu * self.sd
            + theta * self.grounding
        )
        return 1.0 / (1.0 + np.exp(-raw))
