# WP_017: Active Inference & Predictive Reasoning

**Phase 111 — Proactive Knowledge Graph Traversal**

## Abstract
Traditional graph reasoning engines operate reactively, exploring the topology only after a query is received. CEREBRUM Phase 111 introduces **Active Inference**, a mechanism where the system leverages historical reasoning patterns (Engrams) to project predictive priors before traversal initiates. This "Top-Down" projection biases the beam search toward high-probability relation sequences, significantly reducing the computational cost of "Expected" reasoning while automatically increasing search resolution for "Surprising" branches.

## 1. Predictive Coder
The `PredictiveCoder` interfaces with the `EngramCache` to retrieve the top N frequent relation patterns associated with a query seed. These patterns are converted into a `PredictivePrior`.

## 2. Proactive Bias
During `BeamTraversal`, the system compares candidate expansions against the `PredictivePrior`. 
- **Matching Branches:** Receive a `1.5x` score boost, ensuring they stay within the beam even at lower raw similarity.
- **Non-Matching Branches:** Processed at baseline, allowing for "Surprise-Driven" discovery if evidence is strong enough to overcome the prior bias.

## 3. Homeostatic Integration
The **Prediction Error (PE)** signal — the divergence between the projected prior and the actual traversal path — is used as a high-arousal trigger for the `ChemicalModulator`. A high PE causes a temporary spike in search energy, widening the beam to investigate the anomaly.

## 4. Conclusion
Active Inference transforms CEREBRUM from a passive searcher into a proactive reasoner, aligning the system's architecture more closely with the predictive coding mechanisms of the human brain.

---
**Reviewed on**: April 21, 2026 for version v2.24.0
