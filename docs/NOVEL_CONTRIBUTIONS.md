# Novel Contributions and IP Claims

## CEREBRUM: Community-Structured Graph Attention for Knowledge Graph Reasoning

**Document Classification**: Intellectual Property Reference
**Authors**: Bryan Alexander Buchorn (AMP)
**Date**: March 2026
**Version**: v1.1.0 (Phase 20 COMPLETE)

> This document consolidates the novel technical contributions of the CEREBRUM framework for use in patent applications, academic priority claims, and commercial IP protection. Each claim is substantiated with prior art analysis and a statement of the specific technical distinction.

---

## Part I: Core Algorithm Claims

### Claim 1: Dual/Triple Signal Community Fusion (DSCF / TSC)

**Description**: A community detection algorithm that applies Local Propagation Algorithm (LPA), modularity gain (Louvain-style), and optionally Infomap flow signals **simultaneously at each node update**, fusing them via a weighted consensus mechanism at each iteration.

**Novelty Statement**: All prior community detection algorithms that combine multiple signals do so by operating on disjoint node subsets (e.g., apply LPA to low-degree nodes, Louvain to high-degree nodes [Sun et al., 2024]) or by running algorithms sequentially and merging results. DSCF is the first algorithm that applies all signals to every node at every update step. The simultaneous per-node fusion produces communities with distinct "dual-signal" structural character that is not achievable by sequential or partitioned combination.

**Closest Prior Art**:
- Louvain (Blondel et al., 2008): Global modularity only, no LPA signal
- Leiden (Traag et al., 2019): Refinement phase + modularity, no LPA signal
- LPA (Raghavan et al., 2007): Local propagation only, no modularity signal
- LPA-Louvain hybrids (Sun et al., 2024): Apply signals to disjoint node subsets by degree threshold — categorically different from DSCF's per-node simultaneous fusion

**Key Technical Differentiator**: The specific architectural choice to compute both $\Delta Q_{modularity}(v, c)$ and $f_{LPA}(v, c)$ for every candidate community $c$ at every node $v$ in every iteration, then fuse them via: $\text{score}(v, c) = \alpha \cdot \Delta Q(v,c) + \beta \cdot f_{LPA}(v,c) + \gamma \cdot \text{flow}(v,c)$

**Relevant files**: `core/community_engine.py`, `core/leiden_native.py`
**Documented in**: `docs/arxiv/PAPER_001_DSCF_TSC.md`, `docs/specifications/SPEC_001_DSCF_TSC.md`

---

### Claim 2: Community-Structured Attention (CSA) Formula

**Description**: A graph edge attention weight formula that incorporates community membership as a soft global constraint alongside semantic similarity, relation type, path length penalty, hop decay, and PageRank centrality. The formula is training-free, computed analytically from graph topology at query time.

**The Formula**:
$$a(u,v,k) = \sigma\left(\alpha \cdot \cos(\vec{e}_u, \vec{e}_v) + \beta \cdot S_C(u,v) + \gamma \cdot w_{rel} - \delta \cdot d_{norm}(u,v) + \varepsilon \cdot \phi(k) + \zeta \cdot PR(v)\right)$$

**Novelty Statement**: Graph Attention Networks (GAT, HAN, HGT) compute local attention over immediate neighborhoods using learned weight matrices. CSA is the first attention formulation that includes global community membership ($S_C(u,v)$) as a term. No published GNN or KG reasoning paper computes attention weights using community assignment as an explicit term in the attention formula.

**Closest Prior Art**:
- GAT (Veličković et al., 2018): Local neighborhood attention, learned weights, no community term
- HAN (Wang et al., 2019): Meta-path attention, no community term
- HGT (Hu et al., 2020): Heterogeneous attention, learned, no community term
- GraphRAG (Microsoft, Edge et al., 2024): Communities used for LLM summarization, not as attention weights in traversal

**Key Technical Differentiator**: The $\beta \cdot S_C(u,v)$ term, where $S_C(u,v) \in [0,1]$ measures community co-membership and is computed from live DSCF partitions, not from learned parameters.

**Relevant files**: `core/attention_engine.py`
**Documented in**: `docs/arxiv/PAPER_002_CSA.md`, `docs/specifications/SPEC_002_CSA.md`

---

### Claim 3: CSA-Guided Multi-Hop Beam Search (Glass-Box Reasoning)

**Description**: A multi-hop KG traversal architecture where beam search is guided exclusively by CSA attention weights derived from graph topology, producing a complete verifiable reasoning trace at zero inference cost (no trained model, no LLM).

**Novelty Statement**: All published multi-hop KG reasoning systems using beam search (MINERVA, DeepPath, BeamQA) require training on labeled question-answer pairs. CEREBRUM's BeamTraversal uses CSA weights computed analytically from graph structure — achieving competitive H@10 on MetaQA zero-shot, with no training data.

**Closest Prior Art**:
- MINERVA (Das et al., 2018): RL-trained path agent, requires labeled QA pairs
- DeepPath (Xiong et al., 2017): RL, requires training
- BeamQA (Atif et al., 2023): Supervised sequence-to-sequence beam search

**Key Technical Differentiator**: Zero-shot multi-hop reasoning at competitive H@10 using only graph topology.

**Relevant files**: `reasoning/traversal.py`, `reasoning/path_scorer.py`

---

## Part II: Neuromorphic Architecture Claims

### Claim 4: Experience-Dependent Bridge Twin Nodes (LTP/LTD Analog for KG)

**Description**: A mechanism that materializes structural relay nodes (bridge twins) between graph communities based on traversal co-occurrence frequency, analogous to synaptic LTP/LTD in hippocampal circuits. Bridge twins form when crossing count exceeds a threshold and are pruned when crossings decay below a threshold.

**Novelty Statement**: No published KG system uses experience-dependent structural modification inspired by LTP/LTD. Static graph shortcuts (agentic LLM expansion, shortcut edges in GNNs) are added by external agents or hand-coded rules, not by the reasoning engine's own traversal history. The specific mechanism of counting traversal co-occurrences and triggering structural node materialization from those counts is, to our knowledge, original.

**Relevant files**: `core/bridge_engine.py`
**Documented in**: `docs/arxiv/PAPER_003_BRIDGE_TWINS.md`

---

### Claim 5: STDP-Inspired Directional Causal Edge Inference

**Description**: A streaming discretizer that materializes directional `CAUSES` edges in a Knowledge Graph from event timing patterns, using Hebbian-inspired weight accumulation with configurable LTP/LTD asymmetry and lazy O(1) weight decay.

**Novelty Statement**: STDP has been applied to spiking neural network learning rules extensively, but no published work applies STDP mechanics to dynamic causal edge discovery in Knowledge Graphs. The combination of: (1) Hebbian co-occurrence weighting, (2) directional edge materialization (pre→post), (3) lazy weight decay reducing per-event complexity from O(N) to O(1), and (4) adversarial causal flood protection (min_causal_span, chi-squared) is an original contribution.

**Relevant files**: `core/discretizer.py`
**Documented in**: `docs/arxiv/PAPER_004_STDP_CAUSAL.md`

---

### Claim 6: REM Cycle — Sleep-Inspired KG Maintenance

**Description**: A three-phase background maintenance loop (Prune → Consolidate → Synthesize) for autonomous KG quality management, inspired by the biological sleep-cycle memory consolidation model (synaptic homeostasis, NREM slow-wave, hippocampal replay).

**Novelty Statement**: KG maintenance systems (NELL, ATOMIC) focus on *adding* new facts. The REM Cycle's three-phase design for autonomous removal, structural consolidation, and proactive synthesis with differential decay rates per edge relation type has no precedent in the KG maintenance literature.

**Relevant files**: `core/rem_engine.py`
**Documented in**: `docs/arxiv/PAPER_007_REM_CYCLE.md`

---

## Part III: Cross-Modal and Federated Claims

### Claim 7: Procrustes SVD Sensor-to-KG Alignment with Canonical Basis Anchor

**Description**: A cross-modal alignment system that projects heterogeneous sensor signals (time-series, waveforms) into the same embedding space as Knowledge Graph text entities via Procrustes SVD rotation, with a canonical embedding basis that prevents geometric drift accumulation across federated hops.

**Novelty Statement**: Multimodal KG papers (MMKG, MKGFormer) integrate images and text but not continuous sensor signals. The specific combination of: (1) statistical/spectral signal feature extraction, (2) Procrustes SVD alignment to KG embedding space, (3) a canonical root basis shared across all federation participants, and (4) namespace isolation preventing cross-modal entity collisions, is original.

**Relevant files**: `core/signal_encoder.py`
**Documented in**: `docs/arxiv/PAPER_008_SIGNAL_ENCODER.md`

---

### Claim 8: Holographic Index for Privacy-Preserving Federated Discovery

**Description**: A federated graph discovery system using Bloom filter sketches (probabilistic membership) and community centroid vectors (structural fingerprints) to enable two CEREBRUM nodes to identify likely-relevant shared content without revealing individual entity identities.

**Novelty Statement**: Federated learning (McMahan et al., 2017) aggregates model gradients, not graph structure. SPARQL federation requires full query exposure. The Holographic Index is the first KG-specific privacy-preserving discovery mechanism using the combination of Bloom filters and community centroid structural fingerprints.

**Relevant files**: `core/hologram.py`
**Documented in**: `docs/arxiv/PAPER_005_HOLOGRAPHIC_INDEXING.md`

---

## Part IV: Production Hardening Claims

### Claim 9: Bayesian Beam Search with Warm-Start Prior Seeding for KG Traversal

**Description**: Application of Beta-distribution path modeling and Thompson sampling to multi-hop KG beam search, with a warm-start mechanism that seeds the first-hop Beta prior from the CSA attention weight to reduce cold-start variance.

**Novelty Statement**: Bayesian bandits and Thompson sampling are well-established in recommendation and exploration-exploitation literature. Their application to *multi-hop KG beam search with path-level Beta distributions* seeded from graph-structural attention weights is novel — no prior KG reasoning paper uses this formulation.

**Relevant files**: `reasoning/traversal.py`
**Documented in**: `docs/arxiv/PAPER_006_BAYESIAN_BEAM.md`

---

### Claim 10: Eight-Hole Structural Hardening Taxonomy for KG Systems

**Description**: A systematic methodology for identifying and patching cross-feature interaction bugs ("structural holes") in complex KG systems, covering five taxonomic categories: stale reference, adversarial input, namespace collision, bias/saturation, and evaluation artifacts.

**Novelty Statement**: The specific taxonomy of structural hole categories as applied to KG reasoning systems — and the complete set of eight patched holes with backward-compatible fixes — constitutes a novel engineering contribution for production KG deployment.

**Documented in**: `docs/arxiv/PAPER_016_PRODUCTION_HARDENING.md`

---

## Part V: Composite System Claims

### Claim 11: CEREBRUM as an Integrated Zero-Shot KG Reasoning System

**Description**: The complete CEREBRUM system integrating DSCF community detection, CSA attention, zero-shot beam traversal, LTP/LTD bridge twins, STDP causal inference, REM Cycle maintenance, Bayesian beam search, cross-modal alignment, federated discovery, streaming ingest, metacognitive verification, and production hardening into a single coherent framework with zero training data requirements and complete reasoning transparency.

**Novelty Statement**: No prior work integrates this combination of capabilities in a single KG reasoning framework. The specific architectural choices — using communities as attention heads, using traversal co-occurrence for structural modification, using STDP timing for causal discovery, and composing these into a sleep-cycle maintenance loop — represent a novel systems-level contribution.

---

## Prior Art Summary Table

| CEREBRUM Component | Closest Prior Art | Key Distinction |
|---|---|---|
| DSCF simultaneous fusion | LPA-Louvain hybrids (Sun 2024) | Per-node vs. per-population fusion |
| CSA formula ($S_C$ term) | GAT, HAN, HGT | Community term absent from all GNN attention |
| Zero-shot beam traversal | MINERVA, DeepPath, BeamQA | Training required vs. fully training-free |
| Bridge Twins (LTP/LTD) | GNN shortcuts, agentic expansion | Experience-dependent vs. static/agent-added |
| STDP causal edges in KG | SNN-STDP | Neural learning rule vs. KG edge discovery |
| REM Cycle maintenance | NELL, ATOMIC, KGE training | Proactive pruning + synthesis vs. fact addition |
| Procrustes sensor→KG | MMKG, MKGFormer | Sensor signals vs. image/text only |
| Holographic Index | Federated learning, SPARQL federation | Privacy-preserving structural discovery |
| Bayesian beam + warm-start | Bandit algorithms | KG-specific Beta seeding from CSA weights |
| Structural hole taxonomy | Standard testing methodology | Cross-feature interaction analysis for KG |

---

## Legal Notice

All rights, title, and interest in and to the CEREBRUM software, documentation, algorithms, and related intellectual property documented herein are and shall remain the exclusive property of **Bryan Alexander Buchorn (AMP)**.

CEREBRUM is dual-licensed:
- **Non-Commercial Use**: PolyForm Noncommercial License 1.0.0
- **Commercial Use**: Separate commercial license required

For commercial licensing: **bryan.alexander@buchorn.com**

**Copyright © 2026 Bryan Alexander Buchorn (AMP). All Rights Reserved.**
