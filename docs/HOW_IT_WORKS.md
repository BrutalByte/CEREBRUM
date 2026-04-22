# How It Works: The CEREBRUM Architecture

CEREBRUM is a **Community-Structured Graph Attention** framework designed for Knowledge Graph (KG) reasoning. Unlike Large Language Models (LLMs) that rely on probabilistic next-token prediction, CEREBRUM performs deterministic reasoning over structured facts. It mimics the functional organization of the human brain to achieve high-accuracy, multi-hop inference without the need for traditional training data.

This document breaks down the four core pillars of the CEREBRUM architecture.

---

## 1. The Foundation: Triple-Signal Consensus (TSC)

The first step in CEREBRUM's reasoning process is **Community Detection**. In a Knowledge Graph, facts are not isolated; they form dense clusters or "neighborhoods." CEREBRUM identifies these neighborhoods using a novel algorithm called **Triple-Signal Consensus (TSC)**.

TSC fuses three distinct signals during its update loop:
1.  **Local Propagation (LPA)**: Nodes adopt the most frequent labels of their immediate neighbors (local topology).
2.  **Modularity Gain (Global)**: Nodes move into communities that maximize the overall density of internal connections (global structure).
3.  **Centrality Flow (Flow)**: Nodes are grouped based on the "traffic" or flow of information between them (centrality).

By fusing these signals, TSC creates a partition of the graph where each community acts as a specialized "attention head." This allows the system to focus its computational effort on relevant facts while ignoring billions of unrelated ones.

---

## 2. The Flashlight: Community-Structured Attention (CSA)

Once the graph is organized into communities, CEREBRUM uses **Community-Structured Attention (CSA)** to navigate it. CSA is an analytical attention formula that scores the "importance" of every potential hop during a search.

The current CSA formula uses **10 parameters** to evaluate each edge:
- **Semantic Similarity**: How well the labels of the nodes match the query.
- **Community Co-membership**: Whether the jump stays within the same "neighborhood."
- **Edge-Type Strength**: The inherent weight of the relationship (e.g., "parent of" vs "lives in").
- **Structural Priors**: PageRank authority, temporal recency, and node visit history.
- **Synthetic Penalties**: A "trust" score that penalizes edges synthesized by the AI itself compared to hard-coded facts.

This formula allows CEREBRUM to "pay attention" to the right paths without ever being trained on them. It is a "glass-box" mechanism—you can inspect exactly why one path was chosen over another.

---

## 3. The Search: Bayesian Beam Traversal

To find answers, CEREBRUM sends out "scouts" in a **Beam Search**.
- **Beam Width**: The number of scouts sent out at each step (typically 10-50).
- **Beam Depth**: How many "hops" (steps) the scouts are allowed to take.

This search is **Bayesian**, meaning it uses Thompson Sampling and Beta distributions to handle uncertainty. Each scout maintains a "prior" belief about its path's quality, which is updated as it discovers new facts. 

In latest versions, this search is **Looped**. The system can run multiple passes of the search, using the answers from the first pass to refine its strategy for the second pass—much like a human re-reads a difficult sentence to understand it better.

### Active Inference (Phase 111)
CEREBRUM now performs **Active Inference**, moving from a reactive searcher to a proactive prosector. Before the beam search starts, the **PredictiveCoder** uses historical Engram patterns to project the "Expected Path." This projection biases the search toward likely relation sequences, drastically reducing computational waste for standard queries while focusing energy on "surprising" discoveries.

---

## 4. The Decision: Multi-Agent Consensus & Global Workspace

To ensure the highest accuracy, CEREBRUM uses **Multi-Agent Consensus Hierarchies (MACH)**. Instead of relying on a single path, it runs multiple reasoning strategies in parallel and requires them to reach a consensus.

### Global Workspace (Phase 110)
Phase 110 introduces the **Global Workspace (GWS)** blackboard. Communities broadcast "surprise" signals (high-novelty discoveries) to a shared workspace.
- **Pre-emption**: If the Global Workspace identifies a highly corroborated, high-novelty signal, it can pre-empt the standard hierarchical escalation.
- **Cognitive Flexibility**: This allows the system to switch "cognitive focus" instantly, prioritizing paths that offer the most information gain (maximum surprise reduction).

---

## 5. The Brain: Metabolic and Mnemonic Regulation

CEREBRUM isn't just a static algorithm; it is a **dynamic system** regulated by biological analogs:

### Metabolic Homeostasis
The system maintains five "metabolic scalars" (Dopamine, Adrenaline, etc.) that adjust its behavior in real-time. If a search is failing, "Adrenaline" rises, causing the beam width to expand and the system to work harder. If it finds a success, "Dopamine" reinforces that logic for the future.

### Mnemonic Engrams
Successful reasoning paths leave a physical trace in the system called an **Engram**. These Engrams are compressed using **SpeedTalk** (a phonemic dialect) and stored in a durable cache. Future searches are "steered" by these Engrams, allowing the system to learn from experience without a gradient descent training loop.

### Predictive Coding
CEREBRUM continuously makes predictions about its own search results. The **Prediction Error (PE)** between its expectation and the reality drives the metabolic modulation, allowing the system to self-calibrate its "worldview" stability (measured as the **Soliton Index**).

---

## Summary: A New Paradigm for AI

CEREBRUM represents a shift from **Stochastic AI** (guessing) to **Structural AI** (reasoning). By combining the formal logic of Knowledge Graphs with the dynamic flexibility of Brain-inspired regulation, it achieves:
- **Zero Hallucination**: Every answer is grounded in a verifiable graph path.
- **Zero Training**: It reasons directly from the topology of your data.
- **Full Observability**: You can watch every "synaptic pulse" in real-time.

**CEREBRUM is the first formal reasoning engine designed for the age of complex, high-stakes data.**
