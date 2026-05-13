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

### Vectorized Beam Scoring (Phase 134)
To handle the scale of billions of possible paths, CEREBRUM uses **Vectorized Beam Scoring**. By moving the core attention calculation into optimized NumPy operations, the system achieves a **10x speedup**, allowing complex multi-hop reasoning in sub-30ms latencies even on standard CPU hardware.

### H1SE: Solving the Hub Problem (Phase 137)
In many Knowledge Graphs, "popular" nodes (hubs) like *United States* or *Action Movie* can dominate the beam search, preventing smaller, more specific paths from being explored. CEREBRUM uses **Hop-1 Intermediate Seed Expansion (H1SE)** to solve this. Instead of one global beam, each unique node reached at hop 1 receives its own independent search budget. This ensures that every promising direction gets a fair chance to reach the answer.

---

## 4. The Intelligence: Auto-Configuration and Zero-Shot Boosts

In latest versions, CEREBRUM has moved from a "manual config" tool to an **Autonomous Reasoner**.

### GraphProfiler: Automatic Strategy (Phase 166)
Graphs come in many shapes. Some are "hub-heavy" (like Wikipedia), while others are "type-rich" (like medical databases). CEREBRUM's **GraphProfiler** analyzes the topology of your data at build time. It automatically detects the "Graph Regime" and configures the reasoning engine accordingly—turning on features like H1SE or STRB only when they are statistically likely to help.

### STRB: Zero-Config Intent Matching (Phase 172)
Traditionally, you had to tell a graph searcher which "relation type" mattered (e.g., "Look for *treats* edges"). With **Semantic Terminal Relation Boost (STRB)**, CEREBRUM does this automatically. It uses its embedding engine to compare your query text to the labels of every relation in the graph. If you ask "What treats X?", STRB automatically boosts paths ending in "treats" or "therapeutic_use," enabling high-accuracy reasoning with zero manual configuration.

### TAB: Navigating Deep Heterogeneity (Phase 164)
For complex questions (3+ hops), the search can get lost in the "middle" hops. **Terminal-Anchor Boost (TAB)** provides a navigational beacon. It identifies "anchor sets"—groups of entities that are known to be just one step away from the target type—and applies a bonus to paths that reach those anchors at the penultimate hop. This "look-ahead" biasing significantly improves recall for deep, heterogeneous queries.

---

## 5. The Decision: Multi-Agent Consensus & Global Workspace

To ensure the highest accuracy, CEREBRUM uses **Multi-Agent Consensus Hierarchies (MACH)**. Instead of relying on a single path, it runs multiple reasoning strategies in parallel and requires them to reach a consensus.

### Global Workspace (Phase 110)
Phase 110 introduces the **Global Workspace (GWS)** blackboard. Communities broadcast "surprise" signals (high-novelty discoveries) to a shared workspace.
- **Pre-emption**: If the Global Workspace identifies a highly corroborated, high-novelty signal, it can pre-empt the standard hierarchical escalation.
- **Cognitive Flexibility**: This allows the system to switch "cognitive focus" instantly, prioritizing paths that offer the most information gain (maximum surprise reduction).

---

## 5. The Sleep: Sleep-Phase Consolidation (REM Cycle)

Knowledge in CEREBRUM is not static. Phase 172 introduces **Sleep-Phase Consolidation**, also known as the **REM Cycle**.

### Hebbian Replay (Phase 96)
During idle cycles, the system "replays" high-quality entries from its Working Memory. It identifies the most successful reasoning paths and applies **Hebbian weight boosts** to the edges on those paths. This "strengthens the synapses" of successful reasoning, making those paths more likely to be chosen in the future.

### Shortcut Synthesis (Phase 172)
The REM Cycle also analyzes the **QueryLog** to find patterns of frequent multi-hop reasoning. If the system frequently follows a path from Node A to Node C via Node B, the REM Cycle will eventually **synthesize a shortcut edge** directly from A to C (tagged as `REM_SHORTCUT`).
- **Efficiency**: These shortcuts allow the system to "jump" over intermediate steps, turning complex reasoning into instantaneous "reflexes."
- **Self-Indexing**: Over time, the graph becomes self-optimizing, adapting its structure to the specific types of questions it is asked most often.

---

## 6. The Brain: Metabolic and Mnemonic Regulation

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

---
**Reviewed on**: May 9, 2026 for version v2.52.0
