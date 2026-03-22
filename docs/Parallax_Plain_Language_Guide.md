# PARALLAX EXPLAINED

## A Plain-Language Guide to the Research

**Who this is for**

You're comfortable with computers and technology, but you don't have a background
in machine learning or knowledge graphs. This guide explains what Parallax is,
why it matters, and what we're trying to prove — without assuming prior knowledge.

When the math shows up, we won't skip it. We'll break every formula down to its
individual pieces, name each piece in plain English, and walk through a concrete
numerical example so you can see the numbers actually working.

**Bryan Alexander Buchorn**

March 2026  |  Version 2.1

---

## How to Read This Guide

Each chapter builds on the last. If you're already comfortable with a concept, feel free to skip ahead. Every technical term is explained when it first appears. Formulas appear in clearly marked "Math, Unpacked" sections — you can read around them without losing the narrative, or dive in for a full understanding.

| Chapter | Topic | You'll Understand |
|---|---|---|
| 1 | What is a Knowledge Graph? | The database behind Parallax |
| 2 | What is an LLM? | What ChatGPT and its peers actually are |
| 3 | The problem with current AI reasoning | Why we need something new |
| 4 | What is attention in AI? | The mechanism Parallax borrows |
| 5 | What is community detection? | The other key concept |
| 6 | What is Parallax? | The full picture |
| 7 | Why does it matter? | Real-world stakes |
| 8 | What are we trying to prove? | The research questions |
| 9 | Does it work? | Validation results |
| 10 | The Studio: Seeing it live | The interactive interface |
| 11 | Real-time streaming data | Sensors, signals, and live feeds |
| 12 | What kinds of data can it use? | Beyond spreadsheets |
| 13 | What comes next? | The road ahead |

---

## CHAPTER 1 — What Is a Knowledge Graph?

Before we can explain Parallax, we need to understand the tool it works with. A Knowledge Graph is a structured way to store facts as a web of connected pieces of information.

**Think of it like a city map of facts**

A city map shows places (nodes) connected by roads (edges). A Knowledge Graph shows concepts — people, drugs, laws, companies — connected by named relationships. "Marie Curie" is connected to "Polonium" by the edge "discovered." "Polonium" is connected to "Radioactive" by the edge "exhibits."

Every fact in a Knowledge Graph is stored as a triple: a subject, a relationship, and an object. For example:

| Subject | Relationship | Object |
|---|---|---|
| Marie Curie | discovered | Polonium |
| Polonium | exhibits | Radioactivity |
| Aspirin | inhibits | COX-2 enzyme |
| COX-2 enzyme | overexpressed in | Colorectal cancer |

This structure has powerful advantages. Every fact is explicit. You can update it without rebuilding the whole system. You can verify every statement directly. And because facts are stored as connections, you can follow a chain of connections to discover something that isn't stated anywhere directly — that Aspirin might be relevant to colorectal cancer, for example.

That process of following a chain of connections is called multi-hop reasoning, and it's one of the central things Parallax is designed to do better.

> **Multi-hop reasoning**: Following a trail of facts — A → B → C → D — to reach a conclusion that isn't stated anywhere directly. Like connecting the dots.

---

## CHAPTER 2 — What Is a Large Language Model?

A Large Language Model (LLM) — the technology behind ChatGPT, Claude, and similar tools — works in a completely different way from a Knowledge Graph.

**Think of it like an extremely well-read person with a terrible memory for sources**

An LLM has read billions of pages of text. It understands language deeply and can reason, summarize, and synthesize. But it can't tell you exactly where it learned anything. And sometimes — because it generates text based on patterns rather than verified facts — it confidently states things that aren't true.

The key thing to understand is that an LLM stores knowledge in its weights — billions of numerical parameters adjusted during training. This is opaque. You can't look inside and see "here is where it stores the fact that Marie Curie discovered Polonium." It just knows it, in some distributed, unverifiable way.

This creates a serious problem called hallucination.

> **Hallucination**: When an AI generates a confident, plausible-sounding statement that is factually wrong. The AI has no way to check itself against ground truth, so it can't detect its own errors.

Hallucination isn't a bug in the code that can be fixed. It's a structural consequence of how LLMs represent knowledge. They generate likely text, not verified facts. In casual conversation, hallucination is annoying. In medical, legal, or financial contexts, it can be dangerous.

---

## CHAPTER 3 — The Problem With Current AI Reasoning

So we have two systems with complementary strengths and weaknesses. The natural response is to combine them. And that's exactly what the AI research community has done.

The dominant pattern is called Retrieval-Augmented Generation, or RAG. Here's how it works:

**How a RAG system works:**

1. User asks a question
2. System searches the Knowledge Graph for relevant facts
3. Those facts are handed to the LLM as context
4. The LLM reasons about the question using those facts and generates an answer

This is better than using the LLM alone. But there's a fundamental flaw: the Knowledge Graph is still passive. It just hands facts to the LLM and then stands back. The LLM still does all the reasoning. And the LLM can still hallucinate — inventing connections between the retrieved facts that don't actually exist.

The KG retrieves. The LLM reasons. The KG cannot verify whether the LLM's reasoning is correct.

This is the problem Parallax is designed to solve. The question it asks is: what if the Knowledge Graph did the reasoning itself? What if every intermediate step in a chain of logic was a verified edge in the graph, not a guess by a language model?

---

## CHAPTER 4 — What Is Attention in AI?

To understand Parallax, you need to understand one key mechanism from modern AI: attention. This is the idea at the heart of Transformer models — the architecture behind all major LLMs.

**Attention is about deciding what to focus on**

When you read a sentence, you don't give equal weight to every word. If someone asks "Who wrote Hamlet?", your brain gives high weight to "wrote" and "Hamlet" and low weight to "Who" and the question mark. That's attention — selectively focusing on what matters for the task.

In a Transformer model, attention is computed mathematically for every word (or "token") in a sequence. The model learns which tokens to pay attention to when processing each position. And crucially, it runs this process multiple times in parallel using different "attention heads" — each one looking for different kinds of relationships.

One attention head might specialize in grammatical relationships ("which noun does this verb refer to?"). Another might specialize in long-range semantic relationships ("this pronoun refers to that entity mentioned three sentences ago"). The combination of multiple specialized heads is a large part of what makes Transformers so powerful.

> **Attention head**: A single specialized focus-mechanism within a Transformer. Most LLMs run dozens of these in parallel, each learning to look for different patterns. The results are combined to produce the final output.

**The Parallax Observation**

Here is the central insight of Parallax: Knowledge Graphs already contain the natural equivalent of attention heads. They're called communities.

A community in a graph is a group of nodes that are more connected to each other than to the rest of the graph. In a medical knowledge graph, there might be a community of nodes about "drug mechanisms," another about "clinical symptoms," another about "genetic markers." Each community naturally specializes on a domain — exactly like an attention head specializes on a type of relationship.

Parallax uses this correspondence to let a Knowledge Graph perform the same kind of structured, multi-head reasoning that Transformers do — but using graph traversal instead of matrix multiplication, and using community membership instead of learned numerical weights.

---

## CHAPTER 5 — What Is Community Detection?

Community detection is the process of automatically finding groups of closely connected nodes in a graph. It's a well-studied problem with several standard algorithms, and it's one of the two core technical pieces of Parallax.

**Think of it like finding neighborhoods in a city**

A city organically develops neighborhoods — areas where residents interact with each other more than with people across town. Community detection algorithms do the same thing automatically for any graph, without being told in advance where the neighborhoods are.

The standard algorithms each have trade-offs:

| Algorithm | Strength | Weakness |
|---|---|---|
| Louvain | Fast, produces good communities | Can create disconnected communities |
| Leiden | Promotes connected communities | Optimizes only for global structure |
| LPA | Very fast, purely local | Random results, can over-merge |
| DSCF (new) | Combines local + global signals simultaneously | Non-deterministic (same as LPA) |

**The DSCF Innovation**

DSCF stands for Dual-Signal Community Fusion. It's a new algorithm created as part of the Parallax research.

The key idea: every existing algorithm uses one signal to decide where each node belongs. Either it asks "what are my immediate neighbors doing?" (local, like LPA) or "what assignment maximizes the overall structure score?" (global, like Leiden). DSCF asks both questions simultaneously for every node at every step.

**The Aircraft Analogy: Mid-Level Voting**

The inspiration for this "two-signal" approach comes from the world of aviation. High-stakes aircraft navigation systems use mid-level voting to promote safety. Instead of trusting a single sensor, the flight computer takes inputs from three independent sources and selects the middle value. This "vote" automatically rejects any single sensor that is giving a wrong (outlier) reading.

Parallax applies this same principle to AI reasoning. By requiring consensus between different structural signals, it "rights the navigation errors" — or hallucinations — that are so common in current language models.

This shifts AI from a **Black-Box** (where the logic is hidden inside billions of math weights) to a **Glass-Box** (where every reasoning step is a visible, verifiable path through a graph).

Here's how DSCF makes the decision for each node:

**DSCF decision logic (simplified):**

1. Local signal: "What community are most of my neighbors in right now?" (majority vote)
2. Global signal: "What assignment would improve the overall structure of the whole graph?"
3. If both signals agree: move with high confidence
4. If they disagree: a temperature parameter governs which signal wins (starts high for local, cools toward global)

The temperature annealing is borrowed from metallurgy and simulated annealing in optimization. Start hot (local signal dominates), cool slowly (global signal gets more weight). The result is communities that are both locally tight and globally meaningful — the exact combination that Parallax needs.

---

### The Math, Unpacked: How DSCF Decides Where a Node Belongs

#### Part A — The Global Signal: Modularity Gain

When DSCF asks "would it help the whole graph to move node v into community C?", it calculates a number called modularity gain. If that number is positive, the move is globally beneficial. If it's negative, the move would make the overall structure worse.

**The formula:**

```
ΔQ(v → C) = (edges from v into C) / (total graph edges)
           - ρ × (v's total connections × C's total connections) / (2 × total graph edges)²
```

Written compactly: `ΔQ = k_{v,C}/m − ρ · (k_v · Σk_u) / (2m²)`

Every symbol in plain English:

| Symbol | Meaning | Example value |
|---|---|---|
| `k_{v,C}` | How many of v's edges connect directly into community C | 3 edges |
| `m` | Total number of edges in the entire graph | 100 edges |
| `ρ` | Resolution — controls how large communities can grow (default 1.0) | 1.0 |
| `k_v` | v's total degree — how many edges v has overall | 5 edges |
| `Σk_u` | Sum of degrees of every node already in C | 40 edge-stubs |

**Worked example:**

Node v has 5 edges total. 3 of them go into community C. Community C has nodes whose degrees sum to 40. The graph has 100 edges total.

```
ΔQ = 3/100 − 1.0 × (5 × 40) / (2 × 100²)
   = 0.030 − 200 / 20,000
   = 0.030 − 0.010
   = +0.020
```

`ΔQ = +0.020` is positive, so moving v into C would improve the global structure. The global signal says: move.

Now suppose v had only 1 edge into C (instead of 3):

```
ΔQ = 1/100 − 1.0 × (5 × 40) / 20,000
   = 0.010 − 0.010
   = 0.000
```

No benefit — the global signal is indifferent. The move would be decided by the local signal alone.

The intuition: if you're already well-connected to a community (`k_{v,C}` is large), moving there increases modularity. If C is already very large (`Σk_u` is large), moving there adds diminishing returns — it makes the community even larger but not necessarily better.

---

#### Part B — The Temperature Schedule

DSCF uses a temperature parameter τ (the Greek letter tau) to control how much the local vs. global signal is trusted at each stage of the algorithm. Think of it like baking: high temperature = fluid, exploratory; low temperature = rigid, settled.

**The formula:**

```
τ(next iteration) = max(τ(current) × 0.92, 0.01)
```

In English: each iteration, multiply the temperature by 0.92. But never let it fall below 0.01.

**How it progresses:**

| Iteration | Temperature |
|---|---|
| 0 (start) | 1.00 |
| 5 | 0.66 |
| 10 | 0.43 |
| 20 | 0.19 |
| 30 | 0.08 |
| 50 | 0.01 (floor hit) |

Early in the algorithm (high τ), the local LPA signal carries most of the weight — the algorithm is flexible and willing to move nodes around based on what neighbors are doing right now. Later (low τ), the global modularity signal dominates — the algorithm locks in the large-scale structure and stops making local moves that might undo good global organization.

This is exactly how metallurgical annealing works: heat metal hot enough to let atoms reorganize freely, then cool slowly so they settle into a stable crystalline structure rather than a random frozen mess.

---

#### Part C — When Signals Disagree: The Decision Probability

If both signals agree on where to move node v, DSCF moves it and marks the assignment as "anchored" (high confidence). But when the local and global signals disagree, DSCF uses a probability formula to decide which signal to follow:

**The formula:**

```
P(follow local LPA signal) = (lpa_confidence × τ) / (lpa_confidence × τ + mod_confidence × (2 − τ))
```

Every symbol in plain English:

| Symbol | Meaning |
|---|---|
| `lpa_confidence` | How strong the local majority vote was (e.g., 0.8 = 80% of neighbors agree) |
| `mod_confidence` | How positive the modularity gain ΔQ was (normalized to 0–1) |
| `τ` | Current temperature |
| `2 − τ` | The "complement" — as τ falls, this rises, so global signal gets stronger |

**Worked example — early stage (τ = 0.9):**

LPA says: "80% of my neighbors are in community A" → lpa_confidence = 0.8
Modularity says: "community B has a larger ΔQ" → mod_confidence = 0.6

```
P(follow LPA) = (0.8 × 0.9) / (0.8 × 0.9 + 0.6 × (2 − 0.9))
              = 0.72 / (0.72 + 0.6 × 1.1)
              = 0.72 / (0.72 + 0.66)
              = 0.72 / 1.38
              ≈ 0.52
```

52% chance of following the local LPA signal. Roughly a coin flip, with a slight lean toward local because we're still early.

**Same scenario — late stage (τ = 0.1):**

```
P(follow LPA) = (0.8 × 0.1) / (0.8 × 0.1 + 0.6 × (2 − 0.1))
              = 0.08 / (0.08 + 0.6 × 1.9)
              = 0.08 / (0.08 + 1.14)
              = 0.08 / 1.22
              ≈ 0.066
```

Only 6.6% chance of following local. The algorithm is now 93.4% likely to follow the global modularity signal. The structure has been "annealed" — local moves no longer override the larger picture.

---

#### Part D — TSC: Adding a Third Signal

The Triple-Signal Consensus (TSC) — now the default mode in Parallax — adds a third signal alongside LPA and modularity: flow-based clustering (using an algorithm called Infomap). Where LPA asks "who are my neighbors?" and Leiden asks "what maximizes global structure?", Infomap asks "which way does information naturally flow?"

Think of it as adding a third voice to the vote:

```
P(move) = weighted combination of:
  - LPA signal strength × τ_local
  - Modularity gain × τ_global
  - Infomap flow score × τ_mid
```

A node move only proceeds with high confidence if all three signals substantially agree — the same logic as the aircraft navigation system accepting a sensor reading only when all three sensors produce similar values.

---

## CHAPTER 6 — What Is Parallax?

Now we have all the pieces. Parallax is the framework that puts them together.

**Parallax makes the Knowledge Graph the reasoner. The LLM, if used at all, only writes the final answer in plain English.**

**The Big Inversion**

Every existing hybrid AI system treats the Knowledge Graph as a lookup tool and the LLM as the thinker. Parallax flips this:

| System | Process | Problem |
|---|---|---|
| Standard RAG | KG retrieves. LLM reasons. LLM generates. | LLM can hallucinate during the reasoning step. You can't verify intermediate conclusions. |
| Parallax | KG traverses itself using attention. KG generates a scored reasoning path. LLM (optional) writes the answer in English. | None — every intermediate step is a verified graph edge. Nothing is invented. |

**How Parallax Traverses a Graph**

When you ask Parallax a question, it follows this process:

| Step | What happens |
|---|---|
| 1 | **Find the starting nodes.** Extract the key entities from your question and locate them in the graph. |
| 2 | **Encode their position.** Give each starting entity a score based on its structural importance — how connected it is, how central it is to the whole graph. |
| 3 | **Traverse with attention.** For each hop outward, score every neighboring node using the CSA formula: how similar is it? Is it in the same community (same attention head)? How far away is it? |
| 4 | **Prune the beam.** Keep only the most promising paths at each step (beam search). This prevents the traversal from exploding into too many branches. |
| 5 | **Score the full paths.** Rank completed paths by their total attention score, community coherence, and how well they answer the question. |
| 6 | **Return the results.** Output the top-K paths with their full reasoning chain. Optionally, pass to an LLM to generate a readable answer. |

**A Concrete Example**

Suppose you ask: "Did Marie Curie discover anything radioactive?"

> **Parallax returns a path, not a paragraph:**
>
> Marie Curie `[Scientific Discoveries community]` → **discovered** → Polonium `[Scientific Discoveries community]` → **exhibits** → Radioactivity `[Physics Phenomena community]`
>
> Score: 0.94  |  2 hops  |  1 community transition
>
> Every edge in this path actually exists in the graph. Nothing is invented.

If an LLM bridge is attached, it receives this structured path and writes: "Yes. Marie Curie discovered Polonium, which is a radioactive element." The LLM did not reason — it only translated a verified graph path into English.

**How Cross-Section Connections Work**

A natural question is: how does Parallax find connections between completely separate parts of a graph — different domains, different communities?

The answer is that the traversal doesn't avoid community boundaries. It *discounts* them proportionally. An edge that stays within the same community gets full credit. An edge that crosses into a neighboring community gets half credit. An edge that leaps across distant, unrelated sections gets exponentially less credit. This means:

- Paths that make one meaningful conceptual bridge (drug → gene → disease) remain competitive because the individual hops have high semantic similarity.
- Paths that zigzag incoherently across five unrelated domains compound their penalty with each hop and fall out of the candidate pool within two or three steps.
- A **Bridge Bonus** can be applied to specific edge types (like `INTERACTS_WITH` or `CO_OCCURS_WITH`) to deliberately reward cross-domain connections when the domain structure calls for it.

This is how the same system can reason tightly within a topic or leap productively across topics — it's controlled by what the graph actually contains, not by heuristics.

---

### The Math, Unpacked: The CSA Attention Formula

CSA stands for Community-Structured Attention. This is the formula that decides, at each hop, how much "credit" to give to a potential next step. It produces a single number between 0 and 1: close to 1 means "this is a highly promising next step," close to 0 means "don't go this way."

**The full formula:**

```
a(u, v, k) = sigmoid(
    0.4 × similarity(u, v)        [How similar are they in meaning?]
  + 0.4 × community_score(u, v)   [Are they in the same neighborhood?]
  + 0.1 × edge_type_bonus         [Is this a special relationship type?]
  − 0.05 × normalized_depth       [How deep are we already?]
  + 0.05 × hop_bonus(k)           [Small bonus for early hops]
)
```

The result of the sum inside the sigmoid can be any number. The sigmoid function (σ) compresses any number into the range [0, 1]:

- A sum of +3 or higher → sigmoid gives ≈ 0.95 (very confident: take this step)
- A sum of 0 → sigmoid gives 0.50 (coin flip)
- A sum of −3 or lower → sigmoid gives ≈ 0.05 (very unlikely: skip this step)

Now let's break down each of the five terms.

---

#### Term 1 — Semantic Similarity (weight: 0.4)

**What it measures:** How similar are the two nodes in meaning?

Each entity in the graph is represented as a vector of numbers called an embedding. You can think of an embedding as a point in a high-dimensional space where similar concepts end up close together and dissimilar concepts end up far apart.

The similarity between two vectors is measured by cosine similarity — which measures the angle between them (not the distance). Two vectors pointing in the same direction score 1.0. Two vectors pointing in perpendicular directions score 0.0. Two vectors pointing opposite directions score −1.0.

**Plain English:** "Aspirin" and "ibuprofen" embeddings point in similar directions (both are anti-inflammatory drugs) → high score. "Aspirin" and "moon" embeddings point in very different directions → low score.

**Worked example:**

- `similarity("Aspirin", "COX-2 enzyme") = 0.72` (chemically related)
- `similarity("Aspirin", "Galaxy") = 0.08` (unrelated)

Contribution to formula:
- Related hop: 0.4 × 0.72 = **+0.288**
- Unrelated hop: 0.4 × 0.08 = **+0.032**

The related hop gets nearly 9× the contribution from this term alone.

---

#### Term 2 — Community Score (weight: 0.4)

**What it measures:** Are the two nodes in the same neighborhood, and if not, how far apart are their neighborhoods?

The community score S_C(u,v) has three cases:

| Situation | Score | Why |
|---|---|---|
| u and v are in the same community | 1.0 | Same attention head — maximum reward |
| u and v are in adjacent communities | 0.5 | Neighboring heads — moderate reward |
| u and v are in distant communities | e^(−λ × distance) | Exponential decay with community distance |

`e^(−λ × distance)` is an exponential decay function. If λ = 0.5 and the community distance is 3 hops apart:

```
community_score = e^(−0.5 × 3) = e^(−1.5) ≈ 0.22
```

So a hop across three community boundaries gives only 22% of the credit that a same-community hop gives.

**Contribution to formula:**
- Same community: 0.4 × 1.00 = **+0.400**
- Adjacent community: 0.4 × 0.50 = **+0.200**
- 3 communities away: 0.4 × 0.22 = **+0.088**

This single term can decide whether a path stays competitive or falls out of the beam.

---

#### Term 3 — Edge Type Bonus / Bridge Bonus (weight: 0.1)

**What it measures:** Is this a specially designated relationship type that should be rewarded?

By default this term is 0 (no bonus). But some domains have edge types that reliably signal important cross-domain connections. For example:
- In a biomedical graph, `TREATS` edges connect drugs to diseases — exactly the kind of cross-domain hop you want to make.
- In a security graph, `EXPLOITS` edges connect vulnerabilities to attack methods.

When a Bridge Bonus is configured (e.g., w_rel = 0.4 for `TREATS` edges):
- Contribution: 0.1 × 0.4 = **+0.040**

This small boost can tip the balance for a path that would otherwise be penalized for crossing a community boundary.

---

#### Term 4 — Depth Penalty (weight: −0.05)

**What it measures:** How far into the traversal are we? Longer paths are penalized slightly.

`normalized_depth` runs from 0.0 (first hop) to 1.0 (maximum allowed hops). The penalty grows as traversal depth increases.

**Why this exists:** Given two paths that both lead to the correct answer, Parallax prefers the shorter one. A 2-hop explanation is simpler and more trustworthy than a 5-hop explanation reaching the same conclusion. This reflects Occam's Razor — the simplest sufficient explanation is preferred.

**Contribution at various depths (assuming max hops = 6):**
- Hop 1: depth = 1/6 ≈ 0.17 → penalty = −0.05 × 0.17 = **−0.0083**
- Hop 3: depth = 3/6 = 0.50 → penalty = −0.05 × 0.50 = **−0.025**
- Hop 6: depth = 6/6 = 1.00 → penalty = −0.05 × 1.00 = **−0.050**

These are small penalties. They matter at the margin — they won't kill a good path, but they'll prefer shorter good paths over longer good paths when scores are otherwise similar.

---

#### Term 5 — Hop Decay Bonus (weight: +0.05)

**What it measures:** A small position-based bonus that discounts later hops more than earlier hops.

The hop bonus function φ(k) = 1/(1 + k):
- Hop 1: φ(1) = 1/2 = 0.50 → bonus = 0.05 × 0.50 = **+0.025**
- Hop 2: φ(2) = 1/3 = 0.33 → bonus = 0.05 × 0.33 = **+0.017**
- Hop 5: φ(5) = 1/6 = 0.17 → bonus = 0.05 × 0.17 = **+0.008**

This term provides a gentle gradient that helps score comparison between hops at different depths.

---

#### Putting It All Together: A Worked Example

**Scenario: Hop from "Aspirin" to "COX-2 enzyme" at hop 2 of a 6-hop-max traversal**

- Similarity: 0.72 (chemically related)
- Community score: both in "Drug Mechanisms" community → 1.00
- Edge type: `INHIBITS` edge has no Bridge Bonus → 0.0
- Normalized depth: 2/6 = 0.33
- Hop bonus: φ(2) = 1/3 = 0.33

```
Sum = 0.4 × 0.72  +  0.4 × 1.00  +  0.1 × 0.0  −  0.05 × 0.33  +  0.05 × 0.33
    = 0.288        +  0.400        +  0.0          −  0.017          +  0.017
    = 0.688

a("Aspirin", "COX-2 enzyme", 2) = sigmoid(0.688) ≈ 0.67
```

**Scenario: Hop from "Aspirin" to "Galaxy" at hop 2 (wrong direction)**

- Similarity: 0.08 (completely unrelated)
- Community score: "Drug Mechanisms" vs. "Astronomy" — very distant → 0.05
- Edge type: no bonus → 0.0
- Normalized depth: 0.33 (same)
- Hop bonus: 0.33 (same)

```
Sum = 0.4 × 0.08  +  0.4 × 0.05  +  0.1 × 0.0  −  0.05 × 0.33  +  0.05 × 0.33
    = 0.032        +  0.020        +  0.0          −  0.017          +  0.017
    = 0.052

a("Aspirin", "Galaxy", 2) = sigmoid(0.052) ≈ 0.51
```

A score of 0.67 vs. 0.51. When beam search keeps only the top candidates, "COX-2 enzyme" easily survives and "Galaxy" is pruned away. The system has made the correct choice without any training — purely from the structure of the graph.

---

### The Math, Unpacked: Positional Encoding (Structural Features)

Before traversal starts, each node gets a numerical "structural importance score" based on three graph-theoretic measures:

**PageRank:** How many important nodes link to this node? (Borrowed from Google's original search algorithm.) A node that many high-degree nodes point to has high PageRank. In plain English: important nodes are pointed to by other important nodes.

**Betweenness centrality:** How often does this node sit on the shortest path between other pairs of nodes? High betweenness = a bridge or hub. In plain English: remove this node and many paths between other nodes break.

**Degree:** Simply how many edges the node has. High degree = many direct connections.

These three numbers are combined into a single positional encoding vector for each node. The formula:

```
structural_encoding(v) = [PageRank(v), betweenness(v) / max_betweenness, degree(v) / max_degree]
```

This is analogous to the positional encoding in a Transformer, which tells the model where in the sequence each token sits. In Parallax, the structural encoding tells the attention formula where in the graph each node sits — how important and how central it is.

---

### The Math, Unpacked: Embedding Aggregation Along a Path

As traversal hops from node to node, Parallax builds up a running "summary vector" of the path so far — like a memory of where you've been. This running summary is used at the end to measure how well the path answers the original question.

**The update rule at each hop:**

```
h(at hop k) = LayerNorm( h(at hop k−1) + ReLU( w × embedding(new node) + h(at hop k−1) ) )
```

In English, three things happen at each step:

1. **ReLU:** Compute a proposed update from the new node's embedding. ReLU (Rectified Linear Unit) keeps only positive values — it sets any negative number to zero. This means only "positive information" from the new node gets added to the summary.

2. **Residual connection (the + sign):** Add the proposed update to the previous summary `h(k-1)`. This "residual connection" ensures the path memory never completely forgets earlier hops — even at hop 5, the summary still contains information from hop 1.

3. **LayerNorm:** Normalize the result so all values stay in a comparable range. Without this, path summaries could grow very large for long paths, making comparison between short and long paths meaningless.

The final `h(L)` — the summary after the last hop — is compared to the query embedding using cosine similarity. High similarity means the path ends up "near" the answer in concept-space. This becomes the last factor in the path score.

---

### The Math, Unpacked: Path Scoring

After traversal completes, each candidate path P gets a final score that combines three factors:

**The formula:**

```
score(P) = (product of all hop scores) × (community coherence) × (query alignment)
```

Written out:

```
score(P) = [a(hop 1) × a(hop 2) × ... × a(hop L)] × γ_C(P) × similarity(h_L, query)
```

**Why multiply?**

Unlike addition, multiplication means a single bad hop tanks the whole path. If one hop has an attention weight of 0.10, even if every other hop scores 0.90, the product is dragged down. This models the intuition that a chain of reasoning is only as trustworthy as its weakest link — one invented or implausible step invalidates the whole argument.

**The community coherence factor γ_C(P):**

This multiplier measures whether the path stayed within meaningful conceptual regions or zigzagged incoherently.

```
γ_C(P) = (1/L) × sum over all hops of:
           1.0  if hop stayed in same community
           0.5  if hop crossed a community boundary
```

**Worked example — 3-hop path:**

| Hop | Same community? | Score |
|---|---|---|
| 1 | Yes (Marie Curie → Polonium, both in Scientific Discoveries) | 1.0 |
| 2 | No (Polonium → Radioactivity, crosses to Physics Phenomena) | 0.5 |
| 3 | Yes (Radioactivity → Nuclear decay, stays in Physics Phenomena) | 1.0 |

```
γ_C = (1/3) × (1.0 + 0.5 + 1.0) = (1/3) × 2.5 = 0.833
```

A coherent path with one meaningful domain transition scores 0.833. A path that crosses a different community at every single hop would score (1/3) × (0.5 + 0.5 + 0.5) = 0.50 — a 40% penalty applied before even looking at the query alignment.

**Full worked example — scoring the Curie path:**

Path: Marie Curie → Polonium → Radioactivity

| Component | Calculation | Value |
|---|---|---|
| Hop scores | a(hop1) = 0.91, a(hop2) = 0.87 | |
| Product | 0.91 × 0.87 | 0.792 |
| Community coherence | As above | 0.833 |
| Query alignment | similarity(path summary, "radioactive discovery") | 0.92 |
| **Final score** | 0.792 × 0.833 × 0.92 | **0.607** |

A competing path that reaches the same answer via three unrelated hops might produce a product of 0.50, a coherence of 0.50, and an alignment of 0.80 → score = 0.200. Parallax ranks the direct, coherent path first by a factor of 3. The correct answer rises to the top without any training.

---

## CHAPTER 7 — Why Does This Matter?

The interpretability property of Parallax — every answer is a traceable path of verified facts — is not just an academic nicety. It has direct practical consequences in any domain where you cannot afford to be wrong.

| Domain | Why Parallax's Grounded Reasoning Matters |
|---|---|
| Biomedical | Drug repurposing chains ("Drug X inhibits enzyme Y which is overexpressed in disease Z"). An LLM inventing a plausible-sounding drug-disease link could harm patients. Every step must be citable. |
| Legal | Precedent and statutory cross-reference. Every step in a legal argument must be traceable to an actual source. An invented connection is malpractice. |
| Cybersecurity | Attack path analysis. "How does this exposed service lead to root access?" Every hop in the chain is a real vulnerability relationship. Invented steps waste incident response time. |
| Finance | Regulatory compliance and transaction tracing. Auditors need to see the exact chain of relationships that triggered a flag. A guess is not acceptable. |
| Industrial / IoT | Anomaly correlation across sensors. "This temperature spike and pressure drop — are they connected?" Every link is a real physical relationship, not a statistical guess. |

There's a second reason this matters: cost. Running an LLM at scale is expensive. Every query costs tokens, time, and money. Parallax can answer many types of structured questions — especially multi-hop factual questions — without ever calling an LLM. For organizations with large Knowledge Graphs and high query volumes, this could represent significant operational savings.

And a third reason: adaptability. Because Parallax works with any graph database, any embedding method, and any LLM (or none), it can be deployed alongside existing infrastructure without rebuilding systems from scratch.

---

## CHAPTER 8 — What Are We Trying to Prove?

Research is only as good as the questions it asks. Parallax has three specific hypotheses that will be tested against real data and real competing systems.

> **A hypothesis is a claim you can test and potentially disprove.** Good science doesn't just argue that something works. It specifies exactly what "works" means, measures it, and compares it to the best existing alternatives. If the tests fail, the hypothesis is rejected — and that's also a useful result.

**Hypothesis 1: DSCF Makes Better Attention Heads**

The claim: using DSCF communities as attention heads produces better multi-hop reasoning than using communities from Leiden (global-only) or LPA (local-only).

How it's tested: run all three variants on the same benchmark datasets. Compare their accuracy. If DSCF consistently beats both, Hypothesis 1 is supported.

**Hypothesis 2: CSA Attention Beats Standard Graph Attention**

The claim: the CSA weight formula — which combines embedding similarity and community structure — produces better multi-hop reasoning than Graph Attention Networks (GATs), the current state-of-the-art learned graph attention method.

How it's tested: compare Parallax with CSA against a trained 2-layer GAT baseline on the same graph and same benchmarks. This is a fair comparison because the GAT gets to train on labeled examples while Parallax runs zero-shot.

**Hypothesis 3: Parallax Paths Are More Understandable**

The claim: when humans are shown Parallax's reasoning paths alongside LLM-generated reasoning chains for the same questions, they find Parallax's paths more coherent and easier to verify.

How it's tested: a human evaluation study with at least three independent annotators rating reasoning chains. Inter-annotator agreement is measured to check that the ratings are consistent.

**The Benchmarks**

The hypotheses are tested on three publicly available datasets:

- **MetaQA-2hop**: 118,980 two-hop questions over a movie knowledge graph
- **MetaQA-3hop**: 114,196 three-hop questions — the most challenging test
- **WebQSP**: 4,737 questions over a general knowledge graph
- **Hetionet**: 500,000-edge biomedical knowledge graph (genes, diseases, drugs)

All datasets have published results from previous systems, so a direct comparison is possible without ambiguity.

---

## CHAPTER 9 — Does It Work? (Validation Results)

As of March 2026, Parallax has been validated on several large-scale datasets and is operating at v0.3.0 production-ready status.

**What the tests showed:**

1. **Beam steering works.** Parallax successfully uses community signals to steer its search toward correct answers more effectively than traditional methods.

2. **Medical accuracy.** On the Hetionet biomedical graph, Parallax was over 50% more accurate than simple search (BFS) at finding connections between diseases and genes. For specific tasks like connecting genes to pathways, accuracy exceeded the baseline by over 170%.

3. **Real-world recall.** On the WebQSP general knowledge benchmark, Parallax's ability to keep the correct answer within its top-10 candidates was significantly higher than the baseline.

4. **Zero training required.** Most AI systems require weeks of expensive training on powerful computers before they produce useful results. Parallax achieved all of these results with no training at all — it reasons purely from the structure of the graph it's given.

5. **Learning from failure: the Bridge Bonus.** On the MetaQA movie dataset, we found that community structure can work against you. Movie questions always require jumping between types (Movie → Actor → Director), so staying in the same community is actually wrong. This "Structural Mismatch" finding led to the Bridge Bonus: a mechanism for explicitly rewarding cross-domain edges in domains where those crossings are expected. The system got smarter by studying its own failure.

6. **Production hardening.** Phase 10 added JWT authentication (secure access tokens), a ResourceGovernor (prevents runaway queries from consuming too much compute), and AsyncBeamTraversal (streaming results back to the client in real-time as each hop completes, rather than waiting for the full traversal).

**Experiment Conditions**

To support reproducibility:
- **Processor**: AMD Ryzen 9 9950X3D (16-core research processor)
- **Memory**: 64 GB DDR5 RAM
- **Software**: Windows 11 Pro, Python 3.14.0
- **Tools**: NetworkX, igraph — both random and sentence-transformer embeddings tested

---

## CHAPTER 10 — The Studio: Seeing It Live

Parallax includes an interactive browser-based interface called the Studio. You don't need to write any code to use it.

**What the Studio lets you do:**

- **Load any graph** by uploading a file or typing a file path. Supported formats include spreadsheets, JSON, GraphML, and more — covered in Chapter 12.
- **See the communities form.** After loading, the Studio runs DSCF and displays a color-coded interactive visualization of the graph. Each color represents a community (an attention head). You can zoom, pan, and click nodes to inspect their community assignment.
- **Ask questions.** Type a node name (an entity from your graph) into the query box and click Run. The Studio runs the full reasoning pipeline and shows you the resulting paths — scored, ranked, and explained.
- **Inspect the reasoning trace.** For each answer, the Studio shows the exact path through the graph, the attention weight at each hop, and the community score at each transition. You can see precisely why Parallax ranked one answer above another.
- **View structured results.** A JSON output tab shows the raw path metadata — for integration with downstream systems or export.

**The Interactive Walkthrough Notebook**

For a deeper visual demonstration, an interactive Jupyter Notebook is available at `examples/Validation_Walkthrough.ipynb`. It steps through community formation, attention weight calculation for specific edges, and a full 3-hop reasoning trace with visualizations at each stage.

---

## CHAPTER 11 — Real-Time Streaming Data

One of the most significant recent expansions to Parallax (Phase 11) is the ability to reason over *live, changing data* — not just static graphs loaded from a file.

**The core idea: every event becomes a graph edge**

Any stream of observations can be expressed as a triple: who observed it, what the observation type was, and what was observed. For example:

| Source | Relation | Target |
|---|---|---|
| `temperature_sensor_42` | `READS` | `temperature_HIGH` |
| `camera_north` | `DETECTS` | `person` |
| `server_log` | `REPORTS` | `connection_timeout` |
| `pressure_sensor_12` | `CO_ACTIVATES` | `temperature_sensor_42` |

Each of these becomes an edge in a live knowledge graph. As new events arrive, edges are added. As events age past a configurable time window, edges are removed. The graph at any moment reflects the most recent window of observed reality.

**What kinds of live sources does Parallax support?**

| Source Type | What it connects to |
|---|---|
| File tail | A log file or CSV being written by another process (like `tail -f` in a terminal) |
| HTTP polling | A REST API that returns events when polled |
| WebSocket | A persistent connection that pushes events in real-time |
| MQTT | The standard protocol for IoT sensor networks |
| Python callback | Any custom function — wrap your own data source in one line of code |

**Signal discretization: converting numbers to graph concepts**

Sensors produce continuous numbers (72.5°C, 1013 hPa). Knowledge graphs work with named concepts. Parallax includes a set of discretizers that bridge this gap.

---

### The Math, Unpacked: Signal Discretization

#### Threshold Discretizer — "Is the reading normal, high, or alarming?"

This discretizer classifies any numeric reading into one of four named states, with a "hysteresis" guard to prevent the system from flipping back and forth on noisy signals at a boundary.

**The classification rule:**

```
state(x) = SPIKE   if x > nominal + 3 × scale
           HIGH    if x > nominal + scale
           LOW     if x < nominal − scale
           NORMAL  otherwise
```

Every symbol in plain English:

| Symbol | Meaning | Example |
|---|---|---|
| `x` | The current sensor reading | 87.3°C |
| `nominal` | The expected normal value (μ) | 70.0°C |
| `scale` | The threshold width (σ) | 5.0°C |

**Worked example** — factory temperature sensor, nominal = 70°C, scale = 5°C:

```
Thresholds:
  LOW boundary:   70 − 5  = 65°C
  HIGH boundary:  70 + 5  = 75°C
  SPIKE boundary: 70 + 15 = 85°C
```

| Reading | Calculation | State emitted |
|---|---|---|
| 62°C | 62 < 65 | `temperature_LOW` |
| 71°C | 65 ≤ 71 ≤ 75 | `temperature_NORMAL` |
| 79°C | 79 > 75 | `temperature_HIGH` |
| 92°C | 92 > 85 | `temperature_SPIKE` |

Each time the state *changes*, a new graph edge is emitted: `sensor_42 → READS → temperature_HIGH`.

**The hysteresis guard:** If the reading sits right at 75.1°C (barely HIGH), a tiny vibration in the sensor could cause it to oscillate: HIGH → NORMAL → HIGH → NORMAL, generating a flood of graph edges. Hysteresis prevents this by requiring the signal to move a margin (e.g., 0.3°C) past the boundary before the state actually transitions. Once in HIGH, the reading must fall below 74.7°C before it's reclassified as NORMAL.

---

#### Binning Discretizer — "Which bucket does this reading fall into?"

Instead of named states, this discretizer divides the full range of possible values into N equal-width bins and maps each reading to a bin label. Useful when you want fine-grained continuous tracking without defining named thresholds.

**The formula:**

```
bin(x) = floor( (x − minimum) / (maximum − minimum) × N )
```

Every symbol in plain English:

| Symbol | Meaning |
|---|---|
| `x` | Current reading |
| `minimum` | Lowest expected value (configurable) |
| `maximum` | Highest expected value (configurable) |
| `N` | Number of bins (configurable, e.g., 10) |
| `floor(...)` | Round down to the nearest whole number |

**Worked example** — pressure sensor, range 950–1050 hPa, 10 bins:

```
Range width = 1050 − 950 = 100 hPa
Each bin covers 10 hPa

bin(1013) = floor( (1013 − 950) / 100 × 10 )
          = floor( 63 / 100 × 10 )
          = floor( 6.3 )
          = 6
```

Graph edge emitted: `pressure_sensor → READS → pressure_bin_6`

A reading of 987 hPa → bin 3. A reading of 1048 hPa → bin 9 (the highest). The bin label is the "concept" in the graph. Over time, the graph shows how often each sensor lands in each bin — and community detection finds which sensors cluster together in their bin behavior.

---

#### Co-Activation Discretizer — "Which sensors fire together?"

This discretizer watches for sensors that activate near each other in time. When two sensors fire within a short window and have done so enough times, a `CO_ACTIVATES` edge is emitted between them. This edge represents a discovered correlation — not stated in any configuration, but observed from the data.

**The rule:**

```
emit CO_ACTIVATES(A, B)  if:
    |time(A fired) − time(B fired)| ≤ δ_t       [close enough in time]
    AND count(A co-fired with B) ≥ n_min         [happened often enough to be real]
```

Every symbol in plain English:

| Symbol | Meaning | Typical value |
|---|---|---|
| `δ_t` | Maximum time gap to count as "co-activation" | 0.5 seconds |
| `n_min` | Minimum co-activations before emitting an edge | 5 times |

**Why `n_min`?** Sensors in a noisy factory occasionally fire near each other by pure coincidence. Requiring at least `n_min` co-activations filters out chance coincidences and only creates graph edges for genuine correlations. Think of it as requiring "at least 5 witnesses before filing a report."

**Worked example:**

Sensor A (temperature) fires at t = 10.000 seconds.
Sensor B (pressure) fires at t = 10.043 seconds.

Time gap = |10.043 − 10.000| = 0.043 seconds ≤ 0.5 seconds ✓

This is their 4th co-activation. `n_min = 5`. No edge emitted yet — waiting for one more.

Next time: 5th co-activation. Edge emitted: `temperature_sensor → CO_ACTIVATES → pressure_sensor`.

Now when you query "what is related to the temperature sensor?", Parallax can follow this edge and discover that the pressure sensor is structurally correlated with it — and community detection may have already grouped them into the same community because of it.

---

#### Temporal Sequence Discretizer — "What happened just before this?"

For log analysis or ordered event streams, this discretizer emits `PRECEDES` edges between consecutive events on the same source:

```
event_A → PRECEDES → event_B → PRECEDES → event_C → ...
```

A query like "what typically happens before `connection_timeout`?" becomes a graph traversal that follows `PRECEDES` edges backward from the timeout event node, finding the most common predecessor chains. No statistics are explicitly computed — the structure of the graph itself encodes the sequence information.

---

#### Object Detection Discretizer — "What objects appear together?"

For video or image analysis pipelines, each frame's detected objects become graph nodes:

```
camera_north → DETECTS → person
camera_north → DETECTS → vehicle
person → CO_OCCURS_WITH → vehicle    [both in the same frame]
```

Over time, community detection finds clusters of objects that appear together — people and vehicles, equipment and operators, and so on. A query of "what is typically near a person?" returns the co-occurrence graph without any pre-configuration or training.

---

### The Math, Unpacked: The Sliding Window Buffer

The live graph doesn't grow forever. Parallax maintains a time-bounded, count-bounded window of active edges using the sliding window buffer.

**The rule in English:** "Keep only edges that: (1) arrived within the last Δt seconds, AND (2) are among the N_max most recent edges. Whichever constraint is more restrictive, it wins."

**The formula:**

```
Live edges at time t = {
    all edges e where:
        (current time − time edge e arrived) ≤ Δt     [time window]
    AND total number of live edges ≤ N_max              [count cap]
}
```

**Why reference counting?** The same physical relationship might be observed multiple times within the window. If sensor A co-activates with sensor B at t=5s, t=7s, and t=9s, there are three separate events — but they all represent the same edge `(A, CO_ACTIVATES, B)`. The buffer uses reference counting: each arrival increments the edge's reference count, and each expiry decrements it. The edge is only removed from the graph when the count hits zero — i.e., when the last event referencing it has expired.

**Worked example with time_window = 10 seconds and N_max = 1000:**

```
t=0.0s:  Event 1 arrives: (A, CO_ACTIVATES, B). Edge count: ref=1. Graph: {A-B}.
t=2.0s:  Event 2 arrives: (A, CO_ACTIVATES, B). Edge count: ref=2. Graph: {A-B}.
t=5.0s:  Event 3 arrives: (C, READS, HIGH). Edge count for (C,READS,HIGH): ref=1. Graph: {A-B, C-HIGH}.
t=10.0s: Event 1 expires (10 seconds old). ref(A-B) drops from 2 to 1. Edge still in graph.
t=12.0s: Event 2 expires (10 seconds old). ref(A-B) drops from 1 to 0. Edge removed. Graph: {C-HIGH}.
t=15.0s: Event 3 expires. ref(C-HIGH) drops to 0. Edge removed. Graph: {}.
```

The graph automatically cleans itself up. No manual maintenance required.

---

### The Math, Unpacked: Incremental Community Updates

Running the full DSCF algorithm over the entire graph after every single event would be computationally prohibitive at high event rates. Instead, Parallax uses incremental updates: it re-runs community detection only on the immediate neighborhood of nodes that changed.

**The idea:**

When nodes u and v gain or lose an edge, their community assignments might change — but nodes far away from u and v are unlikely to be affected. So instead of re-running DSCF on all N nodes, we re-run it on a small "ego-network" around the affected nodes.

**The formula:**

```
Local subgraph = {all nodes v where: shortest_path(v, any affected node) ≤ r}
```

In English: "Find every affected node. Draw a circle of radius r around each of them. Only re-run DSCF inside those circles."

With the default radius r = 2, the local subgraph typically contains only tens to hundreds of nodes — even in a graph with millions of edges.

**Batching:** Updates don't trigger on every single event. They accumulate until at least 10 affected nodes have been queued, then run as a batch. This amortizes the overhead across multiple events.

**Result:** Community assignments outside the local subgraph are preserved exactly from the previous full run. This creates a partition that is "locally fresh" (updated around recent changes) and "globally stable" (unchanged in the rest of the graph). This hybrid approach keeps the community map approximately correct at all times without the cost of full recomputation.

**The streaming API**

Three new API endpoints support streaming deployments:

- `POST /stream/ingest` — push a batch of events from any external system
- `GET /stream/status` — get live statistics (events/second, node count, edge count, community count)
- `GET /stream/events` — subscribe to a real-time SSE (Server-Sent Events) feed of all graph mutations

---

## CHAPTER 12 — What Kinds of Data Can Parallax Use?

As of v0.3.0, Parallax can load a knowledge graph from any of the following formats, with no configuration required beyond pointing it at the file:

| Format | Extension | What it looks like |
|---|---|---|
| Comma-separated values | `.csv` | A spreadsheet with source, relation, target columns |
| Tab-separated values | `.tsv` | Same as CSV but with tabs |
| JSON | `.json` | A list of edge objects: `[{"source": "a", "relation": "R", "target": "b"}]` |
| JSON Lines | `.jsonl` | One edge per line as a JSON object |
| GraphML | `.graphml` | Standard XML graph format, used by Gephi and NetworkX |
| GEXF | `.gexf` | Graph Exchange XML Format — another Gephi standard |
| GML | `.gml` | Graph Modeling Language — compact text format |
| Parquet | `.parquet` | Columnar binary format used in data engineering pipelines |
| Excel | `.xlsx` / `.xls` | Microsoft Excel workbook with source, relation, target columns |

The format is detected automatically from the file extension. The only requirement is that the file contains recognizable source, relation, and target columns or equivalent structure.

**Connecting to live databases**

Beyond files, Parallax has adapter plugins for:

- **Neo4j** — the leading graph database, via a direct Bolt connection
- **RDF/SPARQL** — the semantic web standard, including Wikidata and DBpedia
- **Remote Parallax nodes** — federated reasoning across multiple Parallax instances (Phase 6)

---

## CHAPTER 13 — What Comes Next?

Parallax is currently at v0.3.0, with all eleven development phases complete and the system in production-ready condition. The roadmap to v1.0.0 focuses on three areas:

**Performance at true scale**

The current system has been tested on graphs with hundreds of thousands of edges. The next milestone targets millions of edges and billions of events in streaming mode. This requires:
- GPU acceleration for community detection and embedding (hardware hooks are already in place)
- Sparse data structures for the community distance matrix at very high community counts
- Distributed beam traversal across multiple machines

**Richer data types**

The streaming and file adapter work has opened the door to entirely new domains:
- Time-series signals from industrial equipment, medical monitors, and financial feeds
- Video understanding pipelines where object detections become graph events
- Natural language event extraction — converting text into graph triples on the fly

**The research paper**

The formal academic paper is in preparation. It will present the DSCF and CSA algorithms, the benchmark results, and the human evaluation study comparing Parallax reasoning paths against LLM-generated reasoning chains. Publication will make the methodology available to the broader research community.

---

## Acknowledgments & Credits

Parallax stands on the shoulders of decades of foundational research. We explicitly acknowledge the work of:

- **LPA**: Raghavan et al. (2007)
- **Louvain**: Blondel et al. (2008)
- **Leiden**: Traag et al. (2019)
- **GATs**: Veličković et al. (2018)
- **Embeddings**: Bordes et al. (2013), Sun et al. (2019)
- **GraphRAG**: Microsoft Research / Edge et al. (2024)
- **Avionics Engineering**: Mid-level voting systems in triplex-redundant aircraft navigation — the inspiration for multi-signal consensus

---

Questions? Contact the author:

**Bryan Alexander Buchorn** — bryan.alexander@buchorn.com

*End of Guide — Version 2.1 — March 2026*
