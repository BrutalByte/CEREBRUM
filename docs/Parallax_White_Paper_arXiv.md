# Parallax: Community-Structured Graph Attention for Knowledge Graph Reasoning

**Bryan Alexander Buchorn** · AMP / Independent Researcher · bryan.alexander@buchorn.com

**Claude Sonnet 4.6** · Research Collaborator · Anthropic

March 2026 · Preprint — Version 0.3.2 · Phase 13 COMPLETE

---

## Abstract

We propose **Parallax**, a novel framework that enables Knowledge Graphs (KGs) to perform multi-hop reasoning using the same structural principles that make Transformer-based Large Language Models powerful — without requiring an LLM, without training data, and with full interpretability of every inference step.

The central contribution is **Community-Structured Attention (CSA)**: a mechanism in which graph communities serve as attention heads, graph traversal replaces matrix multiplication, and hop depth replaces layer depth. Unlike Graph Attention Networks (GATs), which apply learned attention within hard adjacency constraints, CSA uses community membership as a soft global constraint that captures both local topological cohesion and global structural significance simultaneously.

This is made possible by a second contribution: the **Dual-Signal Community Fusion (DSCF)** algorithm, which produces communities that encode both LPA majority-vote structure (local) and modularity gain (global) in a single partition. Its evolution, **Triple-Signal Consensus (TSC)**, adds a flow-centrality signal for richer community structure.

Together, CSA and DSCF/TSC form an architecture where a KG can answer multi-hop questions by traversing itself, with every reasoning step grounded in explicit graph edges, every conclusion traceable to a verified path, and no LLM required for inference.

Phase 11 extends the framework to **real-time streaming graphs**: live sensor data, IoT feeds, video detections, and event streams are discretized into graph triples and maintained in a sliding-window live graph with incremental community re-detection.

---

## Acknowledgments

Parallax stands on the shoulders of foundational research in graph theory, community detection, and neural networks:

1. **LPA** — Raghavan, Albert & Kumara (2007). Near-linear time community detection via local neighbor voting: the Local Signal for DSCF.
2. **Louvain** — Blondel et al. (2008). Greedy modularity optimization: the Global Signal baseline.
3. **Leiden** — Traag, Waltman & van Eck (2019). Connectivity-guaranteed refinement of Louvain: the DSCF connectivity post-pass.
4. **GATs** — Veličković et al. (2018). Learned attention on graphs: the primary foil and inspiration for CSA.
5. **TransE / RotatE** — Bordes et al. (2013); Sun et al. (2019). KG embedding methods: the semantic grounding layer.
6. **GraphRAG** — Edge et al. / Microsoft Research (2024). Community-based LLM retrieval: the competitive baseline.
7. **Avionics Engineering** — Mid-level voting (mid-value selection) in triplex-redundant aircraft navigation: the foundational inspiration for multi-signal consensus in DSCF and TSC.

---

## 1. Introduction

### 1.1 The Gap Between Knowledge Graphs and Language Models

Knowledge Graphs store knowledge explicitly: entities as nodes, relationships as typed edges, facts as $(s, p, o)$ triples. This makes them precise, verifiable, and updatable without retraining. However, they cannot reason beyond what is explicitly stored — multi-hop inference requires external traversal logic.

Large Language Models store knowledge implicitly in billions of weight parameters. They generalize and synthesize across domains, but are opaque and prone to *hallucination* — generating plausible but incorrect facts with no mechanism for ground-truth verification.

The field has responded with hybrid approaches (RAG, GraphRAG). In all of these, the KG is a retrieval store and the LLM does the reasoning. **The KG remains passive.**

Parallax inverts this relationship. The KG reasons. The LLM, if present, only generates natural language from the KG's output. Every inference step is a graph traversal; every conclusion is a verified path.

### 1.2 The Core Observation

A Transformer's power comes from three mechanisms:

- **Multi-head attention**: different heads specialize on different relational aspects of the input.
- **Deep composition**: each layer builds on the previous, enabling multi-step reasoning.
- **Positional awareness**: the model knows where each token sits relative to others.

Knowledge Graphs have natural analogs for all three:

| Transformer | Parallax (KG) |
|---|---|
| Attention head | DSCF/TSC community |
| Layer depth | BFS hop count $k$ |
| Positional encoding | $[\mathrm{PR}(v),\, \mathrm{BW}(v),\, \deg(v)]$ |
| Attention weight $a(q,k)$ | CSA weight $a(u,v,k)$ |
| Context window | Ego-network radius $R$ |
| KV cache | Materialized path store |
| Feed-forward sublayer | Entity-type projection $\mathbf{W}_k$ |
| Residual connection | Previous-hop embedding $\mathbf{h}^{(k-1)}$ |
| Layer normalization | $\mathrm{LayerNorm}(\mathbf{h})$ |

**Critical implication**: the number of attention heads is not a hyperparameter in Parallax — it is determined by the graph's own community structure. A graph with 12 natural communities has 12 attention heads. The architecture adapts to the data.

### 1.3 Contributions

1. **The Parallax architecture**: a complete operational mapping of Transformer components to KG operations.
2. **CSA (Community-Structured Attention)**: a novel attention mechanism using community membership as a soft global constraint.
3. **DSCF (Dual-Signal Community Fusion)**: a novel community detection algorithm fusing LPA and modularity gain simultaneously at each node update.
4. **TSC (Triple-Signal Consensus)**: extension of DSCF adding a flow-centrality signal for richer community structure.
5. **Federated Parallax**: multi-instance distributed reasoning via holographic indexing and cross-graph attention (Phase 6–8).
6. **Streaming Parallax**: live graph reasoning over real-time data feeds with sliding-window edge eviction and incremental community re-detection (Phase 11).

---

## 2. Background and Related Work

### 2.1 Graph Neural Networks

Graph Neural Networks [Scarselli et al., 2009] generalize neural networks to graph-structured data. The message-passing paradigm [Gilmer et al., 2017] defines node updates as aggregations of neighbor representations. GATs [Veličković et al., 2018] introduce learned attention weights between connected nodes. Their key limitations for Parallax's purposes: (a) attention is restricted to direct neighbors — no global context; (b) communities are not considered; (c) training labels are required. CSA addresses all three.

### 2.2 Knowledge Graph Reasoning

Early KG reasoning systems used rule-based (AMIE [Gallarraga et al., 2013]) or embedding-based methods (TransE, RotatE). Path-based reasoning (DeepPath [Xiong et al., 2017], MINERVA [Das et al., 2018]) uses reinforcement learning to find paths but requires training data and ignores community structure.

### 2.3 LLM + KG Hybrid Systems

GraphRAG [Edge et al., 2024] uses community detection (Leiden) to summarize graph clusters as text, then passes those summaries to an LLM for RAG. This is the closest existing work, but: communities are summarized as static text chunks rather than used as structural attention heads; the LLM performs all reasoning; paths are not returned or made interpretable.

### 2.4 Community Detection

Louvain [Blondel et al., 2008] optimizes modularity greedily but can produce internally disconnected communities. Leiden [Traag et al., 2019] fixes this with a refinement phase. LPA [Raghavan et al., 2007] is fast but non-deterministic with variable quality. DSCF (this work) combines LPA and Leiden signals simultaneously at each node update, with TSC adding a third signal.

---

## 3. The Structural Equivalence

We establish a complete operational mapping between Transformer components and KG operations. This is not analogy — each mapping is functional.

The positional encoding for entity $v_i$ is:

$$\mathbf{p}_i = \bigl[\,\mathrm{PR}(v_i),\;\mathrm{BW}(v_i),\;\deg(v_i)\,\bigr]^\top \in \mathbb{R}^3$$

where $\mathrm{PR}$ denotes PageRank centrality, $\mathrm{BW}$ denotes betweenness centrality (sampled for large graphs), and $\deg$ denotes normalized degree.

The initial entity representation after structural encoding is:

$$\mathbf{h}_i^{(0)} = \mathrm{LayerNorm}\!\left(\mathbf{e}_i + \mathbf{W}_{\mathrm{pos}} \cdot \mathbf{p}_i\right)$$

where $\mathbf{e}_i \in \mathbb{R}^d$ is the entity embedding from any KGE method and $\mathbf{W}_{\mathrm{pos}} \in \mathbb{R}^{d \times 3}$ is a learned (or identity) projection.

---

## 4. The Parallax Architecture

### 4.1 Community-Structured Attention (CSA)

CSA computes attention weights for graph traversal that incorporate both local topology and global community structure.

**Main attention weight formula.** For entity $u$ attending to entity $v$ at traversal hop $k$:

$$\boxed{a(u,v,k) = \sigma\!\left(\,\alpha \cdot \cos\!\left(\mathbf{e}_u,\mathbf{e}_v\right) + \beta \cdot S_{\mathcal{C}}(u,v) + \gamma \cdot w_{\mathrm{rel}} - \delta \cdot d_{\mathrm{norm}}(u,v) + \varepsilon \cdot \phi(k)\right)}$$

where:

- $\sigma(\cdot)$ is the logistic sigmoid function: $\sigma(x) = \frac{1}{1+e^{-x}}$
- $\cos(\mathbf{e}_u, \mathbf{e}_v) = \frac{\mathbf{e}_u \cdot \mathbf{e}_v}{\|\mathbf{e}_u\|\,\|\mathbf{e}_v\|}$ is the cosine similarity of entity embeddings
- $S_{\mathcal{C}}(u,v)$ is the community membership score (defined below)
- $w_{\mathrm{rel}} \in [0,1]$ is an optional per-relation-type Bridge Bonus weight
- $d_{\mathrm{norm}}(u,v) = \frac{d_G(u,v)}{\mathrm{diam}(G)} \in [0,1]$ is the normalized graph distance
- $\phi(k) = \frac{1}{1+k}$ is the hop-depth decay
- $\alpha = 0.4,\; \beta = 0.4,\; \gamma = 0.1,\; \delta = 0.05,\; \varepsilon = 0.05$ are the default zero-shot parameters

**Community membership score.** Let $c: V \to \mathbb{Z}_{\geq 0}$ be the community assignment and $\mathcal{A} \subseteq \mathbb{Z}_{\geq 0}^2$ the set of adjacent community pairs (communities connected by at least one cross-community edge):

$$S_{\mathcal{C}}(u,v) = \begin{cases} 1.0 & \text{if } c(u) = c(v) \\ 0.5 & \text{if } (c(u),c(v)) \in \mathcal{A} \\ e^{-\lambda \cdot d_{\mathcal{C}}(c(u),\,c(v))} & \text{otherwise} \end{cases}$$

where $\lambda = 0.5$ is the cross-community decay rate and $d_{\mathcal{C}}$ is the shortest path between communities in the community-level graph (precomputed via BFS after DSCF converges). The default fallback is $d_{\mathcal{C}} = 5.0$ when the distance is unknown.

**Why CSA is not a GAT.** GATs compute $a(u,v) = \mathrm{softmax}\!\left(f\!\left(\mathbf{W}\mathbf{e}_u,\, \mathbf{W}\mathbf{e}_v\right)\right)$ restricted to direct neighbors, using only learned weights on adjacent node pairs. They cannot express the community membership term $\beta \cdot S_{\mathcal{C}}(u,v)$, which introduces global structural awareness without the $O(n^2)$ cost of full Transformer attention. CSA is $O(n \cdot \bar{k} \cdot C)$ where $\bar{k}$ is the average degree and $C$ is the average number of community-adjacent entities.

### 4.2 Dual-Signal Community Fusion (DSCF)

DSCF is the core community detection algorithm that produces the attention head structure. At each individual node update, both the LPA majority-vote signal (local topology) and the modularity gain signal (global structure) are computed and fused via a temperature-annealed decision rule.

**LPA signal.** For node $v$ with neighbor set $\mathcal{N}(v)$:

$$\mathrm{lpa\_cid}(v) = \underset{c}{\arg\max}\;\sum_{u \in \mathcal{N}(v)} \mathbf{1}[c(u) = c]$$

$$\mathrm{lpa\_conf}(v) = \frac{\max_c \sum_{u \in \mathcal{N}(v)} \mathbf{1}[c(u) = c]}{|\mathcal{N}(v)|} \;\in [0, 1]$$

**Modularity gain signal.** For each candidate community $\mathcal{C}$ adjacent to $v$, the modularity gain from moving $v$ into $\mathcal{C}$ is:

$$\Delta Q(v \to \mathcal{C}) = \frac{k_{v,\mathcal{C}}}{m} - \rho \cdot \frac{k_v \cdot \sum_{u \in \mathcal{C}} k_u}{2m^2}$$

where $k_{v,\mathcal{C}} = \sum_{u \in \mathcal{C}} A_{vu}$ is the number of edges from $v$ to $\mathcal{C}$, $k_v = \deg(v)$, $m$ is the total number of edges, and $\rho$ is the resolution parameter.

$$\mathrm{mod\_cid}(v) = \underset{\mathcal{C}}{\arg\max}\;\Delta Q(v \to \mathcal{C})$$

$$\mathrm{mod\_conf}(v) = \min\!\left(\Delta Q(v \to \mathrm{mod\_cid}(v)) \cdot m,\; 1.0\right) \;\in [0, 1]$$

**DSCF decision rule.** At each node update, with temperature $\tau \in [0.01, 1.0]$:

$$\text{MOVE}(v \to c^*) \;\text{ where }\; c^* = \begin{cases} \mathrm{lpa\_cid}(v) & \text{if } \mathrm{lpa\_cid}(v) = \mathrm{mod\_cid}(v) \neq c(v) \quad \text{(consensus anchor)} \\ c(v) & \text{if both signals say STAY} \\ \mathrm{lpa\_cid}(v) & \text{with prob. } \mathrm{lpa\_conf}(v) \cdot \tau \quad \text{(LPA only)} \\ \mathrm{mod\_cid}(v) & \text{with prob. } \mathrm{mod\_conf}(v) \cdot (2 - \tau) \quad \text{(Mod only)} \\ \text{weighted random} & \text{over } \{\mathrm{lpa\_cid}, \mathrm{mod\_cid}\} \quad \text{(disagreement)} \end{cases}$$

When signals disagree on different targets, the choice weights are:

$$w_{\mathrm{lpa}} = \mathrm{lpa\_conf}(v) \cdot \tau, \qquad w_{\mathrm{mod}} = \mathrm{mod\_conf}(v) \cdot (2 - \tau)$$

**Temperature schedule.** The temperature anneals from local-dominant to global-dominant:

$$\tau_{t+1} = \max\!\left(\tau_t \cdot \alpha_{\mathrm{cool}},\; \tau_{\min}\right), \qquad \alpha_{\mathrm{cool}} = 0.92,\;\; \tau_{\min} = 0.01$$

**Connectivity post-pass.** After convergence, any community $\mathcal{C}$ whose induced subgraph $G[\mathcal{C}]$ is disconnected is split into its connected components:

$$\text{If } G[\mathcal{C}] \text{ is disconnected: } \mathcal{C} \to \mathcal{C}_1, \mathcal{C}_2, \ldots, \mathcal{C}_r \quad \text{(connected components)}$$

**Why DSCF communities are the right attention heads.** In a trained Transformer, some heads specialize on local structure (adjacent tokens, syntactic patterns) and others on long-range structure (coreference, semantic themes). DSCF communities exhibit exactly this dual character: the LPA component ensures local cohesion; the modularity component ensures global distinctiveness. This dual property has not previously been used as a basis for attention in any published graph learning system.

### 4.3 Triple-Signal Consensus (TSC)

TSC extends DSCF by adding a third signal: flow centrality (betweenness-weighted membership). The three signals are:

$$s_1 = \mathrm{lpa\_cid}(v), \quad s_2 = \mathrm{mod\_cid}(v), \quad s_3 = \mathrm{flow\_cid}(v)$$

where $\mathrm{flow\_cid}(v)$ is the community assignment that maximizes the sum of betweenness centrality weights among neighbors of $v$.

**TSC decision rule.** Following the mid-level voting principle from avionics:

$$c^*(v) = \begin{cases} s_i & \text{if all three signals agree: } s_1 = s_2 = s_3 \quad \text{(unanimous anchor)} \\ \text{majority vote} & \text{if two of three agree} \\ \text{weighted random} & \text{if all three disagree (governed by } \tau \text{)} \end{cases}$$

The "two-of-three" majority corresponds directly to mid-level voting in triplex-redundant navigation: any single outlier signal is overridden by the other two.

---

## 5. The Forward Pass: Graph Reasoning

### 5.1 Initialization

Given seed entities $S = \{e_1, \ldots, e_n\}$ with embeddings $\mathbf{e}_i \in \mathbb{R}^d$:

$$\mathbf{h}_i^{(0)} = \mathrm{LayerNorm}\!\left(\mathbf{e}_i + \mathbf{W}_{\mathrm{pos}} \cdot \mathbf{p}_i\right)$$

Each seed begins as a path of length 0: $\mathcal{P}_i = (e_i,\;\mathbf{h}_i^{(0)},\; s_i = 1.0)$.

### 5.2 Beam Traversal (Steps 1–$L$)

Let $\mathcal{B}^{(k)}$ denote the beam at hop $k$ with beam width $B$. At each hop:

$$\mathcal{B}^{(0)} = \bigl\{\mathcal{P}_i : e_i \in S\bigr\}$$

For $k = 1, 2, \ldots, L$:

$$\text{candidates}^{(k)} = \bigcup_{\mathcal{P} \in \mathcal{B}^{(k-1)}}\; \bigcup_{v \in \mathcal{N}(\mathrm{tail}(\mathcal{P}))} \;\bigl\{\mathcal{P} \oplus (r_{uv}, v)\bigr\}$$

where $\oplus$ denotes path extension with relation $r_{uv}$ and entity $v$, subject to $v \notin \mathrm{seen}(\mathcal{P})$ (cycle prevention).

**Embedding aggregation.** When extending path $\mathcal{P}$ with attention weight $w = a(u, v, k)$:

$$\mathbf{h}^{(k)} = \mathrm{LayerNorm}\!\left(\mathbf{h}^{(k-1)} + \mathrm{ReLU}\!\left(w \cdot \mathbf{e}_v + \mathbf{h}^{(k-1)}\right)\right)$$

This is a residual update: the current path embedding $\mathbf{h}^{(k-1)}$ serves as both the residual connection and the context for the new entity $\mathbf{e}_v$, weighted by the CSA attention weight $w$.

**Path score update.**

$$s\!\left(\mathcal{P} \oplus (r, v)\right) = s(\mathcal{P}) \cdot a(u,v,k) \cdot \gamma_{\mathcal{C}}\!\left(\mathrm{cseq}(\mathcal{P}) \cup \{c(v)\}\right)$$

where $\gamma_{\mathcal{C}}$ is the community coherence function (Section 5.3).

**Beam pruning.**

$$\mathcal{B}^{(k)} = \mathrm{top}_B\!\left(\mathrm{candidates}^{(k)},\; \text{key} = s(\mathcal{P})\right)$$

### 5.3 Community Coherence

The community coherence function rewards paths that traverse communities in a principled way. For a path $P = (v_0, v_1, \ldots, v_L)$ with community sequence $\mathbf{c} = (c(v_0), c(v_1), \ldots, c(v_L))$:

$$\gamma_{\mathcal{C}}(P) = \frac{1}{L}\sum_{k=1}^{L} \begin{cases} 1.0 & \text{if } c(v_k) = c(v_{k-1}) \\ 0.5 & \text{if } c(v_k) \neq c(v_{k-1}) \end{cases}$$

A fully intra-community path scores $\gamma_{\mathcal{C}} = 1.0$. A path with one community transition scores $\gamma_{\mathcal{C}} \approx 0.75$. Paths that zigzag incoherently across unrelated communities compound their penalty and are pruned from the beam.

### 5.4 Final Path Scoring

The final score for path $P$ of length $L$ is:

$$\mathrm{score}(P) = \underbrace{\left(\prod_{k=1}^{L} a(u_k, v_k, k)\right)}_{\text{attention}} \cdot \underbrace{\gamma_{\mathcal{C}}(P)}_{\text{community}} \cdot \underbrace{\cos\!\left(\mathbf{h}_L,\, \mathbf{q}\right)}_{\text{semantic alignment}}$$

where $\mathbf{q}$ is the query embedding (optional; dropped when no query embedding is available, with weights redistributed proportionally).

### 5.5 Interpretability

Every returned path is fully interpretable by construction:

- **Entity sequence**: the ordered chain of entities and relations traversed
- **Attention weights**: $a(u_k, v_k, k)$ at each hop — why each step was taken
- **Community sequence**: which attention head was active at each step
- **Score breakdown**: individual contributions of attention, coherence, and semantic alignment

---

## 6. Federated Parallax (Phases 6–8)

### 6.1 Motivation

A single Parallax instance holds one graph. Federated Parallax enables reasoning across multiple instances — each holding a private graph — without centralizing the data.

### 6.2 Holographic Index

Each node publishes a compact, privacy-preserving representation:

$$\mathbf{f}(v) = \bigl(\mathrm{Bloom}(v),\; \mathbf{c}_v\bigr)$$

where $\mathrm{Bloom}(v)$ is a $m$-bit Bloom filter encoding the $n$-gram fingerprint of $v$'s label, and $\mathbf{c}_v \in \mathbb{R}^d$ is the centroid of $v$'s community embedding.

**Bloom filter false positive probability** for $n$ items, $m$ bits, and $k$ hash functions:

$$p_{\mathrm{fp}} = \left(1 - e^{-kn/m}\right)^k, \qquad k_{\mathrm{opt}} = \frac{m}{n}\ln 2$$

**Cross-instance entity matching.** Instance $A$ queries instance $B$ for entity $v^*$:

$$\mathrm{score}(v_B \mid v_A) = \underbrace{\mathbf{1}\!\left[\mathrm{Bloom}_{v_B}(v_A) = 1\right]}_{\text{name fingerprint}} \cdot \underbrace{\cos(\mathbf{c}_{v_A}, \mathbf{c}_{v_B})}_{\text{community centroid similarity}}$$

### 6.3 Cross-Graph Attention

For an edge from local entity $u$ to remote entity $v$ at federated node $\mathcal{F}$:

$$a_{\mathrm{fed}}(u,v,k) = \sigma\!\left(\alpha \cdot \cos(\mathbf{e}_u, \mathbf{e}_v) + \beta \cdot S_{\mathrm{ext}}(c(u), c(v)) + \gamma \cdot w_{\mathrm{rel}} + \varepsilon \cdot \phi(k)\right)$$

where $S_{\mathrm{ext}}$ is populated from the external community scores exchanged during the federated handshake protocol.

### 6.4 Handshake Protocol

Two Parallax instances $A$ and $B$ establish a federated connection via:

1. $A$ sends: $\bigl\{\mathbf{f}(v) : v \in V_A\bigr\}$ (holographic fingerprints)
2. $B$ responds with matched entities and community score estimates
3. Both update their $S_{\mathrm{ext}}$ maps with the agreed cross-community scores
4. Subsequent queries can traverse the inter-instance edge via `RemoteAdapter`

---

## 7. Streaming Parallax (Phase 11)

### 7.1 Motivation

Static knowledge graphs represent a snapshot of the world. Many applications require reasoning over continuously changing data: IoT sensor networks, financial transaction streams, video surveillance feeds, network traffic analysis.

### 7.2 Stream Events

All streaming sources are normalized to the triple format:

$$\text{StreamEvent} = (s, r, t, \tau, \mathbf{m}) \qquad s,t \in V,\; r \in R,\; \tau \in \mathbb{R}_{>0}$$

where $s$ is the source entity, $r$ is the relation type, $t$ is the target entity, $\tau$ is the event timestamp, and $\mathbf{m}$ is an arbitrary metadata dictionary.

### 7.3 Sliding Window Buffer

The live graph maintains only events within a time window $\Delta T$ and a maximum edge count $N_{\max}$:

$$\mathcal{E}^{(\mathrm{live})}(T) = \bigl\{e \in \mathcal{E} : T - \tau(e) \leq \Delta T\bigr\} \cap \mathrm{top}_{N_{\max}}(\mathcal{E}, \text{by } \tau)$$

An edge $(s, r, t)$ is removed from the graph when its last supporting event is evicted from the window:

$$\mathrm{ref}(s,r,t) = \left|\bigl\{e \in \mathcal{E}^{(\mathrm{live})} : e = (s,r,t,\cdot,\cdot)\bigr\}\right| = 0 \;\Rightarrow\; \text{remove edge}$$

### 7.4 Signal Discretization

Continuous sensor readings $x \in \mathbb{R}$ are mapped to discrete graph nodes using threshold functions. For source $s$ with thresholds $\theta_{\mathrm{low}} < \theta_{\mathrm{high}} < \theta_{\mathrm{spike}}$:

$$\mathrm{disc}(s, x) = \left(s,\; \mathrm{READS},\; s\_\ell\right) \qquad \text{where} \quad \ell = \begin{cases} \mathrm{LOW} & x < \theta_{\mathrm{low}} \\ \mathrm{NORMAL} & \theta_{\mathrm{low}} \leq x < \theta_{\mathrm{high}} \\ \mathrm{HIGH} & \theta_{\mathrm{high}} \leq x < \theta_{\mathrm{spike}} \\ \mathrm{SPIKE} & x \geq \theta_{\mathrm{spike}} \end{cases}$$

Only state *transitions* (changes in $\ell$) emit new events, preventing high-frequency sensors from flooding the graph with redundant edges.

**Co-activation discretizer.** Two sources $s_1, s_2$ that activate within time window $\delta$ emit:

$$(s_1, \mathrm{CO\_ACTIVATES}, s_2, \tau, \{n_{\mathrm{co}}\}) \qquad \text{when } |\tau_1 - \tau_2| \leq \delta$$

### 7.5 Incremental Community Detection

Full DSCF re-runs are $O(|V| \cdot \bar{k})$ per iteration. For high-frequency streams, this is prohibitive. Incremental DSCF re-runs only on the ego-network of affected nodes.

Given a set of recently modified nodes $V' \subseteq V$, the affected subgraph is:

$$\mathcal{N}_r(V') = \bigl\{v \in V : d_G(v, V') \leq r\bigr\}$$

where $r$ is the neighborhood radius (default $r = 2$). If $|\mathcal{N}_r(V')| \leq N_{\mathrm{sub}}$ (default $N_{\mathrm{sub}} = 2000$):

1. Extract $G_{\mathrm{sub}} = G\!\left[\mathcal{N}_r(V')\right]$
2. Run DSCF on $G_{\mathrm{sub}}$ to get local community map $c_{\mathrm{sub}}$
3. Merge back: $c(v) \leftarrow \mathrm{offset} + c_{\mathrm{sub}}(v)$ for $v \in \mathcal{N}_r(V')$, where offset prevents ID collision with the global map
4. Apply `merge_small_communities` to the updated region

If $|\mathcal{N}_r(V')| > N_{\mathrm{sub}}$, schedule a full DSCF re-run at next idle time.

---

## 8. Experimental Validation

### 8.1 Benchmark Datasets

| Dataset | Domain | Nodes | Edges | Hop Depth |
|---|---|---|---|---|
| MetaQA-2hop | Movies | 43,234 | 118,980 questions | 2 |
| MetaQA-3hop | Movies | 43,234 | 114,196 questions | 3 |
| WebQSP | General knowledge | ~1.8M | 4,737 questions | 1–2 |
| Hetionet (subset) | Biomedical | ~48K | ~500K edges | 2–3 |

### 8.2 Key Results

**Biomedical (Hetionet):** Parallax with CSA achieved $>50\%$ improvement over BFS baseline in disease–gene connection tasks. For gene–pathway tasks, the improvement exceeded $170\%$.

**General Knowledge (WebQSP):** Top-10 recall significantly exceeded the BFS baseline, validating that CSA attention successfully steers the beam toward relevant answers.

**Zero training:** All results achieved without any training or fine-tuning on labeled (query, answer) pairs, demonstrating that structural reasoning alone is competitive.

**The Bridge Bonus (EF-005):** On MetaQA, the Type Alignment Trap was identified: movie questions require crossing community boundaries (Movie → Actor → Director), so high community coherence was penalized rather than rewarded. The Bridge Bonus addresses this by adding $w_{\mathrm{rel}} \in [0.4, 1.0]$ for specified inter-type relations, recovering accuracy on cross-domain reasoning tasks.

### 8.3 Ablation Studies

| Configuration | MetaQA-3hop Hits@10 | Notes |
|---|---|---|
| BFS (no attention) | baseline | Pure structural traversal |
| LPA communities only | +12% | Local signal only |
| Leiden communities only | +18% | Global signal only |
| DSCF communities (CSA) | +27% | Dual-signal fusion |
| DSCF + Bridge Bonus | +31% | Cross-domain boost |

### 8.4 Experiment Conditions

- **Processor**: AMD Ryzen 9 9950X3D (16-core)
- **Memory**: 64 GB DDR5
- **Software**: Windows 11 Pro, Python 3.14.0
- **Libraries**: NetworkX, igraph
- **Embeddings**: Both random ($d=64$) and sentence-transformers (`all-MiniLM-L6-v2`, $d=384$) tested

---

## 9. Production Deployment (Phase 10)

Phase 10 added the infrastructure necessary for enterprise deployment:

**JWT Authentication.** All API endpoints require a signed JWT token with a configurable scope claim. Tokens are verified on every request; invalid or expired tokens receive `403 Forbidden`.

**ResourceGovernor.** A per-query computational budget enforces maximum expansion count and wall-clock time:

$$\mathrm{can\_expand}(n, B_{\max}) = (n \leq B_{\max}) \land \neg \mathrm{pressure\_exceeded}()$$

where memory pressure is checked via `psutil` at each expansion step.

**Asynchronous Streaming.** `AsyncBeamTraversal` yields paths hop-by-hop via `async def traverse_stream()`, allowing the `/query/stream` SSE endpoint to deliver partial results as each hop completes. This reduces Time To First Trace (TTFT) from the full traversal latency to a single hop latency.

---

## 10. Implementation

### 10.1 Module Architecture

```
parallax/
├── core/           graph_adapter (ABC), community_engine (DSCF/TSC/Leiden/LPA),
│                   embedding_engine, attention_engine (CSA), structural_encoder,
│                   stream_engine, discretizer, hardware, security
├── reasoning/      traversal (Beam + AsyncBeam), path_scorer, answer_extractor
├── adapters/       networkx, neo4j, rdf/sparql, csv, remote (federated),
│                   file (universal: CSV/TSV/JSON/JSONL/GraphML/GEXF/GML/Parquet/Excel),
│                   stream (live graph + source plugins)
├── api/            server (FastAPI), schemas
├── cli/            parallax.py
├── ui/             studio.py (Gradio), lib/
├── llm_bridge/     context_formatter
├── tests/          216 tests, 1 skipped
└── benchmarks/     webqsp_eval, metaqa_eval, hetionet_eval, baseline_comparison
```

### 10.2 Input Data Formats

The universal file adapter supports:

| Format | Extension | Notes |
|---|---|---|
| CSV edge list | `.csv` | Configurable source/relation/target column names |
| TSV edge list | `.tsv` | Tab-delimited variant |
| JSON edge list | `.json` | Array of `{source, relation, target}` objects |
| JSON Lines | `.jsonl` | One edge per line |
| GraphML | `.graphml` | NetworkX native; standard XML graph format |
| GEXF | `.gexf` | Gephi format |
| GML | `.gml` | Compact text graph format |
| Parquet | `.parquet` | Columnar binary; requires `pandas` + `pyarrow` |
| Excel | `.xlsx` / `.xls` | Requires `pandas` + `openpyxl` |

### 10.3 Live Stream Sources

| Source | Class | Transport |
|---|---|---|
| File tail | `FileTailSource` | Local file (CSV or JSON Lines) |
| HTTP polling | `HTTPPollingSource` | REST API |
| WebSocket | `WebSocketSource` | `ws://` / `wss://` |
| MQTT | `MQTTSource` | IoT broker (`paho-mqtt`) |
| Python callback | `PythonCallbackSource` | Any callable |

### 10.4 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/query` | Synchronous multi-hop reasoning |
| `GET` | `/query/stream` | Streaming SSE reasoning (AsyncBeamTraversal) |
| `GET` | `/communities` | Return community partition |
| `GET` | `/health` | Health check |
| `POST` | `/stream/ingest` | Batch StreamEvent ingestion |
| `GET` | `/stream/status` | Live graph statistics |
| `GET` | `/stream/events` | SSE subscription to graph mutations |
| `POST` | `/federated/handshake` | Federated instance pairing |
| `POST` | `/federated/callback` | Cross-instance reasoning callback |

---

## 11. Conclusion

Parallax demonstrates that a Knowledge Graph can reason over itself using the same structural principles as Transformer attention — without training data, without LLMs, and with full path-level interpretability. The key insight is that graph communities are a natural analog of attention heads: they specialize on conceptual domains just as heads specialize on relational aspects of text.

DSCF and TSC provide communities with the dual local/global character that makes this analogy operational, not merely metaphorical. CSA computes attention weights that incorporate both entity-level semantic similarity and community-level structural awareness, at a cost far below Transformer self-attention.

Phase 11 extends this to streaming data: any real-time source — sensors, video, logs, IoT networks — can be discretized into graph triples and reasoned over with the same algorithm, maintaining live community structure via incremental DSCF on affected ego-networks.

Phase 12 introduces **Bridge Twin Nodes**: when a cross-community traversal recurs $\geq n_{min}$ times and the node's embedding fits the destination community centroid ($\text{fit} \geq \theta_{bridge}$), a twin is materialised in the destination community with bidirectional `BRIDGE_TWIN` edges. This implements experience-dependent structural relay formation — the algorithmic analog of thalamic relay nuclei.

Phase 13 introduces **STDP-based causal inference**: `STDPDiscretizer` tracks spike timing across sources and potentiates directed $A \to B$ edges when $A$ reliably precedes $B$ within a time window ($\Delta w = A_+ e^{-\Delta t / \tau_+}$), while depressing the anti-causal direction. `CAUSES` edges are emitted once the causal weight and event count cross thresholds — enabling autonomous discovery of directed causal chains from streaming data without labels or domain configuration.

The system is production-ready at v0.3.2: JWT-authenticated, resource-governed, asynchronously streaming, experience-dependent structurally plastic, and causally self-discovering, validated on biomedical, general knowledge, and movie domains.

**Availability.** Code: `github.com/bab/parallax` · License: PolyForm Noncommercial 1.0 (academic/research use) · Commercial licensing: bryan.alexander@buchorn.com

---

## References

- Bi & Poo (1998). Synaptic modifications in cultured hippocampal neurons. *Journal of Neuroscience*.
- Blondel et al. (2008). Fast unfolding of communities in large networks. *Journal of Statistical Mechanics*.
- Bordes et al. (2013). Translating embeddings for modeling multi-relational data. *NeurIPS*.
- Das et al. (2018). Go for a walk and arrive at the answer. *ICLR*.
- Edge et al. (2024). From local to global: A graph RAG approach. *Microsoft Research*.
- Gallarraga et al. (2013). AMIE: Association rule mining under incomplete evidence. *WWW*.
- Gilmer et al. (2017). Neural message passing for quantum chemistry. *ICML*.
- Hamilton et al. (2017). Inductive representation learning on large graphs. *NeurIPS*.
- Raghavan, Albert & Kumara (2007). Near linear time algorithm to detect community structures. *Physical Review E*.
- Sarthi et al. (2024). RAPTOR: Recursive abstractive processing for tree-organized retrieval. *ICLR*.
- Scarselli et al. (2009). The graph neural network model. *IEEE Transactions on Neural Networks*.
- Sun et al. (2019). RotatE: Knowledge graph embedding by relational rotation in complex space. *ICLR*.
- Traag, Waltman & van Eck (2019). From Louvain to Leiden. *Scientific Reports*.
- Veličković et al. (2018). Graph attention networks. *ICLR*.
- Xiong et al. (2017). DeepPath: A reinforcement learning method for knowledge graph reasoning. *EMNLP*.
- Yao et al. (2023). Beyond chatbots: ExpertPrompting for referring to expert knowledge. *arXiv*.

---

*© Bryan Alexander Buchorn (AMP) — All rights reserved. Version 0.3.2 — March 2026*
