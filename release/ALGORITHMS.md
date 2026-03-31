# CEREBRUM: Algorithm Specification

*Precise mathematical definitions for DSCF, CSA, BeamTraversal, and all Phase 19–20 extensions.*
*Version 1.4.0 — Phase 24 COMPLETE — 1042 tests passing.*

---

## 1. Notation

| Symbol | Definition |
|---|---|
| $G = (V, E)$ | Knowledge graph with entity set $V$ and edge set $E$ |
| $\mathbf{e}_v \in \mathbb{R}^d$ | Embedding of entity $v$ |
| $c(v) \in \mathbb{Z}$ | Community assignment of entity $v$ |
| $\mathcal{C}_i$ | The set of nodes in community $i$ |
| $k_v$ | Degree of node $v$ |
| $m = |E|$ | Number of edges |
| $B$ | Beam width |
| $L$ | Maximum hop depth |
| $\sigma(x)$ | Sigmoid: $1 / (1 + e^{-x})$ |

---

## 2. DSCF: Dual-Signal Community Fusion

### 2.1 Inputs
- Graph $G = (V, E)$
- Number of trials $N$ (default 5)
- Initial temperature $\tau_0$ (default 1.0)
- Minimum temperature $\tau_{min}$ (default 0.01)
- Cooling rate $\rho$ (default 0.92)
- Modularity threshold $\varepsilon$ (default 0.0)
- Maximum iterations $T$ (default 100)

### 2.2 Algorithm

**Initialization**: Assign each node $v$ to its own community: $c(v) \leftarrow v$.

**For each iteration $t = 1, \ldots, T$:**

Shuffle $V$ randomly. For each node $v \in V$:

1. Compute the **global signal** (modularity gain) for assigning $v$ to each candidate community $C$ (i.e., communities of $v$'s neighbors):

$$\Delta Q(v, C) = \frac{k_{v,C}}{m} - \frac{k_v \cdot \text{vol}(C)}{2m^2}$$

where $k_{v,C} = |\{(v,u) \in E : c(u) = C\}|$ and $\text{vol}(C) = \sum_{u \in C} k_u$.

2. Compute the **local signal** (LPA majority fraction) for each candidate community:

$$S_L(v, C) = \frac{|\{u \in \mathcal{N}(v) : c(u) = C\}|}{|\mathcal{N}(v)|}$$

3. Let $C^* = \arg\max_C [\Delta Q(v,C) + S_L(v,C)]$.

4. **Decision rule**:
   - If $\Delta Q(v, C^*) > \varepsilon$: assign $c(v) \leftarrow C^*$ deterministically.
   - Otherwise: assign with probability $P = \sigma\!\left(\frac{\Delta Q(v,C^*) + S_L(v,C^*)}{\tau_t}\right)$.

**Temperature update** (after each full pass over $V$):

$$\tau_{t+1} = \max(\tau_t \times \rho,\; \tau_{min})$$

**Termination**: Stop when fewer than 1% of nodes change assignment in a pass, or at iteration $T$.

**Best-of-N**: Run the full algorithm $N$ times with different random seeds. Return the partition with the highest final modularity $Q$.

### 2.3 Output
- Community map $c : V \to \mathbb{Z}$
- Number of communities $K = |{c(v) : v \in V}|$
- Final modularity $Q$

### 2.4 Complexity
$\mathcal{O}(T \cdot |E| \cdot N)$ per full run. Typically converges in $T < 50$ iterations.

---

## 3. Community Distance Matrix

Given community map $c : V \to \mathbb{Z}$:

1. Build community graph $G_\mathcal{C}$ where nodes are communities and edges exist between communities sharing at least one cross-community edge in $G$.
2. Compute all-pairs shortest path lengths in $G_\mathcal{C}$.
3. $d_\mathcal{C}(i, j) = $ shortest path length between communities $i$ and $j$ in $G_\mathcal{C}$ (or $\infty$ if disconnected).

For large graphs ($K > 2000$), the all-pairs BFS is skipped and a fallback distance of 5.0 is used for non-adjacent community pairs.

---

## 4. CSA: Community-Structured Attention

### 4.1 Inputs
- Source entity $u$, destination entity $v$, hop index $k$
- Edge relation type $r$ with weight $w_r$ (default 1.0)
- Parameters: $\alpha=0.4$, $\beta=0.4$, $\gamma=0.1$, $\delta=0.05$, $\varepsilon=0.05$

### 4.2 Community Score

$$S_\mathcal{C}(u,v) = \begin{cases} 1.0 & \text{if } c(u) = c(v) \\ \exp\!\left(-\lambda \cdot d_\mathcal{C}(c(u), c(v))\right) & \text{if } c(u) \neq c(v) \end{cases}$$

Default $\lambda = 1.0$. Normalized community distance:

$$d_{norm}(u,v) = \min\!\left(\frac{d_\mathcal{C}(c(u),c(v))}{d_{max}},\; 1.0\right)$$

### 4.3 Hop Decay

$$\phi(k) = \frac{1}{1 + k}$$

Values: $\phi(0)=1.0$, $\phi(1)=0.5$, $\phi(2)=0.333$, $\phi(3)=0.25$.

### 4.4 Attention Weight

$$a(u,v,k) = \sigma\!\left(\alpha \cdot \cos(\mathbf{e}_u, \mathbf{e}_v) + \beta \cdot S_\mathcal{C}(u,v) + \gamma \cdot w_r - \delta \cdot d_{norm}(u,v) + \varepsilon \cdot \phi(k) + \zeta \cdot \hat{r}(v)\right)$$

where $\hat{r}(v) = \text{PageRank}(v) / \max_{u \in V} \text{PageRank}(u)$ is the globally normalized PageRank of the destination node. This term gives the beam a gravity signal toward structurally important nodes, closing the recall gap at deep hops without running random walks at query time. PageRank is precomputed once after graph load.

### 4.5 Special Case: Bridge Twin Edges

For edges with relation type `BRIDGE_TWIN`, $\cos(\mathbf{e}_u, \mathbf{e}_v) = 1.0$ and $S_\mathcal{C}(u,v) = 1.0$ by construction. The formula short-circuits to:

$$a_{bridge}(u,v,k) = \sigma\!\left(\alpha + \beta + \gamma + \varepsilon \cdot \phi(k)\right) = \sigma\!\left(0.925 + \varepsilon \cdot \phi(k)\right) \approx 0.716$$

at $k=1$. This is implemented directly for efficiency and numerical stability.

---

## 5. Embedding Aggregation

Along a path $p = (v_0, v_1, \ldots, v_k)$, CEREBRUM maintains a running path embedding:

$$\mathbf{h}_0 = \mathbf{e}_{v_0}$$

$$\mathbf{h}_k = \text{LayerNorm}\!\left(\mathbf{h}_{k-1} + \text{ReLU}(W \mathbf{e}_{v_k} + \mathbf{b})\right)$$

where $W \in \mathbb{R}^{d \times d}$ and $\mathbf{b} \in \mathbb{R}^d$ are fixed random projections (not trained). The residual connection and LayerNorm prevent representation collapse across deep paths.

---

## 6. BeamTraversal

### 6.1 Inputs
- Seed entities $S \subseteq V$
- Beam width $B$ (default 10)
- Maximum hops $L$ (default 3)
- Edge type weights $w_r$ (configurable per relation)

### 6.2 Algorithm

**Initialization**: Beam $= \{(s, [], \mathbf{e}_s, 1.0) : s \in S\}$ — one entry per seed with empty path, seed embedding, and score 1.0.

**For each hop $k = 1, \ldots, L$:**

For each beam entry $(v_{k-1}, \text{path}, \mathbf{h}_{k-1}, \text{score})$:
  - Enumerate neighbors: $\mathcal{N}(v_{k-1})$
  - For each neighbor $v_k$:
    - Compute $a_k = a(v_{k-1}, v_k, k)$
    - Compute $\mathbf{h}_k = \text{LayerNorm}(\mathbf{h}_{k-1} + \text{ReLU}(W\mathbf{e}_{v_k}))$
    - Extended score: $\text{score}' = \text{score} \cdot a_k$
    - Add $(v_k, \text{path} + [(v_{k-1}, r, v_k)], \mathbf{h}_k, \text{score}')$ to candidates

**Pruning rule**: If $k < L$ (not the terminal hop), keep top-$B$ candidates by score. If $k = L$ (terminal hop), retain **all** candidates — pruning at the final step discards valid answers with zero benefit since no further expansion occurs. All beam entries at all hops are retained as candidate answer paths.

### 6.3 Path Scoring

$$\text{score}(p) = \left(\prod_{k=1}^{|p|} a_k\right) \cdot \exp\!\left(-\lambda_c \cdot \bar{d}_\mathcal{C}(p)\right) \cdot \cos(\mathbf{h}_{|p|}, \mathbf{q})$$

where:
- $\bar{d}_\mathcal{C}(p)$ = mean community distance across all hops in $p$ (community coherence term)
- $\mathbf{q}$ = query embedding (if provided; defaults to seed embedding)
- $\lambda_c = 0.1$ (coherence penalty coefficient)

### 6.4 Answer Extraction

Terminal entities are ranked by path score. The top-$K$ distinct terminal entities are returned as answers, each accompanied by the highest-scoring path that reached it.

### 6.5 Complexity

Per query: $\mathcal{O}(B \cdot L \cdot \bar{k} \cdot d)$ where $\bar{k}$ is average node degree.

For a typical production configuration ($B=10$, $L=3$, $\bar{k}=20$, $d=384$): ~230,000 FLOPs. Sub-millisecond on modern CPU.

---

## 7. Structural Encoding (Positional Encoding Analog)

Each entity $v$ is assigned a structural feature vector:

$$\mathbf{s}_v = [\text{PageRank}(v),\; \text{Betweenness}(v),\; \text{Degree}(v) / k_{max}]$$

These features are concatenated with the entity embedding and used by CSAEngine as positional context. They play the same role as positional encodings in Transformers: grounding each entity in its structural position within the graph.

---

## 8. Parameter Summary

| Parameter | Default | Role |
|---|---|---|
| $\alpha$ | 0.4 | Semantic similarity weight |
| $\beta$ | 0.4 | Community score weight |
| $\gamma$ | 0.1 | Edge type weight |
| $\delta$ | 0.05 | Distance penalty weight |
| $\varepsilon$ | 0.05 | Hop decay weight |
| $\zeta$ | 0.1 | Global PageRank prior weight |
| $\lambda$ | 1.0 | Cross-community distance decay |
| $B$ | 10 | Beam width (pruning at intermediate hops only) |
| $L$ | 3 | Maximum hops |
| $N$ | 5 | DSCF best-of-N trials |
| $\tau_0$ | 1.0 | Initial DSCF temperature |
| $\rho$ | 0.92 | DSCF cooling rate |

---

## 9. Bayesian Beam Search (v0.4)

### 9.1 Motivation

When edge weights are near-uniform or evidence is contradictory, the deterministic beam selection collapses: all candidate paths score nearly identically and beam pruning becomes arbitrary. A probabilistic beam selection grounded in Beta distributions adds robustness without changing the API.

### 9.2 Beta Distribution per Path

Each `TraversalPath` maintains a Beta distribution over its score. At initialization:

$$\alpha_0 = \beta_0 = 1.0 \quad \Rightarrow \quad \text{Beta}(1, 1) = \text{Uniform}[0,1]$$

At each edge extension with CSA weight $w_k \in (0,1)$:

$$\alpha \leftarrow \alpha + w_k \qquad \beta \leftarrow \beta + (1 - w_k)$$

This interprets each edge traversal as a Bernoulli trial: weight $w_k$ is the "success" probability, contributing $w_k$ to $\alpha$ and $(1-w_k)$ to $\beta$.

### 9.3 Posterior Statistics

$$\text{Posterior mean: } \mu = \frac{\alpha}{\alpha + \beta}$$

$$\text{Posterior variance: } \sigma^2 = \frac{\alpha \beta}{(\alpha + \beta)^2 (\alpha + \beta + 1)}$$

After $L$ hops with typical weight $\bar{w}$: $\alpha = 1 + L\bar{w}$, $\beta = 1 + L(1-\bar{w})$. Variance decreases monotonically with $L$ — more evidence → tighter distribution.

### 9.4 Thompson Sampling Beam Selection

When `probabilistic=True`:

$$\hat{s}_p \sim \text{Beta}(\alpha_p, \beta_p) \qquad \text{for each candidate path } p$$

The beam is selected as the top-$B$ candidates by sampled score $\hat{s}_p$ rather than the deterministic product score. This implements Thompson sampling: paths with high uncertainty have higher variance in $\hat{s}_p$, giving under-explored paths a chance to be selected.

### 9.5 Answer Uncertainty

`Answer.score_uncertainty` = $\sigma^2$ of the best path at extraction time. Non-zero only when probabilistic mode has been used. Can be surfaced to the LLM bridge or API callers as a confidence indicator.

---

## 10. GlobalRebalancer (v0.4)

### 10.1 Motivation

`IncrementalCommunityUpdater` re-runs DSCF over affected ego-networks after each event. However, accumulated incremental changes cause global modularity $Q$ to drift silently over time — the local updates preserve community assignments but miss global structural shifts.

### 10.2 Drift Detection

Every $N$ events, `GlobalRebalancer` measures:

$$Q = \frac{1}{2m} \sum_{ij} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)$$

using `nx.community.modularity()` on the current `adapter.community_map`. If:

$$|Q_{current} - Q_{last}| > \Delta Q_{threshold} \quad \text{and} \quad t_{now} - t_{last} > t_{min}$$

a full rebalance is triggered.

### 10.3 Background Re-optimization

The rebalance runs `dscf_communities(G)` for $N_{trials}$ independent trials in a daemon thread, selects the partition with the highest $Q$, and commits the new `community_map` under `adapter._lock`. The adapter is readable during the re-run; the update is atomic.

### 10.4 Parameters

| Parameter | Default | Role |
|---|---|---|
| `check_every_n_events` | 200 | Events between Q measurements |
| `drift_threshold` | 0.05 | $\Delta Q$ that triggers a rebalance |
| `min_rebalance_interval` | 60.0 s | Minimum time between full re-runs |
| `n_dscf_trials` | 3 | Best-of-N independent DSCF runs |

---

## 11. Cross-Modal Signal Alignment (v0.4)

### 11.1 Motivation

THALAMUS normalizes text and IDs, but non-textual signals (sensor waveforms, time series, audio) cannot be directly embedded by `EmbeddingEngine`. Without a principled mapping, they must be manually rule-mapped to entities — a fragile and domain-specific process.

### 11.2 Signal Encoders

Two encoders are provided; both output L2-normalized vectors of shape `(entity_dim,)`:

**StatisticalSignalEncoder**: Computes 16 hand-crafted features per signal:
$$\mathbf{f}(x) = [\mu, \sigma, x_{min}, x_{max}, \text{range}, \text{ZCR}, \text{energy}, \text{peaks}, \log(1+|F_1|), \ldots, \log(1+|F_8|)]$$
Projects to entity dimension via a fixed random matrix $W \in \mathbb{R}^{d \times 16}$ (same approach as `RandomEngine`), then L2-normalizes.

**SpectralSignalEncoder**: Computes the FFT magnitude spectrum, applies log compression, pads/truncates to `entity_dim`, and L2-normalizes:
$$\text{emb}(x) = \text{L2norm}\!\left(\log(1 + |\text{rfft}(x)|)_{:d}\right)$$

### 11.3 Procrustes Alignment

Given $n \geq 3$ anchor pairs $(x_i, e_i)$ where $x_i$ are signals and $e_i = \text{adapter.get\_embedding}(id_i)$ are entity embeddings:

Let $A \in \mathbb{R}^{n \times d}$ = stacked raw signal embeddings, $B \in \mathbb{R}^{n \times d}$ = stacked entity embeddings.

The optimal orthogonal rotation $R$ minimizing $\|AR - B\|_F$ is found via SVD:

$$M = A^\top B = U \Sigma V^\top \qquad R = UV^\top$$

After alignment, `encode_signal(x)` returns $\text{L2norm}(R \cdot \text{raw\_encode}(x))$ — a unit vector in the entity embedding space. This vector can be stored directly in `adapter.embeddings` and queried via the standard graph API.

---

## 12. Bayesian Warm-Start (v1.0 — Phase 19)

### 12.1 Motivation

When `BeamTraversal` enters a cold graph segment (few prior traversals, uniform-weight edges), the initial Beta distribution `Beta(1,1)` is a flat uniform prior. Thompson sampling draws from this near-uniform distribution, producing high variance in beam selection at the first hop — effectively making the first step semi-random. The first-hop CSA weight $w_1$ is computed but not used to inform the prior.

### 12.2 Warm-Start Prior Seeding

`BeamTraversal(probabilistic=True, warm_start_strength=s)` applies a scaled first-hop update:

$$\alpha \leftarrow 1 + w_1 \cdot (1 + s) \qquad \beta \leftarrow 1 + (1-w_1) \cdot (1 + s)$$

For subsequent hops $k > 1$, the standard update applies: $\alpha \leftarrow \alpha + w_k$, $\beta \leftarrow \beta + (1-w_k)$.

The parameter `prior_scale = 1.0 + s` is passed to `TraversalPath.copy_with_extension(prior_scale=...)` at the first extension only. Default `s = 0.0` reproduces v0.4.0 behavior exactly.

### 12.3 Effect on Posterior

With `warm_start_strength=5.0` and a first-hop CSA weight $w_1 = 0.8$:

$$\text{Beta}(1 + 0.8 \times 6,\; 1 + 0.2 \times 6) = \text{Beta}(5.8,\; 2.2)$$

Posterior mean: $\mu = 5.8 / 8.0 = 0.725$. Variance: $\sigma^2 = 5.8 \times 2.2 / (8^2 \times 9) \approx 0.0222$.

Without warm-start: $\text{Beta}(1.8, 1.2)$, $\mu = 0.60$, $\sigma^2 = 0.0686$ — more than 3× higher variance. The warm start anchors the beam to the actual CSA score at the first hop, reducing cold-start randomness without biasing downstream accumulation.

### 12.4 Scope of Application

Warm-start applies only when all three conditions hold:
1. `probabilistic=True`
2. `warm_start_strength > 0`
3. The path has exactly one node (i.e., is at the seed — first extension only)

---

## 13. Query Snapshot Isolation (v1.1 — Phase 20)

### 13.1 Motivation

`GlobalRebalancer` commits a new `community_map` atomically under `adapter._lock`. However, a `BeamTraversal.traverse()` call spanning multiple hops may interleave with a rebalance commit: hops 1–2 compute CSA attention weights using community IDs from partition $P_t$, then hop 3 uses IDs from the new partition $P_{t+1}$. Because DSCF IDs are re-assigned sequentially in each run, community ID 7 in $P_t$ may correspond to a completely different cluster in $P_{t+1}$. This produces internally inconsistent attention weights within a single query.

### 13.2 Snapshot Protocol

At the start of `traverse()` and `traverse_stream()`, CEREBRUM captures a shallow copy of the current community map:

$$\text{snapshot} = \{v : c(v)\}_{v \in V} \quad \text{(copy at query start time)}$$

This snapshot is installed into `CSAEngine` via `set_query_snapshot(snapshot)` before the first hop and released via `clear_query_snapshot()` in a `finally` block — guaranteeing release even if the traversal raises an exception.

All community lookups during the traversal use `_get_community(node)`, which resolves against the snapshot:

$$c_\text{query}(v) = \begin{cases} \text{snapshot}[v] & \text{if snapshot is active} \\ \text{adapter.get\_community}(v) & \text{otherwise} \end{cases}$$

### 13.3 Properties

- **Consistency**: All hops within a single query use the same partition. CSA scores for edges $(u, v)$ and $(v, w)$ at different hops are computed in the same community ID space.
- **Isolation**: Rebalancer commits between hops are invisible to the in-flight query. The new partition takes effect for the next query, not the current one.
- **Thread safety**: The snapshot is stored on `CSAEngine` instance state, not on `adapter`. Concurrent queries on separate traversal instances have independent snapshots.
- **Backward compatibility**: When called without a snapshot (e.g., direct `CSAEngine` use in tests), `_get_community` falls back to the adapter, preserving existing behavior.

---

## 14. Community-Specific CSA Parameters (v1.1 — Phase 20)

### 14.1 Motivation

The global CSA parameters $(\alpha, \beta, \gamma, \delta, \varepsilon)$ represent a single trade-off across all communities. In heterogeneous KGs, different sub-graphs have different structural characters: a biomedical community may be semantically dense (favoring high $\alpha$), while a citation community may be structurally sparse (favoring high $\beta$). Using the same parameters for all communities imposes a uniform trade-off — the "homogeneity trap."

### 14.2 Per-Community Parameter Override

`CSAEngine(community_params={cid: (α, β, γ, δ, ε), ...})` stores a dict of per-community parameter vectors. When computing $a(u, v, k)$, the source node's community $c(u)$ is looked up:

$$(\alpha^*, \beta^*, \gamma^*, \delta^*, \varepsilon^*) = \begin{cases} \text{community\_params}[c(u)] & \text{if } c(u) \in \text{community\_params} \\ (\alpha, \beta, \gamma, \delta, \varepsilon) & \text{otherwise (global fallback)} \end{cases}$$

These parameters then replace the global values in the CSA formula for this edge computation only.

### 14.3 Composition with Snapshot Isolation

The community lookup for parameter selection uses `_get_community(u)`, which respects the query snapshot (Section 13). Therefore, snapshot isolation and community-specific parameters compose correctly: within a single query, both the community ID resolution and the parameter selection are consistent.

### 14.4 Special Case: BRIDGE_TWIN Edges

BRIDGE_TWIN edges bypass community parameter lookup entirely. By construction, bridge twin edges have $\cos(\mathbf{e}_u, \mathbf{e}_v) = 1.0$ and $S_\mathcal{C}(u,v) = 1.0$, so the attention formula short-circuits to the global approximation:

$$a_{bridge}(u,v,k) = \sigma(\alpha + \beta + \gamma + \varepsilon \cdot \phi(k)) \approx 0.716 \quad (k=1)$$

This prevents accidentally suppressing validated bridge crossings with zero-weighted community parameter vectors.

---

## 15. Canonical Basis Anchor (v1.1 — Phase 20)

### 15.1 Motivation

`SignalEncoder.learn_alignment()` fits a Procrustes rotation $R$ from signal space to entity embedding space. When two encoders independently fit alignments using different anchor sets, their rotations $R_1$ and $R_2$ may converge to different orientations of the embedding space — both technically correct (each minimizes $\|A_i R_i - B_i\|_F$ for its own anchor set) but mutually incompatible. A signal encoded by encoder 1 occupies a different region of embedding space than the same signal encoded by encoder 2, even though both claim to be in the entity embedding space. This "recursive alignment drift" makes cross-encoder similarity meaningless.

### 15.2 Canonical Embedding Target

`SignalEncoder(canonical_embeddings={entity_id: embedding_vector, ...})` specifies a fixed, shared embedding target for all encoder instances. Instead of fetching anchor entity embeddings from `adapter.get_embedding()` at alignment time:

$$B_\text{anchor}[i] = \text{canonical\_embeddings}[id_i] \quad \text{(fixed, shared)}$$

When a canonical set is provided:
1. The adapter argument to `learn_alignment()` becomes optional — alignment can proceed offline.
2. Both `StatisticalSignalEncoder` and `SpectralSignalEncoder` forward the `canonical_embeddings` parameter to the base class.
3. A bare ID fallback: if `namespace:entity_id` is not found in the canonical set, the lookup retries with the bare `entity_id`.

### 15.3 Correctness Guarantee

All encoder instances sharing the same `canonical_embeddings` dict solve:

$$\underset{R : R^\top R = I}{\min} \|A_i R - B\|_F$$

where $B$ is the **same** matrix for all $i$. Their optimal rotations $R_i^*$ share the same target space. Any two signals encoded by different instances in the same canonical space are directly comparable by cosine similarity.

### 15.4 Federated Alignment

The canonical anchor pattern is the same mechanism used by `FederatedAdapter.align_embeddings()` for aligning remote graph embedding spaces. The canonical embedding set plays the role of "shared anchor nodes" — entities that appear in multiple graphs and whose embeddings serve as the alignment reference. Using consistent canonical embeddings across federated CEREBRUM instances ensures that signals from different sources occupy a coherent shared embedding space.

---

## 16. Path-Preserving Hold-out (v1.1 — Phase 20)

### 16.1 Motivation

`InferenceValidator` evaluates reasoning quality by withholding edges $(u, v)$, running traversal to predict them, and measuring recall. On a sparse graph (average degree $\bar{k} \approx 2$), withholding any edge along a path may sever the only connection between two graph regions. The traversal cannot find a path, scores 0, and the overall recall metric is artificially depressed — a false negative caused by the evaluation procedure itself, not by the reasoning algorithm.

### 16.2 Alternative-Path Check

Before any edge $(u, v)$ is added to the hold-out set, `InferenceValidator._has_alternative_path(G, u, v)` verifies:

1. Remove edge $(u, v)$ from $G$ (temporarily).
2. Check $\text{has\_path}(G \setminus \{(u,v)\}, u, v)$.
3. Restore edge $(u, v)$.
4. Return True iff an alternative path exists.

For undirected graphs, this is equivalent to checking that $(u, v)$ is not a bridge in $G$.

### 16.3 Filtered Hold-out Set

When `InferenceValidator(path_preserving=True)` (the default):

$$\mathcal{H} = \{(u, v, r) \in \mathcal{C} : \text{has\_alternative\_path}(G, u, v)\}$$

where $\mathcal{C}$ is the candidate set from `_find_inferable_edges()`. Only edges with alternative paths are held out; bridge edges are retained in the graph and excluded from evaluation.

When `path_preserving=False`, all candidates in $\mathcal{C}$ are eligible for hold-out regardless of bridge status — reproducing the pre-v1.1 behavior.

### 16.4 Effect on Recall Metrics

On a complete graph (all edges have alternative paths), both modes produce identical hold-out sets. On sparse graphs:

- Path-preserving mode: fewer held-out edges (bridges excluded), but every held-out edge is truly inferable. Recall metrics reflect actual reasoning quality.
- Raw mode: more held-out edges (bridges included), but some edges are structurally impossible to predict from the remaining graph. Recall is underestimated.

The path-preserving mode is conservative by design: it may miss some precision issues on truly bridge-free hold-outs, but it never penalizes the algorithm for failing to predict structurally impossible inferences.
