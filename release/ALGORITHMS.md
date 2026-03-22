# Parallax: Algorithm Specification

*Precise mathematical definitions for DSCF, CSA, and BeamTraversal.*

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

$$a(u,v,k) = \sigma\!\left(\alpha \cdot \cos(\mathbf{e}_u, \mathbf{e}_v) + \beta \cdot S_\mathcal{C}(u,v) + \gamma \cdot w_r - \delta \cdot d_{norm}(u,v) + \varepsilon \cdot \phi(k)\right)$$

### 4.5 Special Case: Bridge Twin Edges

For edges with relation type `BRIDGE_TWIN`, $\cos(\mathbf{e}_u, \mathbf{e}_v) = 1.0$ and $S_\mathcal{C}(u,v) = 1.0$ by construction. The formula short-circuits to:

$$a_{bridge}(u,v,k) = \sigma\!\left(\alpha + \beta + \gamma + \varepsilon \cdot \phi(k)\right) = \sigma\!\left(0.925 + \varepsilon \cdot \phi(k)\right) \approx 0.716$$

at $k=1$. This is implemented directly for efficiency and numerical stability.

---

## 5. Embedding Aggregation

Along a path $p = (v_0, v_1, \ldots, v_k)$, Parallax maintains a running path embedding:

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

Keep top-$B$ candidates by score. These form the beam for hop $k+1$.

All beam entries at all hops are retained as candidate answer paths.

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
| $\lambda$ | 1.0 | Cross-community distance decay |
| $B$ | 10 | Beam width |
| $L$ | 3 | Maximum hops |
| $N$ | 5 | DSCF best-of-N trials |
| $\tau_0$ | 1.0 | Initial DSCF temperature |
| $\rho$ | 0.92 | DSCF cooling rate |
