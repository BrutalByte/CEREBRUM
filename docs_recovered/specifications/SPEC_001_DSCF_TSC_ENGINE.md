# SPEC_001: The DSCF/TSC Engine
## Multi-Signal Consensus in Graph Partitioning

**Status**: v2.51.0 (Phase 167 (Sleep-Phase Consolidation) COMPLETE)
**Authors**: Bryan Alexander Buchorn · Claude Sonnet 4.6 (Research Collaborator)  
**Field**: Graph Theory / Data Mining / Community Detection  
**Module**: `core/community_engine.py`

---

### 1. Introduction
Traditional community detection algorithms typically optimize for a single structural signal—either local (Label Propagation) or global (Modularity). This leads to a "Resolution Limit" where small, coherent clusters are either overlooked or prematurely merged.

**Dual-Signal Community Fusion (DSCF)** and its evolution, **Triple-Signal Consensus (TSC)**, introduce a consensus-based decision rule that fuses local, global, and flow-based signals at the individual node update level. This specification formalizes the mathematics of this fusion, the temperature-annealing schedule, and the lock-free parallel implementation.

### 2. The Three Consensus Signals

For a node $v$ considering a move to community $C$, the Total Consensus Score $S(v, C)$ is a weighted sum of three distinct topological signals.

#### 2.1 Local Signal: Label Propagation (LPA)
The local signal $\mathcal{L}(v, C)$ represents the topological consensus of node $v$'s immediate neighbors. It is the fraction of neighbors $\mathcal{N}(v)$ that currently belong to community $C$.

$$
\mathcal{L}(v, C) = \frac{\sum_{u \in \mathcal{N}(v)} \delta(\text{label}(u), C)}{|\mathcal{N}(v)|}
$$

*   **Range**: $[0, 1]$
*   **Behavior**: High pressure to conform to the local majority. Stabilizes dense micro-clusters.

#### 2.2 Global Signal: Modularity Gain ($\Delta Q$)
The global signal $\mathcal{G}(v, C)$ represents the change in the graph's total modularity $Q$ if node $v$ moves to community $C$. We use the standard Newman-Girvan definition generalized for weighted graphs:

$$
\Delta Q(v, C) = \left[ \frac{\Sigma_{in}(C) + 2k_{v,in}(C)}{2m} - \left( \frac{\Sigma_{tot}(C) + k_v}{2m} \right)^2 \right] - \left[ \frac{\Sigma_{in}(C)}{2m} - \left( \frac{\Sigma_{tot}(C)}{2m} \right)^2 - \left( \frac{k_v}{2m} \right)^2 \right]
$$

Where:
*   $\Sigma_{in}(C)$: Sum of weights of edges inside $C$.
*   $\Sigma_{tot}(C)$: Sum of weights of edges incident to nodes in $C$.
*   $k_v$: Degree of node $v$.
*   $k_{v,in}(C)$: Sum of weights of edges from $v$ to nodes in $C$.
*   $m$: Total graph weight.

*   **Range**: Approx $[-0.5, 0.5]$ (Normalized to $[0, 1]$ via min-max scaling during iteration).
*   **Behavior**: High pressure to separate distinct clusters and penalize large, amorphous blobs.

#### 2.3 Flow Signal: PageRank-Weighted Centrality (TSC Only)
The third signal $\mathcal{F}(v, C)$ anchors communities to their most structurally significant members. Standard LPA suffers from "Hub Drift" where a central node frequently switches labels, destabilizing its neighbors. TSC weights the local consensus by the PageRank centrality $PR(u)$ of the neighbors:

$$
\mathcal{F}(v, C) = \frac{\sum_{u \in \mathcal{N}(v)} PR(u) \cdot \delta(\text{label}(u), C)}{\sum_{u \in \mathcal{N}(v)} PR(u)}
$$

*   **Range**: $[0, 1]$
*   **Behavior**: Ensures that a node considers the "opinion" of a high-centrality neighbor (like a hub) more valuable than a peripheral leaf node.

### 3. The Temperature-Annealed Consensus Rule

The core innovation of DSCF/TSC is the dynamic weighting of these signals governed by a system temperature $\tau$.

#### 3.1 The Fusion Equation
At iteration $t$, node $v$ computes the score for every candidate community $C$ present in its neighborhood:

$$
S_t(v, C) = \mathcal{L}(v, C) \cdot \tau_t + \widehat{\mathcal{G}}(v, C) \cdot (2 - \tau_t) + \mathcal{F}(v, C) \cdot \gamma
$$

*   $\widehat{\mathcal{G}}$ is the min-max normalized modularity gain.
*   $\gamma$ is the fixed centrality weight (default $0.2$).

#### 3.2 The Annealing Schedule
The temperature $\tau_t$ decays from a high "exploration" value to a low "exploitation" value over $T_{max}$ iterations.

$$
\tau_t = \tau_{start} \cdot \left( \frac{\tau_{end}}{\tau_{start}} \right)^{\frac{t}{T_{max}}}
$$

*   **Phase 1 (High $\tau$)**: The system behaves like LPA. Nodes rapidly form small, locally coherent micro-clusters.
*   **Phase 2 (Medium $\tau$)**: Modularity begins to exert influence, merging micro-clusters that maximize global structure.
*   **Phase 3 (Low $\tau$)**: The system "freezes," optimizing strictly for modularity boundaries while respecting the established local topology.

### 4. Algorithm Implementation

#### 4.1 Data Structures
*   `community_map`: Shared array/map of `node_id` $\rightarrow$ `community_id`.
*   `neighbor_cache`: Adjacency list optimized for fast iteration.
*   `modularity_cache`: Stores $\Sigma_{in}$ and $\Sigma_{tot}$ for each community, updated atomically.

#### 4.2 The Update Loop (Pseudocode)

```python
Initialize each node to its own community (C_v = v)
Compute PageRank scores PR(v)

For t in 0 to T_max:
    current_tau = calculate_tau(t)
    Shuffle(nodes)
    
    For node v in nodes:
        current_comm = community_map[v]
        
        # 1. Identify Candidates
        candidates = {community_map[u] for u in neighbors(v)}
        candidates.add(current_comm)
        
        # 2. Score Candidates
        best_score = -infinity
        best_comm = current_comm
        
        For C in candidates:
            L = compute_LPA(v, C)
            G = compute_ModularityGain(v, C)
            F = compute_Flow(v, C)
            
            score = (L * current_tau) + (G * (2 - current_tau)) + (F * gamma)
            
            if score > best_score:
                best_score = score
                best_comm = C
        
        # 3. Consensus Stay Check
        # Only move if the gain exceeds a stability threshold epsilon
        if best_comm != current_comm and (best_score - score(current_comm) > epsilon):
            community_map[v] = best_comm
            update_modularity_cache(current_comm, best_comm, v)

    If modularity_delta < convergence_threshold:
        Break
```

### 5. Performance Characteristics

*   **Time Complexity**: $O(I \cdot E \cdot \bar{d})$, where $I$ is iterations, $E$ is edges, and $\bar{d}$ is average degree (number of candidates).
*   **Space Complexity**: $O(N)$ for the map + $O(E)$ for the graph.
*   **Convergence**: Guaranteed convergence to a local maximum of the modularity function, bounded by the LPA constraint. The annealing schedule prevents getting trapped in low-quality local optima typical of pure greedy modularity optimization.

### 6. Novelty Claims
1.  **Per-Node Signal Fusion**: Unlike algorithms that alternate phases (e.g., LPA then Modularity), DSCF fuses signals *per atomic update*.
2.  **Flow Anchoring**: The integration of PageRank ($\mathcal{F}$) into the clustering decision explicitly prevents the "floating hub" problem in scale-free graphs.
3.  **Variable Resolution**: The temperature $\tau$ acts as a continuous slider between "Resolution Limit" (Modularity) and "Over-segmentation" (LPA).

> **Note**: This specification covers foundational CEREBRUM architecture. For current implementation status and Phase 69-82 additions, see `CHANGELOG.md` and `docs/ARCHITECTURE.md`.

---
**Copyright © 2026 Bryan Alexander Buchorn. All Rights Reserved.**

---
**Reviewed on**: April 21, 2026 for version v2.51.0
