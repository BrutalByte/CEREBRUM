# Causal Proxy Rationale — `LANGUAGE_GRAPH_CAUSAL_PROXIES`

This document is the authoritative reference for why the proxy relation set used
in `causal_epistemic_benchmark.py` is valid, how it was selected, and why the
benchmark's results cannot be skewed by the selection.

---

## 1. Why a proxy set is needed

`CausalEngine` (Phase 120) restricts graph traversal to relations that carry
**directional causal semantics**.  Its canonical relation set, `CAUSAL_RELATIONS`,
uses biomedical / scientific labels:

```
CAUSES  ACTIVATES  INDIRECTLY_CAUSES  PROMOTES  INDUCES
TRIGGERS  STIMULATES  MAY_CAUSE  LEADS_TO  DRIVES
```

The toy graph (`tests/fixtures/toy_graph.csv`) is a **historical–social knowledge
graph** with 21 entities (scientists, rulers, cities) and 30 edges.  None of the
canonical labels appear in it.  Running `CausalEngine` with the default set would
find zero paths — producing empty metrics that are not comparable to any other
benchmark configuration.

Switching to a different graph would break the apples-to-apples comparison with
the existing **Feature Impact Benchmark** (Phase 77), which also uses
`toy_graph.csv`.  Instead, we define `LANGUAGE_GRAPH_CAUSAL_PROXIES` — a small
set of relations from the toy graph that satisfy the same formal property that
justifies the canonical labels: **directional asymmetry**.

---

## 2. Formal selection criteria

A relation `R` qualifies as a causal proxy if and only if **all three** of the
following conditions hold:

### Criterion 1 — Semantic asymmetry

```
¬∀x,y [ R(x,y) → R(y,x) ]
```

`R(x, y)` does **not** logically imply `R(y, x)`.  In plain English: knowing
that the relationship holds in one direction does not mean it holds in the
reverse.

This is the minimum structural requirement shared by all relations in the
canonical `CAUSAL_RELATIONS` set.  A symmetric relation like `PEERS` cannot be
causal because peerhood provides no information about which entity is the cause
and which is the effect.

### Criterion 2 — Temporal orderability

There exists a plausible domain interpretation under which the **source entity
causally precedes the target entity** when `R(source, target)` holds.

This does not require timestamps to be present in the graph.  It requires only
that the relation name is compatible with the causal arrow of time.
`INFLUENCED(newton, einstein)` is temporally orderable because Newton (1643–1727)
predates Einstein (1879–1955) and the direction of influence in the KG encodes
that earlier actor → later actor.

### Criterion 3 — Irreflexivity

`R(x, x)` should not hold in any meaningful domain interpretation.  An entity
cannot be its own cause along a distinct causal chain.

---

## 3. The selected proxy set

| Relation      | Criterion 1 (Asymmetric)                                      | Criterion 2 (Temporally orderable)                   | Criterion 3 (Irreflexive) |
|---------------|---------------------------------------------------------------|------------------------------------------------------|---------------------------|
| `INFLUENCED`  | If A influenced B, B did not influence A (in the KG encoding) | Influencer predates influenced                        | ✓                         |
| `INSPIRED`    | If A inspired B, B did not inspire A (in the KG encoding)    | Inspiration flows from earlier to later actor         | ✓                         |
| `ANCESTOR_OF` | Genealogical DAG: if A is ancestor of B, B is not of A       | Ancestor predates descendant by definition            | ✓                         |
| `PREDECESSOR` | Antisymmetric by definition: if A precedes B, B cannot precede A | Predecessor is strictly earlier                   | ✓                         |
| `LED`         | If A led B, B did not simultaneously lead A (in this KG)     | Leader acts upon / governs the led entity             | ✓                         |

---

## 4. The excluded symmetric set (negative controls)

These relations appear in the same toy graph and **fail Criterion 1**.  They are
listed in `KNOWN_SYMMETRIC_RELATIONS` and used in the audit as negative controls:

| Relation        | Why symmetric                                                                    |
|-----------------|---------------------------------------------------------------------------------|
| `CONTEMPORARIES`| If x and y are contemporaries, y and x are too — contemporaneity is mutual      |
| `PEERS`         | Peerhood requires both parties to hold the relation                             |
| `NEIGHBORS`     | Spatial adjacency is symmetric by definition                                    |
| `COLLABORATED`  | Collaboration requires mutual participation from both entities                   |
| `CORRESPONDED`  | Correspondence is bilateral — both parties exchange                             |
| `ALLIED`        | Alliance is a mutual commitment; neither party is the "cause" of the other      |
| `RIVALED`       | Rivalry is mutual: if Rome rivals Athens, Athens rivals Rome                    |

---

## 5. Worked examples from the toy graph

### Proxy relations — why the reverse is impossible

**INFLUENCED: newton → einstein**

- Newton lived 1643–1727.  Einstein was born in 1879, a full 152 years after Newton's death.
- `INFLUENCED(newton, einstein)`: valid.  Newton's mechanics directly informed Einstein's
  early study of classical physics before he developed special relativity.
- `INFLUENCED(einstein, newton)`: **impossible**.  Einstein was born 152 years after
  Newton died.  No causal signal can travel backwards in time.

**INSPIRED: alexander → caesar**

- Alexander the Great lived 356–323 BC.  Julius Caesar lived 100–44 BC.
- `INSPIRED(alexander, caesar)`: valid.  Caesar openly admired Alexander and modelled
  his military campaigns on Alexandrian strategy.
- `INSPIRED(caesar, alexander)`: **impossible**.  Caesar was born 256 years after
  Alexander's death.

**ANCESTOR_OF: alexander → cleopatra**

- Alexander's general Ptolemy I founded the Ptolemaic dynasty from which Cleopatra VII
  descended (~280-year separation).
- `ANCESTOR_OF(alexander, cleopatra)`: valid within the KG's genealogical encoding.
- `ANCESTOR_OF(cleopatra, alexander)`: **impossible**.  Genealogical DAGs are acyclic
  by biological definition.  A descendant cannot also be an ancestor of their ancestor.

**PREDECESSOR: caesar → augustus**

- Julius Caesar: 100–44 BC.  Augustus: 63 BC–14 AD.
- `PREDECESSOR(caesar, augustus)`: valid.  Caesar preceded Augustus as Roman ruler.
- `PREDECESSOR(augustus, caesar)`: **impossible by definition**.  The PREDECESSOR
  relation is antisymmetric: if A precedes B then B cannot precede A.

**LED: napoleon → paris**

- Napoleon Bonaparte governed France (1799–1815); Paris is the French capital.
- `LED(napoleon, paris)`: valid.  Napoleon led the French state from Paris.
- `LED(paris, paris)` or `LED(paris, napoleon)`: **impossible**.
  Cities are governed entities, not governing agents.  A city cannot lead a person.

---

### Symmetric controls — why the reverse is equally valid

**CONTEMPORARIES: darwin → curie**

- Darwin lived 1809–1882.  Curie lived 1867–1934.  Overlap: 1867–1882.
- `CONTEMPORARIES(darwin, curie)`: valid — they overlapped for 15 years.
- `CONTEMPORARIES(curie, darwin)`: **equally valid** — contemporaneity is symmetric.
  If Darwin and Curie were contemporaries, then Curie and Darwin were contemporaries.
  The relation carries no directional causal information.

**PEERS: maxwell → bohr**

- Maxwell lived 1831–1879.  Bohr lived 1885–1962.  Both theoretical physicists.
- `PEERS(maxwell, bohr)`: valid within the KG's scientific-domain encoding.
- `PEERS(bohr, maxwell)`: **equally valid** — peerhood is a symmetric relation.

---

## 6. Why the benchmark results cannot be skewed

### 6.1 The proxy set is a named compile-time constant

`LANGUAGE_GRAPH_CAUSAL_PROXIES` is defined as a `frozenset` at the top of
`causal_epistemic_benchmark.py`.  It cannot be modified at runtime via command-line
arguments or environment variables.  Changing the set requires editing the source
code — at which point the three-layer audit would re-execute and would flag any
newly added symmetric relations.

### 6.2 The audit is mandatory and runs first

The benchmark will not emit any metrics if the audit fails.  The audit checks:

- **Layer 1 (definitional)**: `LANGUAGE_GRAPH_CAUSAL_PROXIES ∩ KNOWN_SYMMETRIC_RELATIONS == ∅`.
  This is a hard algebraic gate.  Adding any symmetric relation to the proxy set
  would immediately fail Layer 1.

- **Layer 2 (structural)**: No proxy relation may appear in both forward `(u, v, R)`
  and reverse `(v, u, R)` directions in the source CSV.  Note: the audit reads the
  **raw CSV bytes**, not the NetworkX graph, because the toy graph is stored as an
  undirected `nx.Graph` where `G.has_edge(v, u)` always returns `True` regardless
  of relation semantics.  This design choice is documented explicitly in the audit
  output.

- **Layer 3 (semantic)**: Worked examples with historical dates are printed for
  human verification.  This layer is informational; the hard gates are Layers 1 and 2.

### 6.3 The null-baseline guard

For every positive query in Section 1, the benchmark also runs `CausalEngine` with
an **empty proxy set** (`frozenset()`).  The null engine must return zero paths for
all queries.  If it does not — meaning paths are found even with no allowed relations
— that would indicate an implementation bug (not a skewing attempt), and the
benchmark would flag it.

This guard proves that non-zero `effect_estimate` values are caused by the specific
relations in the proxy set traversing actual edges in the graph, not by any
implementation shortcut that bypasses the relation filter.

### 6.4 Effect-estimate provenance is transparent

Every `CausalProof` returned by `CausalEngine` includes `causal_relations_used` —
the exact relation types that appeared on the paths contributing to the
`effect_estimate`.  Any reviewer can inspect these to confirm that only
`LANGUAGE_GRAPH_CAUSAL_PROXIES` members were used.

### 6.5 Comparison methodology is symmetric

All three benchmark sections (causal accuracy, epistemic calibration, sleep impact)
use the **same graph, the same query sample, and the same random seed** as the
Feature Impact Benchmark (Phase 77).  The proxy set affects only which edges
`CausalEngine` traverses; it does not change the MRR, Hits@1, or EU metrics in
Sections 2 and 3, which use the standard `CerebrumGraph.query()` path.

---

## 7. Extending the proxy set to other graphs

If you run this benchmark on a graph with a different relation vocabulary, select
relations that satisfy all three criteria in Section 2.  Verify your candidates
against the audit before reporting results.

**Good candidates** (directional, temporally orderable, irreflexive):
`CAUSES`, `ACTIVATES`, `TRIGGERS`, `PRODUCED`, `FOUNDED`, `INVENTED`, `DERIVED_FROM`

**Poor candidates** (symmetric or non-causal):
`SIMILAR_TO`, `RELATED_TO`, `CO_OCCURS_WITH`, `INTERACTS_WITH`, `ASSOCIATED_WITH`

The audit's Layer 1 check will automatically reject any overlap with
`KNOWN_SYMMETRIC_RELATIONS`.  For relation types not listed there, apply the
three formal criteria manually and document your reasoning in a comment adjacent
to the frozenset definition.
