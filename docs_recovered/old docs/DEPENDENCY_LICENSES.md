# CEREBRUM — Dependency License Manifest

**Purpose**: Legal compliance record for patent filing, commercial licensing, and acquisition due diligence.
**Maintained by**: Bryan Alexander Buchorn
**Last updated**: March 2026

---

## Summary

| Risk Level | Count | Action Required |
|---|---|---|
| Clean (MIT / BSD / Apache 2.0) | 12 | None — compliant as-is |
| Apache 2.0 (NOTICE required) | 2 | NOTICE file created ✓ |
| GPL (copyleft — fully eliminated) | 0 | Native reimplementation complete ✓ |

**Overall status: COMPLIANT.** All GPL dependencies have been eliminated. `igraph` and `leidenalg` are no longer used — replaced by `core/leiden_native.py`, a clean reimplementation of the Leiden algorithm using only numpy/networkx (BSD). The `[leiden]` optional extra has been removed from `pyproject.toml`.

---

## Core Dependencies (always installed)

| Package | Version | License | Notes |
|---|---|---|---|
| networkx | ≥3.0.0 | BSD 3-Clause | Clean. No attribution requirement beyond NOTICE. |
| numpy | ≥1.24.0 | BSD 3-Clause | Clean. |
| scipy | ≥1.10.0 | BSD 3-Clause | Clean. |
| requests | ≥2.28.0 | Apache 2.0 | Requires NOTICE file. ✓ Created. |
| psutil | ≥5.9.0 | BSD 3-Clause | Clean. |
| PyJWT | ≥2.8.0 | MIT | Clean. |

---

## Leiden Algorithm — GPL Eliminated ✓

| Package | Status | Notes |
|---|---|---|
| igraph / python-igraph | **Removed** | Was GPL v2+. No longer a dependency. |
| leidenalg | **Removed** | Was GPL v3. No longer a dependency. |
| `core/leiden_native.py` | **Original work** | Clean reimplementation of Leiden (Traag et al. 2019) using only numpy/networkx. All rights owned by Bryan Alexander Buchorn. Drop-in replacement for `leiden_communities()`. |

The Leiden *algorithm* (Traag, Waltman & van Eck, 2019) is described in full in a public scientific paper and is free to reimplement. Only the `leidenalg` Python *package* carried GPL. The native reimplementation removes this constraint entirely with no loss of functionality.

---

## Optional Extra: `[embeddings]`

| Package | Version | License | Notes |
|---|---|---|---|
| sentence-transformers | ≥2.2.0 | Apache 2.0 | Requires NOTICE file. ✓ Created. Cites Reimers & Gurevych (2019). |

---

## Optional Extra: `[kge]`

| Package | Version | License | Notes |
|---|---|---|---|
| pykeen | ≥1.10.0 | MIT | Clean. Note: CEREBRUM ships its own TransE/RotatE implementation (`core/kge_engine.py`) and does not require pykeen for core KGE functionality. |

---

## Optional Extra: `[api]`

| Package | Version | License | Notes |
|---|---|---|---|
| fastapi | ≥0.100.0 | MIT | Clean. |
| uvicorn | ≥0.23.0 | BSD 3-Clause | Clean. |
| pydantic | ≥2.0.0 | MIT | Clean. |

---

## Optional Extra: `[neo4j]`

| Package | Version | License | Notes |
|---|---|---|---|
| neo4j | ≥5.8.0 | Apache 2.0 | Requires NOTICE file update if distributed. |

---

## Optional Extra: `[rdf]`

| Package | Version | License | Notes |
|---|---|---|---|
| SPARQLWrapper | ≥2.0 | W3C Software License | Permissive. Clean. |

---

## Optional Extra: `[dev]` (not distributed)

| Package | Version | License | Notes |
|---|---|---|---|
| pytest | ≥7.0 | MIT | Dev only — not distributed in product. |
| pytest-asyncio | ≥0.21 | Apache 2.0 | Dev only. |
| matplotlib | ≥3.7.0 | PSF/BSD | Dev only. |

---

## Algorithm Implementations — Copyright Status

The following algorithms are **implemented from scratch** by the CEREBRUM authors based on published academic papers. No code was copied from any existing implementation. These implementations are original works and are fully owned by Bryan Alexander Buchorn.

| Algorithm | Paper | Our Implementation |
|---|---|---|
| DSCF | Novel — this work | `core/community_engine.py:dscf_communities()` |
| CSA | Novel — this work | `core/attention_engine.py:CSAEngine` |
| Bridge Twin | Novel — this work | `core/bridge_engine.py` |
| Leiden | Traag, Waltman & van Eck (2019) | `core/leiden_native.py:leiden_communities_native()` — clean reimplementation |
| TransE | Bordes et al. (2013) | `core/kge_engine.py:TransEEngine` — clean reimplementation |
| RotatE | Sun et al. (2019) | `core/kge_engine.py:RotatEEngine` — clean reimplementation |
| LPA | Raghavan et al. (2007) | `core/community_engine.py:lpa_communities()` — clean reimplementation via networkx |
| Beam Search | Lowerre (1976) | `reasoning/traversal.py:BeamTraversal` — original implementation |
| PageRank | Page et al. (1999) | Used via networkx (BSD) — `core/structural_encoder.py` |
| Betweenness | Freeman (1977) | Used via networkx (BSD) — `core/structural_encoder.py` |
| Bloom Filter | Bloom (1970) | Original implementation in holographic index |
| STDP | Bi & Poo (1998) | `core/discretizer.py:STDPDiscretizer` — original implementation |

---

---

*This manifest should be updated whenever dependencies are added, removed, or version-pinned.*
