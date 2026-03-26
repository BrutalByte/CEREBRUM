"""
tests/test_community_params.py
Hole 2 — Homogeneity Trap: Community-Specific CSA Parameter Overrides.

Validates that CSAEngine uses per-community parameter vectors when provided,
allowing domain-specific attention tuning for heterogeneous KGs.
"""
import math

import numpy as np
import pytest

from adapters.networkx_adapter import NetworkXAdapter
from core.attention_engine import CSAEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter_two_communities():
    import networkx as nx
    G = nx.Graph()
    G.add_edges_from([("A", "B"), ("B", "C"), ("C", "A"),   # community 0
                      ("X", "Y"), ("Y", "Z"), ("Z", "X"),   # community 1
                      ("C", "X")])                           # inter-community edge
    adapter = NetworkXAdapter(G)
    adapter.community_map = {
        "A": 0, "B": 0, "C": 0,
        "X": 1, "Y": 1, "Z": 1,
    }
    adapter.embeddings = {}
    rng = np.random.default_rng(0)
    for n in G.nodes():
        v = rng.standard_normal(8).astype(np.float32)
        adapter.embeddings[n] = v / np.linalg.norm(v)
    return adapter


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_no_community_params_uses_globals():
    adapter = _make_adapter_two_communities()
    csa = CSAEngine(adapter=adapter, alpha=0.4, beta=0.4, gamma=0.1)
    # No override: should use global params
    assert csa._community_params == {}


def test_community_params_stored():
    adapter = _make_adapter_two_communities()
    params = {0: (0.9, 0.05, 0.05, 0.0, 0.0),
              1: (0.1, 0.1, 0.8, 0.0, 0.0)}
    csa = CSAEngine(adapter=adapter, community_params=params)
    assert csa._community_params[0] == (0.9, 0.05, 0.05, 0.0, 0.0)
    assert csa._community_params[1] == (0.1, 0.1, 0.8, 0.0, 0.0)


# ---------------------------------------------------------------------------
# compute_weight uses community-specific params
# ---------------------------------------------------------------------------

def test_community_params_changes_weight():
    """High alpha (semantic similarity) community should weight sim-heavy edges
    differently than high gamma (edge type) community."""
    adapter = _make_adapter_two_communities()

    # Global: balanced
    csa_global = CSAEngine(adapter=adapter, alpha=0.4, beta=0.4, gamma=0.1)
    w_global = csa_global.compute_weight("A", "B", hop=1)

    # Override community 0 with very high alpha
    csa_local = CSAEngine(
        adapter=adapter, alpha=0.4, beta=0.4, gamma=0.1,
        community_params={0: (0.99, 0.01, 0.0, 0.0, 0.0)},
    )
    w_local = csa_local.compute_weight("A", "B", hop=1)

    # Weights should differ due to different alpha values
    assert w_global != w_local


def test_community_params_not_applied_to_other_community():
    """Override for community 0 must not affect nodes in community 1."""
    adapter = _make_adapter_two_communities()
    params = {0: (0.99, 0.01, 0.0, 0.0, 0.0)}

    csa_global = CSAEngine(adapter=adapter, alpha=0.4, beta=0.4, gamma=0.1)
    csa_local  = CSAEngine(adapter=adapter, alpha=0.4, beta=0.4, gamma=0.1,
                           community_params=params)

    # Nodes X, Y are in community 1 (no override) → same weight
    w_global = csa_global.compute_weight("X", "Y", hop=1)
    w_local  = csa_local.compute_weight("X", "Y", hop=1)
    assert abs(w_global - w_local) < 1e-6


def test_community_params_fallback_to_globals_for_unknown_community():
    """Node in community 2 (no override entry) uses global params."""
    import networkx as nx
    G = nx.path_graph(3)
    adapter = NetworkXAdapter(G)
    adapter.community_map = {"0": 2, "1": 2, "2": 2}
    adapter.embeddings = {}
    rng = np.random.default_rng(1)
    for n in G.nodes():
        v = rng.standard_normal(8).astype(np.float32)
        adapter.embeddings[str(n)] = v / np.linalg.norm(v)

    csa_global = CSAEngine(adapter=adapter, alpha=0.4)
    csa_local  = CSAEngine(adapter=adapter, alpha=0.4,
                           community_params={99: (0.9, 0.0, 0.0, 0.0, 0.0)})

    w_global = csa_global.compute_weight("0", "1", hop=1)
    w_local  = csa_local.compute_weight("0", "1", hop=1)
    assert abs(w_global - w_local) < 1e-6


def test_community_params_gamma_dominates_edge_type():
    """Setting gamma=1.0 in community 0 should make edge type weight dominate."""
    adapter = _make_adapter_two_communities()
    adapter.edge_type_weights = {"CAUSES": 1.0}

    csa_low_gamma  = CSAEngine(adapter=adapter, alpha=0.4, beta=0.4, gamma=0.0,
                                community_params={0: (0.0, 0.0, 0.0, 0.0, 0.0)})
    csa_high_gamma = CSAEngine(adapter=adapter, alpha=0.4, beta=0.4, gamma=0.0,
                                community_params={0: (0.0, 0.0, 1.0, 0.0, 0.0)})

    w_low  = csa_low_gamma.compute_weight("A", "B", hop=1,
                                          edge_type="CAUSES",
                                          edge_type_weights={"CAUSES": 1.0})
    w_high = csa_high_gamma.compute_weight("A", "B", hop=1,
                                           edge_type="CAUSES",
                                           edge_type_weights={"CAUSES": 1.0})

    assert w_high > w_low


def test_community_params_interacts_with_snapshot():
    """Community params use _get_community, which respects the query snapshot."""
    adapter = _make_adapter_two_communities()
    params = {
        0: (0.99, 0.0, 0.0, 0.0, 0.0),  # very high alpha
        7: (0.0, 0.99, 0.0, 0.0, 0.0),  # very high beta
    }
    csa = CSAEngine(adapter=adapter, community_params=params)

    # Snapshot maps A → community 7
    csa.set_query_snapshot({"A": 7, "B": 0})
    w_snap = csa.compute_weight("A", "B", hop=1)
    csa.clear_query_snapshot()

    # Without snapshot, A is in community 0
    w_live = csa.compute_weight("A", "B", hop=1)

    # Different community → different params → different weight
    assert abs(w_snap - w_live) > 1e-4


def test_bridge_twin_edge_ignores_community_params():
    """BRIDGE_TWIN shortcut must bypass community param lookup."""
    adapter = _make_adapter_two_communities()
    params = {0: (0.0, 0.0, 0.0, 0.0, 0.0)}  # all zeros → raw=0, sigmoid=0.5
    csa = CSAEngine(adapter=adapter, community_params=params)

    # BRIDGE_TWIN shortcut uses global alpha+beta+gamma (0.4+0.4+0.1=0.9 raw at hop=1)
    # sigmoid(0.925) ≈ 0.716, which is > 0.5 — proves community_params are bypassed
    # (community override is all-zeros which would give sigmoid(0)=0.5 if applied)
    w = csa.compute_weight("A", "B", hop=1, edge_type="BRIDGE_TWIN")
    assert w > 0.5


def test_empty_community_params_dict_behaves_as_global():
    adapter = _make_adapter_two_communities()
    csa_global = CSAEngine(adapter=adapter, alpha=0.4)
    csa_empty  = CSAEngine(adapter=adapter, alpha=0.4, community_params={})
    w_global = csa_global.compute_weight("A", "B", hop=1)
    w_empty  = csa_empty.compute_weight("A", "B", hop=1)
    assert abs(w_global - w_empty) < 1e-6
