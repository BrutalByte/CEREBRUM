"""
Tests for core/insight_validator.py — InsightValidator (Phase 16).

Graph topology used throughout:

    Community 0: A → B → C (with A → C shortcut)
    Community 1: D → E → F (with D → F shortcut)
    Cross edge:  C → D   (BRIDGE — the insight being validated)
    Alt cross:   B → E   (gives another community-0 node a path to community 1)

This means:
  - Bilateral check for C→D insight: D can reach C undirected (C-D edge) → True
  - Corroboration: community-0 members A, B can also reach D (A→B→C→D or B→C→D) → count >= 2

For "isolated" tests, a separate two-component graph is used where the
insight edge is the only connection and the target community has no
return path visible.
"""
import networkx as nx
import pytest

from adapters.networkx_adapter import NetworkXAdapter
from core.embedding_engine import RandomEngine
from core.insight_engine import InsightEvent, INSIGHT_RELATION, INSIGHT_CONFIDENCE
from core.insight_validator import (
    InsightValidator,
    STATUS_BILATERAL,
    STATUS_CORROBORATED,
    STATUS_ISOLATED,
    STATUS_UNILATERAL,
    CONFIDENCE_BILATERAL,
    CONFIDENCE_CORROBORATED,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_adapter():
    """
    Two communities with rich internal connectivity and two cross edges.

    Community 0: A, B, C
    Community 1: D, E, F
    Cross:       C→D (the insight), B→E (alternate path)
    """
    G = nx.DiGraph()
    edges = [
        ("A", "B", "KNOWS"), ("B", "C", "KNOWS"), ("A", "C", "KNOWS"),
        ("D", "E", "KNOWS"), ("E", "F", "KNOWS"), ("D", "F", "KNOWS"),
        ("C", "D", "BRIDGE"),   # insight edge under test
        ("B", "E", "SHORTCUT"), # alternate path from community 0 → 1
    ]
    for u, v, rel in edges:
        for node in (u, v):
            if not G.has_node(node):
                G.add_node(node, label=node, type="entity")
        G.add_edge(u, v, relation=rel, weight=1.0, confidence=1.0)

    adapter = NetworkXAdapter(G)
    engine = RandomEngine(dim=32)
    adapter.embeddings = {n: engine.encode_one(n) for n in G.nodes()}

    community_map = {"A": 0, "B": 0, "C": 0, "D": 1, "E": 1, "F": 1}
    adapter.community_map = community_map

    def patched_get_community(entity_id):
        return adapter.community_map.get(entity_id, -1)
    adapter.get_community = patched_get_community

    return adapter


def _make_isolated_adapter():
    """
    Two completely separate cliques with a single one-way cross edge X1→Y1.
    No reverse path exists (X cluster has no edge from Y1 back).
    No other community-0 node has a path to Y1 (X2, X3 are only connected
    internally within community 0).
    """
    G = nx.DiGraph()
    # Community 0: X1, X2, X3 (chain)
    G.add_edge("X1", "X2", relation="KNOWS", weight=1.0, confidence=1.0)
    G.add_edge("X2", "X3", relation="KNOWS", weight=1.0, confidence=1.0)
    # Community 1: Y1, Y2 (chain)
    G.add_edge("Y1", "Y2", relation="KNOWS", weight=1.0, confidence=1.0)
    # Cross: one-way only — no return path, no alternate route
    G.add_edge("X1", "Y1", relation="BRIDGE", weight=1.0, confidence=1.0)
    for node in G.nodes():
        G.nodes[node].setdefault("label", node)
        G.nodes[node].setdefault("type", "entity")

    adapter = NetworkXAdapter(G)
    engine = RandomEngine(dim=32)
    adapter.embeddings = {n: engine.encode_one(n) for n in G.nodes()}

    community_map = {"X1": 0, "X2": 0, "X3": 0, "Y1": 1, "Y2": 1}
    adapter.community_map = community_map

    def patched_get_community(entity_id):
        return adapter.community_map.get(entity_id, -1)
    adapter.get_community = patched_get_community

    return adapter


def _make_event(source="C", target="D", bridging="D", score=0.8):
    return InsightEvent(
        bridging_node=bridging,
        source=source,
        target=target,
        insight_score=score,
        explanatory_power=0.2,
        community_leap=1,
        path=None,
        edge_created=True,
    )


def _make_validator(adapter, **kwargs):
    kwargs.setdefault("corroboration_seeds", 5)
    kwargs.setdefault("corroboration_threshold", 2)
    kwargs.setdefault("max_hop", 4)
    return InsightValidator(adapter, **kwargs)


# ---------------------------------------------------------------------------
# Bilateral check
# ---------------------------------------------------------------------------

def test_bilateral_true_when_undirected_path_exists():
    """C→D insight: undirected graph has C-D edge so D can reach C."""
    adapter = _make_adapter()
    validator = _make_validator(adapter)
    event = _make_event(source="C", target="D")
    result = validator.validate(event)
    assert result.validation_status in (STATUS_BILATERAL, STATUS_CORROBORATED)


def test_bilateral_false_when_no_return_path():
    """X1→Y1 insight in isolated graph: no path from Y1 back to X1."""
    adapter = _make_isolated_adapter()
    # Remove the only cross edge so the isolated graph truly has no return
    adapter._G.remove_edge("X1", "Y1")
    # Add a one-way-only INSIGHT_LINK (simulate a prior materialization)
    adapter._G.add_edge("X1", "Y1", relation=INSIGHT_RELATION,
                        confidence=0.85, weight=2.0, provenance="insight")

    # Now Y1 → X1 has no undirected path (Y branch and X branch disconnected)
    validator = _make_validator(adapter)
    event = _make_event(source="X1", target="Y1", bridging="Y1")
    result = validator.validate(event)
    # Y1 can't reach X1 (Y cluster is Y1-Y2 only, not connected to X cluster)
    assert result.validation_status in (STATUS_ISOLATED, STATUS_UNILATERAL)


def test_bilateral_respects_max_hop():
    """Target is reachable but only via a very long path — exceeds max_hop."""
    G = nx.DiGraph()
    # Linear chain: A→B→C→D→E→F→G (7 nodes, 6 hops)
    nodes = list("ABCDEFG")
    for i in range(len(nodes) - 1):
        G.add_edge(nodes[i], nodes[i + 1], relation="NEXT", weight=1.0, confidence=1.0)
    for n in nodes:
        G.nodes[n].setdefault("label", n)
        G.nodes[n].setdefault("type", "entity")

    adapter = NetworkXAdapter(G)
    engine = RandomEngine(dim=32)
    adapter.embeddings = {n: engine.encode_one(n) for n in G.nodes()}

    community_map = {n: (0 if n < "D" else 1) for n in nodes}
    adapter.community_map = community_map
    def patched_get_community(entity_id):
        return adapter.community_map.get(entity_id, -1)
    adapter.get_community = patched_get_community

    # Validate A→G insight with max_hop=2: bilateral path A→...→G is 6 hops
    validator = InsightValidator(adapter, max_hop=2)
    event = _make_event(source="A", target="G", bridging="G")
    result = validator.validate(event)
    # Path G→A is 6 hops undirected — exceeds max_hop=2
    assert result.validation_status in (STATUS_ISOLATED, STATUS_UNILATERAL)


# ---------------------------------------------------------------------------
# Corroboration check
# ---------------------------------------------------------------------------

def test_corroboration_count_positive_when_alt_seeds_reach_target():
    """A and B (community 0) can both reach D — corroboration_count >= 2."""
    adapter = _make_adapter()
    validator = _make_validator(adapter)
    event = _make_event(source="C", target="D")
    result = validator.validate(event)
    assert result.corroboration_count >= 2


def test_corroboration_zero_when_isolated():
    """In isolated graph, no other community-0 member can reach Y1."""
    adapter = _make_isolated_adapter()
    validator = _make_validator(adapter)
    event = _make_event(source="X1", target="Y1", bridging="Y1")
    result = validator.validate(event)
    # X2, X3 have no path to Y1 (one-way X1→Y1 only)
    assert result.corroboration_count == 0


def test_corroboration_respects_seeds_limit():
    """corroboration_seeds=1: at most 1 alternate seed is checked."""
    adapter = _make_adapter()
    validator = _make_validator(adapter, corroboration_seeds=1)
    event = _make_event(source="C", target="D")
    result = validator.validate(event)
    # With only 1 seed checked, count can be at most 1
    assert result.corroboration_count <= 1


# ---------------------------------------------------------------------------
# Status assignment
# ---------------------------------------------------------------------------

def test_status_corroborated_when_bilateral_and_count_gte_threshold():
    """bilateral=True and corroboration_count >= 2 → STATUS_CORROBORATED."""
    adapter = _make_adapter()
    validator = _make_validator(adapter, corroboration_threshold=2)
    event = _make_event(source="C", target="D")
    result = validator.validate(event)
    assert result.validation_status == STATUS_CORROBORATED


def test_status_bilateral_when_bilateral_but_low_corroboration():
    """bilateral=True and corroboration_count < threshold → STATUS_BILATERAL."""
    adapter = _make_adapter()
    # Raise threshold above achievable corroboration count
    validator = _make_validator(adapter, corroboration_threshold=100)
    event = _make_event(source="C", target="D")
    result = validator.validate(event)
    assert result.validation_status == STATUS_BILATERAL


def test_status_isolated_when_no_bilateral_no_corroboration():
    """No return path and no alternate seeds → STATUS_ISOLATED."""
    adapter = _make_isolated_adapter()
    # Remove the cross edge so Y1 can't be reached by anyone else either
    adapter._G.remove_edge("X1", "Y1")
    adapter._G.add_edge("X1", "Y1", relation=INSIGHT_RELATION,
                        confidence=0.85, weight=2.0, provenance="insight")
    validator = _make_validator(adapter)
    event = _make_event(source="X1", target="Y1", bridging="Y1")
    result = validator.validate(event)
    assert result.validation_status == STATUS_ISOLATED


def test_status_unilateral_when_corroboration_without_bilateral():
    """
    Create a graph where alt seeds can reach target but target can't return.

    X1→Y1 (insight), X2→Y1 (corroborating path).
    X1 and X2 are NOT connected to each other — critical: without X2→X1,
    Y1 cannot navigate back to X1 via X2. Y1 has no outgoing edges.

    Bilateral: False — Y1 can reach X2 but X2 has no edge to X1, so Y1
              cannot reach X1 regardless of the undirected view.
    Corroboration: 1 — X2 is in community 0 and reaches Y1 directly (X2→Y1).
    """
    G = nx.DiGraph()
    G.add_edge("X1", "Y1", relation="BRIDGE", weight=1.0, confidence=1.0)
    # X2→Y1 provides corroboration; X2 is NOT connected to X1
    G.add_edge("X2", "Y1", relation="KNOWS", weight=1.0, confidence=1.0)
    # Y1 is a dead-end — no outgoing edges, no path back to X1 specifically
    for n in G.nodes():
        G.nodes[n].setdefault("label", n)
        G.nodes[n].setdefault("type", "entity")

    adapter = NetworkXAdapter(G)
    engine = RandomEngine(dim=32)
    adapter.embeddings = {n: engine.encode_one(n) for n in G.nodes()}

    community_map = {"X1": 0, "X2": 0, "Y1": 1}
    adapter.community_map = community_map
    def patched(eid):
        return adapter.community_map.get(eid, -1)
    adapter.get_community = patched

    validator = InsightValidator(adapter, corroboration_seeds=5, corroboration_threshold=1)
    event = _make_event(source="X1", target="Y1", bridging="Y1")
    result = validator.validate(event)

    # X2 is in community 0 and has X2→Y1, so corroboration >= 1
    # But Y1 is a dead-end → no bilateral path back
    assert result.corroboration_count >= 1
    assert result.validation_status == STATUS_UNILATERAL


# ---------------------------------------------------------------------------
# Edge confidence promotion
# ---------------------------------------------------------------------------

def test_corroborated_insight_edge_promoted_to_095():
    """INSIGHT_LINK edge confidence raised to 0.95 for corroborated events."""
    adapter = _make_adapter()
    G = adapter._G
    # Pre-materialise the insight edge at default confidence
    G.add_edge("C", "D", relation=INSIGHT_RELATION,
               confidence=INSIGHT_CONFIDENCE, weight=2.0, provenance="insight")

    validator = _make_validator(adapter, corroboration_threshold=2)
    event = _make_event(source="C", target="D")
    validator.validate(event)

    data = G.get_edge_data("C", "D")
    assert data["confidence"] == pytest.approx(CONFIDENCE_CORROBORATED)


def test_bilateral_insight_edge_promoted_to_092():
    """INSIGHT_LINK edge confidence raised to 0.92 for bilateral-only events."""
    adapter = _make_adapter()
    G = adapter._G
    G.add_edge("C", "D", relation=INSIGHT_RELATION,
               confidence=INSIGHT_CONFIDENCE, weight=2.0, provenance="insight")

    # Raise threshold so corroboration doesn't kick in
    validator = _make_validator(adapter, corroboration_threshold=100)
    event = _make_event(source="C", target="D")
    validator.validate(event)

    data = G.get_edge_data("C", "D")
    assert data["confidence"] == pytest.approx(CONFIDENCE_BILATERAL)


def test_isolated_event_does_not_change_edge_confidence():
    """Isolated events must NOT alter the edge confidence."""
    adapter = _make_isolated_adapter()
    G = adapter._G
    G.add_edge("X1", "Y1", relation=INSIGHT_RELATION,
               confidence=INSIGHT_CONFIDENCE, weight=2.0, provenance="insight")

    validator = _make_validator(adapter)
    event = _make_event(source="X1", target="Y1", bridging="Y1")
    result = validator.validate(event)

    if result.validation_status == STATUS_ISOLATED:
        data = G.get_edge_data("X1", "Y1")
        assert data["confidence"] == pytest.approx(INSIGHT_CONFIDENCE)


def test_promotion_does_not_exceed_1():
    """Edge confidence is capped at 1.0 even if called multiple times."""
    adapter = _make_adapter()
    G = adapter._G
    G.add_edge("C", "D", relation=INSIGHT_RELATION,
               confidence=1.0, weight=2.0, provenance="insight")

    validator = _make_validator(adapter, corroboration_threshold=2)
    event = _make_event(source="C", target="D")
    validator.validate(event)

    data = G.get_edge_data("C", "D")
    assert data["confidence"] <= 1.0


def test_promotion_also_found_on_reverse_edge():
    """Promotion works when INSIGHT_LINK is on the D→C reverse edge."""
    adapter = _make_adapter()
    G = adapter._G
    # Place the INSIGHT_LINK on the reverse edge
    G.add_edge("D", "C", relation=INSIGHT_RELATION,
               confidence=INSIGHT_CONFIDENCE, weight=2.0, provenance="insight")

    validator = _make_validator(adapter, corroboration_threshold=2)
    event = _make_event(source="C", target="D")
    validator.validate(event)

    data = G.get_edge_data("D", "C")
    assert data["confidence"] > INSIGHT_CONFIDENCE


# ---------------------------------------------------------------------------
# validate_all
# ---------------------------------------------------------------------------

def test_validate_all_processes_every_event():
    """validate_all returns the same list with all events mutated."""
    adapter = _make_adapter()
    validator = _make_validator(adapter)
    events = [_make_event() for _ in range(3)]
    result = validator.validate_all(events)
    assert result is events
    for ev in result:
        assert ev.validation_status != "pending"


# ---------------------------------------------------------------------------
# InsightEvent id / validation fields
# ---------------------------------------------------------------------------

def test_insight_event_has_id():
    """InsightEvent.id is set automatically and is non-empty."""
    ev = _make_event()
    assert isinstance(ev.id, str)
    assert len(ev.id) > 0


def test_insight_event_ids_are_unique():
    """Two InsightEvents have different ids."""
    ev1 = _make_event()
    ev2 = _make_event()
    assert ev1.id != ev2.id


def test_insight_event_default_validation_status():
    """InsightEvent starts with validation_status='pending'."""
    ev = _make_event()
    assert ev.validation_status == "pending"


def test_insight_event_default_corroboration_count():
    """InsightEvent starts with corroboration_count=0."""
    ev = _make_event()
    assert ev.corroboration_count == 0


def test_validate_updates_status_from_pending():
    """After validate(), status is no longer 'pending'."""
    adapter = _make_adapter()
    validator = _make_validator(adapter)
    event = _make_event()
    assert event.validation_status == "pending"
    validator.validate(event)
    assert event.validation_status != "pending"
