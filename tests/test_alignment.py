"""
Tests for AlignmentIndex and auto_align_by_label.

AlignmentIndex maps (adapter_name, entity_id) pairs to a single canonical ID,
enabling federated queries across heterogeneous graph sources that use different
ID schemes for the same real-world entities.
"""

from adapters.networkx_adapter import NetworkXAdapter
from core.alignment_engine import AlignmentIndex, auto_align_by_label


# ---------------------------------------------------------------------------
# AlignmentIndex — basic operations
# ---------------------------------------------------------------------------

def test_get_canonical_before_alignment():
    idx = AlignmentIndex()
    assert idx.get_canonical("graphA", "node_1") is None


def test_add_alignment_both_entities_resolve():
    idx = AlignmentIndex()
    idx.add_alignment("graphA", "e1", "graphB", "b1")
    assert idx.get_canonical("graphA", "e1") is not None
    assert idx.get_canonical("graphB", "b1") is not None


def test_add_alignment_same_canonical():
    idx = AlignmentIndex()
    idx.add_alignment("graphA", "e1", "graphB", "b1")
    can_a = idx.get_canonical("graphA", "e1")
    can_b = idx.get_canonical("graphB", "b1")
    assert can_a == can_b


def test_add_alignment_default_canonical_format():
    idx = AlignmentIndex()
    idx.add_alignment("graphA", "e1", "graphB", "b1")
    can = idx.get_canonical("graphA", "e1")
    # Default canonical ID is "adapterA:idA"
    assert can == "graphA:e1"


def test_add_alignment_explicit_canonical():
    idx = AlignmentIndex()
    idx.add_alignment("graphA", "e1", "graphB", "b1", canonical_id="global:entity_X")
    assert idx.get_canonical("graphA", "e1") == "global:entity_X"
    assert idx.get_canonical("graphB", "b1") == "global:entity_X"


def test_get_aliases_returns_both_sides():
    idx = AlignmentIndex()
    idx.add_alignment("graphA", "e1", "graphB", "b1")
    can = idx.get_canonical("graphA", "e1")
    aliases = idx.get_aliases(can)
    assert ("graphA", "e1") in aliases
    assert ("graphB", "b1") in aliases


def test_get_aliases_unknown_canonical():
    idx = AlignmentIndex()
    aliases = idx.get_aliases("nonexistent:canonical")
    assert aliases == set()


def test_resolve_aliases_known_entity():
    idx = AlignmentIndex()
    idx.add_alignment("graphA", "e1", "graphB", "b1")
    aliases = idx.resolve_aliases("graphA", "e1")
    assert ("graphA", "e1") in aliases
    assert ("graphB", "b1") in aliases


def test_resolve_aliases_unknown_entity():
    idx = AlignmentIndex()
    aliases = idx.resolve_aliases("graphA", "unknown_node")
    # Unknown entity returns itself as a fallback
    assert ("graphA", "unknown_node") in aliases


def test_add_alignment_three_adapters():
    idx = AlignmentIndex()
    idx.add_alignment("graphA", "e1", "graphB", "b1")
    idx.add_alignment("graphA", "e1", "graphC", "c1")
    can = idx.get_canonical("graphA", "e1")
    aliases = idx.get_aliases(can)
    assert ("graphA", "e1") in aliases
    assert ("graphB", "b1") in aliases
    assert ("graphC", "c1") in aliases


def test_add_alignment_preserves_existing_canonical():
    """If one side already has a canonical ID, reuse it."""
    idx = AlignmentIndex()
    idx.add_alignment("graphA", "e1", "graphB", "b1", canonical_id="canon:X")
    # Now align graphB:b1 with graphC:c1 — should reuse canon:X
    idx.add_alignment("graphB", "b1", "graphC", "c1")
    # graphC:c1 should share the same canonical as graphB:b1
    assert idx.get_canonical("graphC", "c1") == idx.get_canonical("graphB", "b1")


def test_multiple_independent_alignments():
    idx = AlignmentIndex()
    idx.add_alignment("graphA", "einstein", "graphB", "albert_e")
    idx.add_alignment("graphA", "newton",   "graphB", "isaac_n")
    can_e = idx.get_canonical("graphA", "einstein")
    can_n = idx.get_canonical("graphA", "newton")
    assert can_e != can_n


# ---------------------------------------------------------------------------
# auto_align_by_label
# ---------------------------------------------------------------------------

def test_auto_align_identical_labels():
    G1 = NetworkXAdapter.from_triples([("newton", "INFLUENCED", "faraday")])
    G2 = NetworkXAdapter.from_triples([("newton", "COLLABORATED", "leibniz")])
    adapters = {"g1": G1, "g2": G2}
    idx = auto_align_by_label(adapters)
    # "newton" appears in both — should be aligned
    can_g1 = idx.get_canonical("g1", "newton")
    can_g2 = idx.get_canonical("g2", "newton")
    assert can_g1 is not None
    assert can_g1 == can_g2


def test_auto_align_no_common_labels():
    G1 = NetworkXAdapter.from_triples([("A", "R", "B")])
    G2 = NetworkXAdapter.from_triples([("X", "R", "Y")])
    adapters = {"g1": G1, "g2": G2}
    idx = auto_align_by_label(adapters)
    # No shared labels — nothing aligned
    assert idx.get_canonical("g1", "A") is None
    assert idx.get_canonical("g2", "X") is None


def test_auto_align_returns_alignment_index():
    G1 = NetworkXAdapter.from_triples([("A", "R", "B")])
    idx = auto_align_by_label({"g1": G1})
    assert isinstance(idx, AlignmentIndex)
