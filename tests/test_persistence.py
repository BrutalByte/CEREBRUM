"""
Tests for the CEREBRUM persistence layer.

Covers:
  - Path traversal security (_resolve_safe_path)
  - save_state / load_state round-trip
  - is_state_cached

Uses tmp_path and monkeypatching of PARALLAX_DATA_DIR to keep all I/O
within pytest's managed temporary directory.
"""
import numpy as np
import pytest

import core.persistence as persistence_mod
from adapters.networkx_adapter import NetworkXAdapter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def sandbox(tmp_path, monkeypatch):
    """
    Point SAFE_DATA_DIR at a fresh temp directory for every test.
    monkeypatch.setenv alone is insufficient because SAFE_DATA_DIR is a
    module-level constant evaluated at import time; we patch it directly.
    """
    monkeypatch.setattr(persistence_mod, "SAFE_DATA_DIR", tmp_path.resolve())
    return tmp_path


def _make_simple_state():
    """Minimal valid state dict for save_state."""
    import networkx as nx
    G  = nx.Graph()
    G.add_edge("A", "B", relation="KNOWS")
    adapter = NetworkXAdapter(G)
    community_map = {"A": 0, "B": 0}
    embeddings = {"A": np.zeros(8, dtype=np.float32), "B": np.ones(8, dtype=np.float32)}
    csa_metadata = {"distances": {}, "adjacent_pairs": set()}
    return adapter, community_map, embeddings, csa_metadata


# ---------------------------------------------------------------------------
# Path traversal security
# ---------------------------------------------------------------------------

def test_resolve_safe_path_relative_ok(sandbox):
    path = persistence_mod._resolve_safe_path("state.pkl")
    assert str(path).startswith(str(sandbox))


def test_resolve_safe_path_subdirectory_ok(sandbox):
    path = persistence_mod._resolve_safe_path("subdir/state.pkl")
    assert str(path).startswith(str(sandbox))


def test_resolve_safe_path_traversal_blocked(sandbox):
    with pytest.raises(PermissionError, match="Path traversal"):
        persistence_mod._resolve_safe_path("../../etc/passwd")


def test_resolve_safe_path_absolute_outside_blocked(sandbox):
    # An absolute path outside the sandbox must be blocked
    outside = "/tmp/malicious_state.pkl"
    # On Windows the outside path may not exist but the security check
    # still fires before any I/O attempt.
    with pytest.raises(PermissionError, match="Path traversal"):
        persistence_mod._resolve_safe_path(outside)


def test_resolve_safe_path_absolute_inside_ok(sandbox):
    inside = str(sandbox / "state.pkl")
    path = persistence_mod._resolve_safe_path(inside)
    assert str(path).startswith(str(sandbox))


# ---------------------------------------------------------------------------
# save_state / load_state round-trip
# ---------------------------------------------------------------------------

def test_save_creates_file(sandbox):
    adapter, cmap, emb, csa = _make_simple_state()
    persistence_mod.save_state("state.pkl", adapter, cmap, emb, csa)
    assert (sandbox / "state.pkl").exists()


def test_save_load_round_trip(sandbox):
    adapter, cmap, emb, csa = _make_simple_state()
    persistence_mod.save_state("state.pkl", adapter, cmap, emb, csa)
    loaded = persistence_mod.load_state("state.pkl")
    assert loaded["community_map"] == cmap


def test_load_preserves_embeddings(sandbox):
    adapter, cmap, emb, csa = _make_simple_state()
    persistence_mod.save_state("state.pkl", adapter, cmap, emb, csa)
    loaded = persistence_mod.load_state("state.pkl")
    np.testing.assert_array_equal(loaded["embeddings"]["A"], emb["A"])


def test_load_preserves_version_field(sandbox):
    adapter, cmap, emb, csa = _make_simple_state()
    persistence_mod.save_state("state.pkl", adapter, cmap, emb, csa)
    loaded = persistence_mod.load_state("state.pkl")
    assert "version" in loaded


def test_load_preserves_timestamp(sandbox):
    adapter, cmap, emb, csa = _make_simple_state()
    persistence_mod.save_state("state.pkl", adapter, cmap, emb, csa)
    loaded = persistence_mod.load_state("state.pkl")
    assert "timestamp" in loaded
    assert loaded["timestamp"] > 0


def test_load_preserves_optional_fields(sandbox):
    adapter, cmap, emb, csa = _make_simple_state()
    weights = {"KNOWS": 1.5}
    persistence_mod.save_state("state.pkl", adapter, cmap, emb, csa,
                                default_edge_type_weights=weights)
    loaded = persistence_mod.load_state("state.pkl")
    assert loaded["default_edge_type_weights"] == weights


def test_save_creates_parent_dirs(sandbox):
    adapter, cmap, emb, csa = _make_simple_state()
    persistence_mod.save_state("deep/nested/state.pkl", adapter, cmap, emb, csa)
    assert (sandbox / "deep" / "nested" / "state.pkl").exists()


def test_load_missing_file_raises(sandbox):
    with pytest.raises(FileNotFoundError):
        persistence_mod.load_state("does_not_exist.pkl")


# ---------------------------------------------------------------------------
# is_state_cached
# ---------------------------------------------------------------------------

def test_is_state_cached_false_before_save(sandbox):
    assert persistence_mod.is_state_cached("state.pkl") is False


def test_is_state_cached_true_after_save(sandbox):
    adapter, cmap, emb, csa = _make_simple_state()
    persistence_mod.save_state("state.pkl", adapter, cmap, emb, csa)
    assert persistence_mod.is_state_cached("state.pkl") is True


def test_is_state_cached_traversal_returns_false(sandbox):
    # Path traversal attempt → PermissionError caught → returns False
    assert persistence_mod.is_state_cached("../../etc/passwd") is False
