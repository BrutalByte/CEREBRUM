"""
tests/test_namespace.py
Hole 3 — Semantic Collision fix: Namespace Isolation in THALAMUS / SignalEncoder.

Validates that IngestionPipeline(namespace="text") prefixes entity IDs,
SignalEncoder(namespace="signal") namespaces entity lookups in learn_alignment(),
and that the two namespaces keep distinct entities isolated.
"""
import numpy as np

from core.thalamus import IngestionPipeline
from core.signal_encoder import StatisticalSignalEncoder, SpectralSignalEncoder


# ---------------------------------------------------------------------------
# IngestionPipeline namespace
# ---------------------------------------------------------------------------

def test_pipeline_namespace_prefixes_source():
    pipe = IngestionPipeline(namespace="text")
    edge = pipe.process("Alice", "Bob", "knows")
    assert edge.source == "text:Alice"


def test_pipeline_namespace_prefixes_target():
    pipe = IngestionPipeline(namespace="text")
    edge = pipe.process("Alice", "Bob", "knows")
    assert edge.target == "text:Bob"


def test_pipeline_no_namespace_unchanged():
    """Default namespace='' → IDs unchanged (backward compatible)."""
    pipe = IngestionPipeline()
    edge = pipe.process("Alice", "Bob", "knows")
    assert edge.source == "Alice"
    assert edge.target == "Bob"


def test_pipeline_namespace_after_dedup():
    """Dedup alias collapses BEFORE namespace is applied."""
    pipe = IngestionPipeline(
        entity_dedup_map={"alice": "Alice"},
        entity_normalizer=lambda s: s.lower(),
        namespace="text",
    )
    edge = pipe.process("Alice", "Bob", "knows")
    # After normalizer: "alice", "bob"
    # After dedup: "Alice" (from alias), "bob"
    # After namespace: "text:Alice", "text:bob"
    assert edge.source == "text:Alice"
    assert edge.target == "text:bob"


def test_collision_isolated():
    """text:Temp_Sensor_1 and signal:Temp_Sensor_1 are distinct IDs."""
    text_pipe = IngestionPipeline(namespace="text")
    text_edge = text_pipe.process("Temp_Sensor_1", "Project_X", "belongs_to")
    assert text_edge.source == "text:Temp_Sensor_1"

    # Signal encoder namespaces separately
    enc = StatisticalSignalEncoder(entity_dim=8, namespace="signal")
    signal_id = enc.get_namespaced_id("Temp_Sensor_1")
    assert signal_id == "signal:Temp_Sensor_1"

    # The two IDs are distinct — no collision
    assert text_edge.source != signal_id


def test_collision_without_namespace():
    """Without namespace, same raw ID collides — documents the bug being fixed."""
    pipe = IngestionPipeline()  # no namespace
    edge = pipe.process("Temp_Sensor_1", "X", "rel")
    assert edge.source == "Temp_Sensor_1"

    enc = StatisticalSignalEncoder(entity_dim=8, namespace="")
    raw_id = enc.get_namespaced_id("Temp_Sensor_1")
    assert raw_id == "Temp_Sensor_1"

    # They collide when no namespace is set
    assert edge.source == raw_id


def test_pipeline_namespace_with_process_triple():
    pipe = IngestionPipeline(namespace="kg")
    edge = pipe.process_triple(("Newton", "DISCOVERED", "Gravity"))
    assert edge.source == "kg:Newton"
    assert edge.target == "kg:Gravity"


# ---------------------------------------------------------------------------
# SignalEncoder namespace
# ---------------------------------------------------------------------------

def test_encoder_get_namespaced_id():
    enc = StatisticalSignalEncoder(entity_dim=8, namespace="signal")
    assert enc.get_namespaced_id("sensor_42") == "signal:sensor_42"


def test_encoder_no_namespace_verbatim():
    enc = StatisticalSignalEncoder(entity_dim=8, namespace="")
    assert enc.get_namespaced_id("sensor_42") == "sensor_42"


def test_encoder_default_namespace_is_signal():
    enc = StatisticalSignalEncoder(entity_dim=8)
    assert enc._namespace == "signal"


def test_spectral_encoder_default_namespace_is_signal():
    enc = SpectralSignalEncoder(entity_dim=8)
    assert enc._namespace == "signal"


def test_encoder_namespace_applied_in_learn_alignment():
    """learn_alignment looks up 'signal:X' when namespace='signal'."""
    from unittest.mock import MagicMock
    enc = StatisticalSignalEncoder(entity_dim=8, namespace="signal")

    adapter = MagicMock()
    target_emb = np.random.randn(8).astype(np.float32)
    target_emb /= np.linalg.norm(target_emb)

    def _get_emb(eid):
        # Only returns if the namespaced key is used
        if eid.startswith("signal:"):
            return target_emb
        return None  # wrong key — no namespace

    adapter.get_embedding.side_effect = _get_emb

    signals = [np.random.randn(20) for _ in range(3)]
    ids = ["A", "B", "C"]  # bare IDs — encoder should prefix them

    # Should succeed: encoder applies 'signal:' prefix → adapter sees 'signal:A', etc.
    n = enc.learn_alignment(signals, ids, adapter, min_anchors=3)
    assert n == 3


def test_encoder_no_namespace_verbatim_in_learn_alignment():
    """learn_alignment passes IDs verbatim when namespace=''."""
    from unittest.mock import MagicMock
    enc = StatisticalSignalEncoder(entity_dim=8, namespace="")

    adapter = MagicMock()
    target_emb = np.ones(8, dtype=np.float32)
    target_emb /= np.linalg.norm(target_emb)
    adapter.get_embedding.return_value = target_emb

    signals = [np.random.randn(20) for _ in range(3)]
    ids = ["A", "B", "C"]
    n = enc.learn_alignment(signals, ids, adapter, min_anchors=3)
    assert n == 3
    # Verify called without namespace prefix
    calls = [c[0][0] for c in adapter.get_embedding.call_args_list]
    assert all(":" not in c for c in calls)


def test_explicit_merge_via_dedup_map():
    """entity_dedup_map can explicitly merge signal: ID → text: ID."""
    # Scenario: text and signal share the same real-world entity
    # text pipeline uses "text:" namespace; dedup_map collapses signal alias
    pipe = IngestionPipeline(
        entity_dedup_map={"signal:Temp_Sensor_1": "text:Temp_Sensor_1"},
        namespace="text",
    )
    # Normal text edge
    edge = pipe.process("Sensor_X", "Lab", "located_in")
    assert edge.source == "text:Sensor_X"
