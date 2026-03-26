"""
Tests for core/thalamus.py — IngestionPipeline and adapter integration.

Covers:
  - IngestionPipeline in isolation (entity norm, dedup, relation norm,
    confidence, provenance, weight, properties passthrough)
  - Integration with load_csv_adapter
  - Integration with NetworkXAdapter.from_triples
  - Integration with StreamAdapter
  - Backward-compatibility (no pipeline = no change)
"""
import pytest

from core.thalamus import IngestionPipeline, ProcessedEdge


# ===========================================================================
# IngestionPipeline — unit tests
# ===========================================================================

class TestDefaultPipeline:
    """No-arg pipeline applies safe defaults only."""

    def test_strips_entity_whitespace(self):
        p = IngestionPipeline()
        e = p.process("  newton  ", "  einstein  ", "INFLUENCED")
        assert e.source == "newton"
        assert e.target == "einstein"

    def test_uppercases_relation(self):
        p = IngestionPipeline()
        e = p.process("a", "b", "influenced")
        assert e.relation == "INFLUENCED"

    def test_strips_relation_whitespace(self):
        p = IngestionPipeline()
        e = p.process("a", "b", "  KNOWS  ")
        assert e.relation == "KNOWS"

    def test_empty_relation_becomes_related_to(self):
        p = IngestionPipeline()
        e = p.process("a", "b", "")
        assert e.relation == "RELATED_TO"

    def test_default_confidence_is_one(self):
        p = IngestionPipeline()
        e = p.process("a", "b", "R")
        assert e.confidence == 1.0

    def test_default_provenance_is_empty(self):
        p = IngestionPipeline()
        e = p.process("a", "b", "R")
        assert e.provenance == ""

    def test_default_weight_is_one(self):
        p = IngestionPipeline()
        e = p.process("a", "b", "R")
        assert e.weight == 1.0

    def test_returns_processed_edge(self):
        p = IngestionPipeline()
        e = p.process("a", "b", "R")
        assert isinstance(e, ProcessedEdge)


class TestEntityNormalization:
    def test_custom_normalizer_lowercases(self):
        p = IngestionPipeline(entity_normalizer=lambda s: s.lower().strip())
        e = p.process("Newton", "Einstein", "R")
        assert e.source == "newton"
        assert e.target == "einstein"

    def test_custom_normalizer_replaces_spaces(self):
        p = IngestionPipeline(
            entity_normalizer=lambda s: s.lower().strip().replace(" ", "_")
        )
        e = p.process("Tom Hanks", "Forrest Gump", "R")
        assert e.source == "tom_hanks"
        assert e.target == "forrest_gump"


class TestEntityDedup:
    def test_dedup_map_aliases_source(self):
        p = IngestionPipeline(
            entity_normalizer=lambda s: s.lower().strip(),
            entity_dedup_map={"tom hanks": "tom_hanks", "tomhanks": "tom_hanks"},
        )
        e = p.process("Tom Hanks", "forrest_gump", "R")
        assert e.source == "tom_hanks"

    def test_dedup_map_aliases_target(self):
        p = IngestionPipeline(
            entity_normalizer=lambda s: s.lower().strip(),
            entity_dedup_map={"t. hanks": "tom_hanks"},
        )
        e = p.process("director", "T. Hanks", "DIRECTED")
        assert e.target == "tom_hanks"

    def test_dedup_map_preserves_unmatched(self):
        p = IngestionPipeline(
            entity_dedup_map={"alias": "canonical"},
        )
        e = p.process("other_node", "yet_another", "R")
        assert e.source == "other_node"
        assert e.target == "yet_another"

    def test_has_dedup_property(self):
        p_with = IngestionPipeline(entity_dedup_map={"a": "b"})
        p_without = IngestionPipeline()
        assert p_with.has_dedup is True
        assert p_without.has_dedup is False

    def test_dedup_size(self):
        p = IngestionPipeline(entity_dedup_map={"a": "c", "b": "c"})
        assert p.dedup_size() == 2


class TestRelationNormalization:
    def test_dict_map_exact_match(self):
        p = IngestionPipeline(relation_map={"activates": "ACTIVATES"})
        e = p.process("a", "b", "activates")
        assert e.relation == "ACTIVATES"

    def test_dict_map_case_insensitive(self):
        """Keys in the dict are matched case-insensitively."""
        p = IngestionPipeline(relation_map={"activates": "ACTIVATES"})
        e = p.process("a", "b", "ACTIVATES")
        assert e.relation == "ACTIVATES"

    def test_dict_map_unmapped_falls_through_to_uppercase(self):
        p = IngestionPipeline(relation_map={"activates": "ACTIVATES"})
        e = p.process("a", "b", "inhibits")
        # Not in the map → default uppercase
        assert e.relation == "INHIBITS"

    def test_callable_map_replaces_normalizer(self):
        p = IngestionPipeline(
            relation_map=lambda r: r.strip().upper().replace(" ", "_")
        )
        e = p.process("a", "b", "is related to")
        assert e.relation == "IS_RELATED_TO"

    def test_dict_map_multiple_variants(self):
        rmap = {
            "activates":            "ACTIVATES",
            "positively_regulates": "ACTIVATES",
            "upregulates":          "ACTIVATES",
        }
        p = IngestionPipeline(relation_map=rmap)
        assert p.process("a", "b", "positively_regulates").relation == "ACTIVATES"
        assert p.process("a", "b", "upregulates").relation == "ACTIVATES"
        assert p.process("a", "b", "activates").relation == "ACTIVATES"


class TestConfidence:
    def test_confidence_fn_result_stored(self):
        p = IngestionPipeline(confidence_fn=lambda s, t, r, m: 0.75)
        e = p.process("a", "b", "R")
        assert e.confidence == 0.75

    def test_confidence_fn_receives_meta(self):
        seen = {}
        def _fn(src, tgt, rel, meta):
            seen.update(meta)
            return float(meta.get("score", 1.0))

        p = IngestionPipeline(confidence_fn=_fn)
        p.process("a", "b", "R", {"score": "0.4", "db": "pubmed"})
        assert seen.get("score") == "0.4"
        assert seen.get("db") == "pubmed"

    def test_confidence_clamped_above_one(self):
        p = IngestionPipeline(confidence_fn=lambda *_: 1.5)
        e = p.process("a", "b", "R")
        assert e.confidence == 1.0

    def test_confidence_clamped_below_zero(self):
        p = IngestionPipeline(confidence_fn=lambda *_: -0.3)
        e = p.process("a", "b", "R")
        assert e.confidence == 0.0

    def test_confidence_from_metadata_fallback(self):
        """No confidence_fn — should read from metadata dict."""
        p = IngestionPipeline()
        e = p.process("a", "b", "R", {"confidence": 0.6})
        assert e.confidence == pytest.approx(0.6)

    def test_has_confidence_property(self):
        p_with = IngestionPipeline(confidence_fn=lambda *_: 0.5)
        p_without = IngestionPipeline()
        assert p_with.has_confidence is True
        assert p_without.has_confidence is False


class TestProvenance:
    def test_provenance_fn_result_stored(self):
        p = IngestionPipeline(provenance_fn=lambda s, t, r, m: "pubmed:123")
        e = p.process("a", "b", "R")
        assert e.provenance == "pubmed:123"

    def test_provenance_fn_uses_meta(self):
        p = IngestionPipeline(
            provenance_fn=lambda s, t, r, m: m.get("source_db", "")
        )
        e = p.process("a", "b", "R", {"source_db": "wikidata"})
        assert e.provenance == "wikidata"

    def test_provenance_from_metadata_fallback(self):
        p = IngestionPipeline()
        e = p.process("a", "b", "R", {"provenance": "internal"})
        assert e.provenance == "internal"

    def test_has_provenance_property(self):
        p_with = IngestionPipeline(provenance_fn=lambda *_: "x")
        p_without = IngestionPipeline()
        assert p_with.has_provenance is True
        assert p_without.has_provenance is False


class TestMetadataPassthrough:
    def test_weight_passed_through(self):
        p = IngestionPipeline()
        e = p.process("a", "b", "R", {"weight": 0.3})
        assert e.weight == pytest.approx(0.3)

    def test_extra_columns_in_properties(self):
        p = IngestionPipeline()
        e = p.process("a", "b", "R", {"score": "0.9", "year": "2020"})
        assert e.properties["score"] == "0.9"
        assert e.properties["year"] == "2020"

    def test_promoted_fields_not_in_properties(self):
        """confidence/provenance/weight must not appear twice."""
        p = IngestionPipeline()
        e = p.process("a", "b", "R", {
            "confidence": 0.5, "provenance": "x", "weight": 0.8
        })
        assert "confidence" not in e.properties
        assert "provenance" not in e.properties
        assert "weight" not in e.properties


class TestProcessTriple:
    def test_process_triple_convenience(self):
        p = IngestionPipeline(
            entity_normalizer=lambda s: s.lower(),
            relation_map={"knows": "KNOWS"},
        )
        e = p.process_triple(("Alice", "knows", "Bob"))
        assert e.source == "alice"
        assert e.target == "bob"
        assert e.relation == "KNOWS"


# ===========================================================================
# CSV adapter integration
# ===========================================================================

class TestCSVAdapterPipeline:
    def test_relation_normalized_at_load(self, tmp_path):
        csv_file = tmp_path / "kg.csv"
        csv_file.write_text("source,target,relation\na,b,activates\nc,d,inhibits\n")

        from adapters.csv_adapter import load_csv_adapter
        pipeline = IngestionPipeline(
            relation_map={"activates": "ACTIVATES", "inhibits": "INHIBITS"}
        )
        adapter = load_csv_adapter(str(csv_file), pipeline=pipeline)
        G = adapter.to_networkx()

        assert G.get_edge_data("a", "b")["relation"] == "ACTIVATES"
        assert G.get_edge_data("c", "d")["relation"] == "INHIBITS"

    def test_entity_dedup_merges_nodes(self, tmp_path):
        csv_file = tmp_path / "kg.csv"
        csv_file.write_text(
            "source,target,relation\n"
            "tom hanks,forrest gump,PLAYS\n"
            "Tom Hanks,wilson,THROWS\n"
        )
        from adapters.csv_adapter import load_csv_adapter
        pipeline = IngestionPipeline(
            entity_normalizer=lambda s: s.lower().strip(),
            entity_dedup_map={"tom hanks": "tom_hanks"},
        )
        adapter = load_csv_adapter(str(csv_file), pipeline=pipeline)
        G = adapter.to_networkx()

        # Both rows should use the canonical "tom_hanks" node
        assert G.has_node("tom_hanks")
        assert not G.has_node("tom hanks")
        assert not G.has_node("Tom Hanks")
        # Two edges from tom_hanks
        assert len(list(G.edges("tom_hanks"))) == 2

    def test_confidence_stored_from_extra_column(self, tmp_path):
        csv_file = tmp_path / "kg.csv"
        csv_file.write_text("source,target,relation,score\na,b,R,0.7\n")

        from adapters.csv_adapter import load_csv_adapter
        pipeline = IngestionPipeline(
            confidence_fn=lambda s, t, r, m: float(m.get("score", 1.0))
        )
        adapter = load_csv_adapter(str(csv_file), pipeline=pipeline)
        G = adapter.to_networkx()

        assert G.get_edge_data("a", "b")["confidence"] == pytest.approx(0.7)

    def test_no_pipeline_is_backward_compatible(self, tmp_path):
        csv_file = tmp_path / "kg.csv"
        csv_file.write_text("source,target,relation\nnewton,einstein,INFLUENCED\n")

        from adapters.csv_adapter import load_csv_adapter
        adapter = load_csv_adapter(str(csv_file))
        G = adapter.to_networkx()

        assert G.has_edge("newton", "einstein")
        data = G.get_edge_data("newton", "einstein")
        assert data["relation"] == "INFLUENCED"


# ===========================================================================
# NetworkXAdapter.from_triples integration
# ===========================================================================

class TestFromTriplesPipeline:
    def test_relation_normalized(self):
        from adapters.networkx_adapter import NetworkXAdapter
        pipeline = IngestionPipeline(relation_map={"knows": "KNOWS"})
        adapter = NetworkXAdapter.from_triples(
            [("alice", "knows", "bob")], pipeline=pipeline
        )
        G = adapter.to_networkx()
        assert G.get_edge_data("alice", "bob")["relation"] == "KNOWS"

    def test_entity_dedup_via_triples(self):
        from adapters.networkx_adapter import NetworkXAdapter
        pipeline = IngestionPipeline(
            entity_normalizer=lambda s: s.lower().strip(),
            entity_dedup_map={"tom hanks": "tom_hanks"},
        )
        adapter = NetworkXAdapter.from_triples(
            [("Tom Hanks", "PLAYS", "forrest_gump")], pipeline=pipeline
        )
        G = adapter.to_networkx()
        assert G.has_node("tom_hanks")
        assert not G.has_node("Tom Hanks")

    def test_confidence_stored_from_metadata(self):
        from adapters.networkx_adapter import NetworkXAdapter
        pipeline = IngestionPipeline(
            confidence_fn=lambda s, t, r, m: float(m.get("conf", 1.0))
        )
        adapter = NetworkXAdapter.from_triples(
            [("a", "R", "b", {"conf": "0.55"})], pipeline=pipeline
        )
        G = adapter.to_networkx()
        assert G.get_edge_data("a", "b")["confidence"] == pytest.approx(0.55)

    def test_no_pipeline_is_backward_compatible(self):
        from adapters.networkx_adapter import NetworkXAdapter
        adapter = NetworkXAdapter.from_triples([("x", "LINKS", "y")])
        G = adapter.to_networkx()
        assert G.has_edge("x", "y")
        assert G.get_edge_data("x", "y")["relation"] == "LINKS"


# ===========================================================================
# StreamAdapter integration
# ===========================================================================

class TestStreamAdapterPipeline:
    def test_relation_normalized_on_ingest(self):
        from adapters.stream_adapter import StreamAdapter
        from core.stream_engine import StreamEvent

        pipeline = IngestionPipeline(relation_map={"activates": "ACTIVATES"})
        adapter = StreamAdapter(pipeline=pipeline)
        adapter.ingest(StreamEvent(source="gene_a", relation="activates", target="gene_b"))

        G = adapter.to_networkx()
        assert G.has_edge("gene_a", "gene_b")
        assert G.get_edge_data("gene_a", "gene_b")["relation"] == "ACTIVATES"

    def test_entity_dedup_on_ingest(self):
        from adapters.stream_adapter import StreamAdapter
        from core.stream_engine import StreamEvent

        pipeline = IngestionPipeline(
            entity_normalizer=lambda s: s.lower().strip(),
            entity_dedup_map={"gene a": "gene_a"},
        )
        adapter = StreamAdapter(pipeline=pipeline)
        adapter.ingest(StreamEvent(source="Gene A", relation="ACTIVATES", target="gene_b"))

        G = adapter.to_networkx()
        assert G.has_node("gene_a")
        assert not G.has_node("Gene A")

    def test_confidence_stored_on_ingest(self):
        from adapters.stream_adapter import StreamAdapter
        from core.stream_engine import StreamEvent

        pipeline = IngestionPipeline(
            confidence_fn=lambda s, t, r, m: float(m.get("score", 1.0))
        )
        adapter = StreamAdapter(pipeline=pipeline)
        adapter.ingest(StreamEvent(
            source="a", relation="R", target="b", metadata={"score": 0.42}
        ))

        G = adapter.to_networkx()
        assert G.get_edge_data("a", "b")["confidence"] == pytest.approx(0.42)

    def test_no_pipeline_is_backward_compatible(self):
        from adapters.stream_adapter import StreamAdapter
        from core.stream_engine import StreamEvent

        adapter = StreamAdapter()
        adapter.ingest(StreamEvent(source="x", relation="LINKS", target="y"))
        G = adapter.to_networkx()
        assert G.has_edge("x", "y")
        assert G.get_edge_data("x", "y")["relation"] == "LINKS"

    def test_ingest_batch_with_pipeline(self):
        from adapters.stream_adapter import StreamAdapter
        from core.stream_engine import StreamEvent

        pipeline = IngestionPipeline(relation_map={"inhibits": "INHIBITS"})
        adapter = StreamAdapter(pipeline=pipeline)
        adapter.ingest_batch([
            StreamEvent(source="a", relation="inhibits", target="b"),
            StreamEvent(source="c", relation="inhibits", target="d"),
        ])
        G = adapter.to_networkx()
        assert G.get_edge_data("a", "b")["relation"] == "INHIBITS"
        assert G.get_edge_data("c", "d")["relation"] == "INHIBITS"
