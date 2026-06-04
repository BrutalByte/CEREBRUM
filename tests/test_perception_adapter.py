"""
Tests for adapters/perception_adapter.py

Covers:
  - PerceptionResult dataclass
  - _extract_triples_heuristic
  - _extract_triples_llm
  - VisionBackend (mocked HTTP)
  - AudioBackend (mocked HTTP)
  - DocumentBackend (plaintext path)
  - PerceptionAdapter.ingest_text
  - PerceptionAdapter.ingest_image (mocked backend)
  - PerceptionAdapter.ingest_audio (mocked backend)
  - PerceptionAdapter.ingest_document (plaintext)
  - LLM fallback triggered on low-confidence result
  - No LLM fallback when confidence is high
  - PerceptionStreamSource (directory polling)
"""
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from adapters.perception_adapter import (
    AudioBackend,
    DocumentBackend,
    PerceptionAdapter,
    PerceptionResult,
    PerceptionStreamSource,
    VisionBackend,
    _extract_triples_heuristic,
    _extract_triples_llm,
)
from adapters.networkx_adapter import NetworkXAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_adapter() -> NetworkXAdapter:
    return NetworkXAdapter(nx.DiGraph())


def make_pa(
    llm_fn=None,
    confidence_threshold=0.7,
    vision=None,
    audio=None,
    document=None,
) -> PerceptionAdapter:
    return PerceptionAdapter(
        graph_adapter=make_adapter(),
        confidence_threshold=confidence_threshold,
        llm_fn=llm_fn,
        vision=vision,
        audio=audio,
        document=document,
    )


# ---------------------------------------------------------------------------
# PerceptionResult
# ---------------------------------------------------------------------------

class TestPerceptionResult:
    def test_defaults(self):
        r = PerceptionResult(raw_text="hello", confidence=0.9, provenance="test")
        assert r.structured is None
        assert r.metadata == {}

    def test_full(self):
        r = PerceptionResult(
            raw_text="x",
            confidence=0.5,
            provenance="vision:llava",
            structured={"labels": ["cat"]},
            metadata={"model": "llava"},
        )
        assert r.structured["labels"] == ["cat"]
        assert r.metadata["model"] == "llava"


# ---------------------------------------------------------------------------
# _extract_triples_heuristic
# ---------------------------------------------------------------------------

class TestExtractTriplesHeuristic:
    def test_single_triple(self):
        text = "Paris | LOCATED_IN | France"
        triples = _extract_triples_heuristic(text)
        assert triples == [("Paris", "LOCATED_IN", "France")]

    def test_multiple_triples(self):
        text = (
            "Eiffel Tower | LOCATED_IN | Paris\n"
            "Paris | CAPITAL_OF | France\n"
        )
        triples = _extract_triples_heuristic(text)
        assert len(triples) == 2
        assert ("Eiffel Tower", "LOCATED_IN", "Paris") in triples
        assert ("Paris", "CAPITAL_OF", "France") in triples

    def test_ignores_non_triple_lines(self):
        text = "Here are the triples:\nParis | LOCATED_IN | France\nSome other text"
        triples = _extract_triples_heuristic(text)
        assert len(triples) == 1

    def test_empty_input(self):
        assert _extract_triples_heuristic("") == []

    def test_relation_must_be_uppercase(self):
        # lowercase relation should not match
        triples = _extract_triples_heuristic("Paris | located_in | France")
        assert triples == []

    def test_strips_whitespace(self):
        text = "  Paris  |  LOCATED_IN  |  France  "
        triples = _extract_triples_heuristic(text)
        assert triples == [("Paris", "LOCATED_IN", "France")]


# ---------------------------------------------------------------------------
# _extract_triples_llm
# ---------------------------------------------------------------------------

class TestExtractTriplesLlm:
    def test_parses_llm_output(self):
        llm_fn = lambda _: "Paris | LOCATED_IN | France\nTower | PART_OF | Paris"
        triples = _extract_triples_llm("The Eiffel Tower is in Paris, France.", "", llm_fn)
        assert ("Paris", "LOCATED_IN", "France") in triples

    def test_llm_error_returns_empty(self):
        def bad_llm(_):
            raise RuntimeError("API error")
        triples = _extract_triples_llm("some text", "", bad_llm)
        assert triples == []

    def test_passes_context_in_prompt(self):
        received = {}
        def capture_llm(prompt):
            received["prompt"] = prompt
            return "A | IS_A | B"
        _extract_triples_llm("text", "office_context", capture_llm)
        assert "office_context" in received["prompt"]


# ---------------------------------------------------------------------------
# VisionBackend (mocked HTTP)
# ---------------------------------------------------------------------------

class TestVisionBackend:
    def test_returns_perception_result_on_success(self):
        backend = VisionBackend(endpoint="http://localhost:11434/v1", model="llava")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "A cat | SITS_ON | mat"}}]
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_resp):
            # Use a non-existent path to trigger data-URI branch (no file read)
            result = backend.perceive("/nonexistent/image.jpg")

        # Will fail at Path.exists() check but still return a result
        assert isinstance(result, PerceptionResult)
        assert result.provenance.startswith("vision:")

    def test_returns_low_confidence_on_error(self):
        backend = VisionBackend(endpoint="http://localhost:9999/v1")
        with patch("requests.post", side_effect=Exception("connection refused")):
            result = backend.perceive("/nonexistent/image.jpg")
        assert result.confidence == 0.0
        assert "error" in result.metadata

    def test_modality(self):
        assert VisionBackend().modality == "vision"


# ---------------------------------------------------------------------------
# AudioBackend (mocked HTTP)
# ---------------------------------------------------------------------------

class TestAudioBackend:
    def test_http_returns_result(self, tmp_path):
        audio_file = tmp_path / "clip.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 44)

        backend = AudioBackend(endpoint="http://localhost:9000/transcribe")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"text": "Hello world"}
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_resp):
            result = backend.perceive(str(audio_file))

        assert result.raw_text == "Hello world"
        assert result.confidence == 0.85
        assert result.provenance.startswith("audio:")

    def test_http_error_returns_low_confidence(self, tmp_path):
        audio_file = tmp_path / "clip.wav"
        audio_file.write_bytes(b"\x00" * 100)

        backend = AudioBackend(endpoint="http://localhost:9999")
        with patch("requests.post", side_effect=Exception("refused")):
            result = backend.perceive(str(audio_file))

        assert result.confidence == 0.0

    def test_modality(self):
        assert AudioBackend().modality == "audio"


# ---------------------------------------------------------------------------
# DocumentBackend (plaintext path)
# ---------------------------------------------------------------------------

class TestDocumentBackend:
    def test_reads_text_file(self, tmp_path):
        doc = tmp_path / "notes.txt"
        doc.write_text("London | CAPITAL_OF | United Kingdom\n", encoding="utf-8")

        backend = DocumentBackend()
        result = backend.perceive(str(doc))

        assert "London" in result.raw_text
        assert result.confidence > 0.5
        assert result.provenance.startswith("document:")

    def test_plain_string_passthrough(self):
        backend = DocumentBackend()
        result = backend.perceive("Einstein | DEVELOPED | Theory of Relativity")
        assert result.raw_text == "Einstein | DEVELOPED | Theory of Relativity"
        assert result.confidence == 1.0

    def test_modality(self):
        assert DocumentBackend().modality == "document"


# ---------------------------------------------------------------------------
# PerceptionAdapter.ingest_text
# ---------------------------------------------------------------------------

class TestPerceptionAdapterIngestText:
    def test_structured_triples_written_to_graph(self):
        adapter = make_adapter()
        pa = PerceptionAdapter(graph_adapter=adapter)
        edges = pa.ingest_text(
            "Paris | LOCATED_IN | France\nEiffel Tower | PART_OF | Paris"
        )
        assert len(edges) == 2
        sources = {e.source for e in edges}
        assert "Paris" in sources or "perception:Paris" in sources

    def test_unstructured_text_without_llm_returns_empty(self):
        pa = make_pa(llm_fn=None)
        edges = pa.ingest_text("The quick brown fox jumps over the lazy dog.")
        assert edges == []

    def test_unstructured_text_with_llm_calls_llm(self):
        llm_fn = MagicMock(return_value="Fox | JUMPS_OVER | Dog")
        pa = make_pa(llm_fn=llm_fn)
        edges = pa.ingest_text("The quick brown fox jumps over the lazy dog.")
        llm_fn.assert_called_once()
        assert len(edges) == 1

    def test_empty_text_returns_empty(self):
        pa = make_pa()
        assert pa.ingest_text("") == []


# ---------------------------------------------------------------------------
# PerceptionAdapter.ingest_image (mocked VisionBackend)
# ---------------------------------------------------------------------------

class TestPerceptionAdapterIngestImage:
    def _make_vision_backend(self, text: str, confidence: float) -> VisionBackend:
        backend = MagicMock(spec=VisionBackend)
        backend.perceive.return_value = PerceptionResult(
            raw_text=text,
            confidence=confidence,
            provenance="vision:test",
        )
        return backend

    def test_high_confidence_uses_heuristic(self):
        vision = self._make_vision_backend(
            "Cat | SITS_ON | Mat", confidence=0.9
        )
        llm_fn = MagicMock()
        pa = make_pa(vision=vision, llm_fn=llm_fn, confidence_threshold=0.7)
        edges = pa.ingest_image("img.jpg")
        assert len(edges) == 1
        llm_fn.assert_not_called()

    def test_low_confidence_uses_llm(self):
        vision = self._make_vision_backend(
            "Blurry image of something", confidence=0.3
        )
        llm_fn = MagicMock(return_value="Object | IN | Room")
        pa = make_pa(vision=vision, llm_fn=llm_fn, confidence_threshold=0.7)
        pa.ingest_image("img.jpg")
        llm_fn.assert_called_once()

    def test_no_vision_backend_raises(self):
        pa = make_pa()
        with pytest.raises(RuntimeError, match="No VisionBackend"):
            pa.ingest_image("img.jpg")

    def test_empty_result_returns_empty(self):
        vision = self._make_vision_backend("", confidence=0.9)
        pa = make_pa(vision=vision)
        assert pa.ingest_image("img.jpg") == []


# ---------------------------------------------------------------------------
# PerceptionAdapter.ingest_audio (mocked AudioBackend)
# ---------------------------------------------------------------------------

class TestPerceptionAdapterIngestAudio:
    def test_transcript_extracted(self):
        audio = MagicMock(spec=AudioBackend)
        audio.perceive.return_value = PerceptionResult(
            raw_text="Einstein | DEVELOPED | Theory of Relativity",
            confidence=0.85,
            provenance="audio:whisper",
        )
        pa = make_pa(audio=audio)
        edges = pa.ingest_audio("meeting.wav")
        assert len(edges) == 1

    def test_no_audio_backend_raises(self):
        pa = make_pa()
        with pytest.raises(RuntimeError, match="No AudioBackend"):
            pa.ingest_audio("meeting.wav")


# ---------------------------------------------------------------------------
# PerceptionAdapter.ingest_document (plaintext file)
# ---------------------------------------------------------------------------

class TestPerceptionAdapterIngestDocument:
    def test_reads_txt_and_extracts_triples(self, tmp_path):
        doc = tmp_path / "facts.txt"
        doc.write_text(
            "London | CAPITAL_OF | United Kingdom\n"
            "Big Ben | LOCATED_IN | London\n",
            encoding="utf-8",
        )
        pa = make_pa()
        edges = pa.ingest_document(str(doc))
        assert len(edges) == 2

    def test_auto_creates_document_backend_if_none(self, tmp_path):
        doc = tmp_path / "data.txt"
        doc.write_text("A | IS_A | B\n", encoding="utf-8")
        pa = make_pa()
        assert pa._document is None
        edges = pa.ingest_document(str(doc))
        assert pa._document is not None
        assert len(edges) == 1


# ---------------------------------------------------------------------------
# PerceptionStreamSource
# ---------------------------------------------------------------------------

class TestPerceptionStreamSource:
    def test_auto_modality_routes_by_extension(self, tmp_path):
        (tmp_path / "notes.txt").write_text("A | IS_A | B\n", encoding="utf-8")

        pa = make_pa()
        source = pa.as_source(str(tmp_path), modality="auto", poll_interval=0.0)

        gen = source.read()
        ev = next(gen)  # first file yields one StreamEvent
        source.stop()
        assert ev.source

    def test_repr(self, tmp_path):
        pa = make_pa()
        source = pa.as_source(str(tmp_path))
        assert "PerceptionStreamSource" in repr(source)

    def test_stop_halts_read(self, tmp_path):
        pa = make_pa()
        source = pa.as_source(str(tmp_path), poll_interval=0.0)
        source.stop()
        events = list(source.read())
        assert events == []
