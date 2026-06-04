"""
PerceptionAdapter â€” Real-world perception bridge for CEREBRUM.

Translates vision, audio, and document inputs into knowledge graph triples
via purpose-built perception models, feeding results through IngestionPipeline
into any GraphAdapter.

Architecture
------------
  [Image / Audio / Document / Text]
          â†“
  [PerceptionBackend] â†’ PerceptionResult (raw_text, confidence, provenance)
          â†“
  [TripleExtractor]   â†’ List[(subject, relation, object)]
          â†“  (if confidence < threshold â†’ LLM fallback via llm_bridge)
  [IngestionPipeline] â†’ ProcessedEdge
          â†“
  [GraphAdapter.add_edge()]

Backends
--------
  VisionBackend   â€” Florence-2, PaliGemma, LLaVA via OpenAI-compatible endpoint
  AudioBackend    â€” Whisper (faster-whisper local or HTTP endpoint)
  DocumentBackend â€” Docling / Surya (HTTP endpoint or local library)

All backends also accept raw text output and can run headlessly in tests.

StreamSource integration
------------------------
  PerceptionAdapter.as_source(watch_dir, modality) returns a StreamSource
  compatible with StreamAdapter, enabling live filesystem watching.

Usage
-----
    from adapters.perception_adapter import PerceptionAdapter, VisionBackend, AudioBackend
    from adapters.networkx_adapter import NetworkXAdapter
    from core.thalamus import IngestionPipeline

    adapter = NetworkXAdapter()
    pipeline = IngestionPipeline(namespace="perception")

    vision = VisionBackend(endpoint="http://localhost:11434/v1", model="llava")
    audio  = AudioBackend(endpoint="http://localhost:9000/transcribe")

    pa = PerceptionAdapter(
        graph_adapter=adapter,
        pipeline=pipeline,
        vision=vision,
        audio=audio,
        confidence_threshold=0.6,
    )

    edges = pa.ingest_image("photo.jpg", context="office_scene")
    edges = pa.ingest_audio("meeting.wav", context="project_meeting")
    edges = pa.ingest_text("The Eiffel Tower is located in Paris, France.")
"""
from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, TYPE_CHECKING, Tuple, Type

if TYPE_CHECKING:
    from core.graph_adapter import GraphAdapter
    from core.thalamus import IngestionPipeline, ProcessedEdge

logger = logging.getLogger("cerebrum.perception")


# ---------------------------------------------------------------------------
# PerceptionResult â€” output of any perception backend
# ---------------------------------------------------------------------------

@dataclass
class PerceptionResult:
    """
    Structured output from a perception backend.

    raw_text   : human-readable content (caption, transcript, extracted text)
    confidence : backend-reported confidence in [0, 1]; 1.0 = certain
    provenance : source tag, e.g. "vision:llava", "audio:whisper", "document:docling"
    structured : optional parsed JSON from the backend (object detection boxes, etc.)
    metadata   : arbitrary extra fields (timestamps, bounding boxes, page numbers)
    """
    raw_text: str
    confidence: float
    provenance: str
    structured: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# PerceptionBackend ABC
# ---------------------------------------------------------------------------

class PerceptionBackend(ABC):
    """Abstract base for perception backends (vision, audio, document)."""

    @abstractmethod
    def perceive(self, input_data: Any) -> PerceptionResult:
        """
        Process input_data and return a PerceptionResult.

        input_data : str path, bytes, or plain text depending on backend.
        """
        ...

    @property
    @abstractmethod
    def modality(self) -> str:
        """One of "vision", "audio", "document", "text"."""
        ...


# ---------------------------------------------------------------------------
# VisionBackend
# ---------------------------------------------------------------------------

class VisionBackend(PerceptionBackend):
    """
    Vision perception via any OpenAI-compatible vision endpoint.

    Supports Florence-2, PaliGemma, LLaVA, Gemma 4 Vision, or any server
    exposing ``POST /v1/chat/completions`` with image_url content blocks.

    Parameters
    ----------
    endpoint  : Base URL of the OpenAI-compatible server (no trailing slash).
                e.g. "http://localhost:11434/v1" for Ollama.
    model     : Model name to pass in the API request.
    prompt    : Vision instruction sent with every image.
    api_key   : API key if required (Ollama = "ollama", local = "").
    timeout   : HTTP timeout in seconds.
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:11434/v1",
        model: str = "llava",
        prompt: str = (
            "Describe what you see. List every identifiable object, person, "
            "location, and relationship. Be factual and concise."
        ),
        api_key: str = "local",
        timeout: float = 30.0,
    ) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._model = model
        self._prompt = prompt
        self._api_key = api_key
        self._timeout = timeout

    @property
    def modality(self) -> str:
        return "vision"

    def perceive(self, input_data: Any) -> PerceptionResult:
        """
        input_data : path-like (str / Path) to an image, or base64 data-URI string.
        """
        import base64
        try:
            import requests as _req
        except ImportError:
            raise ImportError("pip install requests")

        path = Path(input_data)
        if path.exists():
            image_bytes = path.read_bytes()
            suffix = path.suffix.lower().lstrip(".")
            mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                    "png": "image/png", "webp": "image/webp",
                    "gif": "image/gif"}.get(suffix, "image/jpeg")
            b64 = base64.b64encode(image_bytes).decode()
            image_url = f"data:{mime};base64,{b64}"
        else:
            image_url = str(input_data)

        payload = {
            "model": self._model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": self._prompt},
                ],
            }],
            "max_tokens": 512,
        }
        headers = {"Authorization": f"Bearer {self._api_key}",
                   "Content-Type": "application/json"}

        try:
            resp = _req.post(
                f"{self._endpoint}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            # OpenAI-compatible endpoints don't expose token-level confidence;
            # treat a clean HTTP 200 as high confidence.
            return PerceptionResult(
                raw_text=text,
                confidence=0.85,
                provenance=f"vision:{self._model}",
                metadata={"model": self._model},
            )
        except Exception as exc:
            logger.warning("VisionBackend error: %s", exc)
            return PerceptionResult(
                raw_text="",
                confidence=0.0,
                provenance=f"vision:{self._model}",
                metadata={"error": str(exc)},
            )


# ---------------------------------------------------------------------------
# AudioBackend
# ---------------------------------------------------------------------------

class AudioBackend(PerceptionBackend):
    """
    Audio transcription via Whisper.

    Supports:
    - Local faster-whisper (``pip install faster-whisper``)
    - HTTP endpoint accepting multipart/form-data (OpenAI ``/v1/audio/transcriptions``)

    Parameters
    ----------
    endpoint    : HTTP endpoint for remote Whisper. If None, uses local faster-whisper.
    model       : Whisper model size ("tiny", "base", "small", "medium", "large-v3").
    language    : ISO language code, or None for auto-detect.
    timeout     : HTTP timeout in seconds.
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        model: str = "large-v3",
        language: Optional[str] = None,
        timeout: float = 60.0,
    ) -> None:
        self._endpoint = endpoint
        self._model = model
        self._language = language
        self._timeout = timeout
        self._local_model = None

    @property
    def modality(self) -> str:
        return "audio"

    def _get_local_model(self) -> Any:
        if self._local_model is None:
            try:
                from faster_whisper import WhisperModel
                self._local_model = WhisperModel(self._model, device="auto")
            except ImportError:
                raise ImportError("pip install faster-whisper  OR  set endpoint=")
        return self._local_model

    def perceive(self, input_data: Any) -> PerceptionResult:
        """input_data : path-like to an audio file."""
        audio_path = str(input_data)

        if self._endpoint:
            return self._perceive_http(audio_path)
        return self._perceive_local(audio_path)

    def _perceive_local(self, audio_path: str) -> PerceptionResult:
        model = self._get_local_model()
        try:
            kwargs: Dict[str, Any] = {}
            if self._language:
                kwargs["language"] = self._language
            segments, info = model.transcribe(audio_path, **kwargs)
            text = " ".join(s.text for s in segments).strip()
            # avg_logprob is per-segment log probability; map to [0,1]
            import math
            confidence = min(1.0, max(0.0, math.exp(info.language_probability) if hasattr(info, "language_probability") else 0.8))
            return PerceptionResult(
                raw_text=text,
                confidence=confidence,
                provenance=f"audio:whisper-{self._model}",
                metadata={"language": getattr(info, "language", "unknown")},
            )
        except Exception as exc:
            logger.warning("AudioBackend (local) error: %s", exc)
            return PerceptionResult(raw_text="", confidence=0.0,
                                    provenance=f"audio:whisper-{self._model}",
                                    metadata={"error": str(exc)})

    def _perceive_http(self, audio_path: str) -> PerceptionResult:
        try:
            import requests as _req
        except ImportError:
            raise ImportError("pip install requests")
        try:
            with open(audio_path, "rb") as f:
                files = {"file": (Path(audio_path).name, f)}
                data: Dict[str, Any] = {"model": self._model}
                if self._language:
                    data["language"] = self._language
                resp = _req.post(self._endpoint, files=files, data=data,
                                 timeout=self._timeout)
                resp.raise_for_status()
                body = resp.json()
                text = body.get("text", "")
                return PerceptionResult(
                    raw_text=text,
                    confidence=0.85,
                    provenance=f"audio:whisper-{self._model}",
                )
        except Exception as exc:
            logger.warning("AudioBackend (http) error: %s", exc)
            return PerceptionResult(raw_text="", confidence=0.0,
                                    provenance=f"audio:whisper-{self._model}",
                                    metadata={"error": str(exc)})


# ---------------------------------------------------------------------------
# DocumentBackend
# ---------------------------------------------------------------------------

class DocumentBackend(PerceptionBackend):
    """
    Document parsing via Docling or Surya.

    Supports:
    - HTTP endpoint accepting a document upload (Docling server)
    - Local docling library (``pip install docling``)
    - Plain text pass-through (no backend needed)

    Parameters
    ----------
    endpoint : HTTP endpoint for Docling server. If None, uses local library.
    timeout  : HTTP timeout in seconds.
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        timeout: float = 60.0,
    ) -> None:
        self._endpoint = endpoint
        self._timeout = timeout

    @property
    def modality(self) -> str:
        return "document"

    def perceive(self, input_data: Any) -> PerceptionResult:
        """input_data : path-like to a document (PDF, DOCX, etc.) or plain text str."""
        if isinstance(input_data, str) and not Path(input_data).exists():
            return PerceptionResult(
                raw_text=input_data,
                confidence=1.0,
                provenance="document:plaintext",
            )

        doc_path = str(input_data)

        if self._endpoint:
            return self._perceive_http(doc_path)
        return self._perceive_local(doc_path)

    def _perceive_local(self, doc_path: str) -> PerceptionResult:
        try:
            from docling.document_converter import DocumentConverter
            converter = DocumentConverter()
            result = converter.convert(doc_path)
            text = result.document.export_to_markdown()
            return PerceptionResult(
                raw_text=text,
                confidence=0.90,
                provenance="document:docling",
                metadata={"source": doc_path},
            )
        except ImportError:
            pass
        except Exception as exc:
            logger.warning("DocumentBackend (local) error: %s", exc)
            return PerceptionResult(raw_text="", confidence=0.0,
                                    provenance="document:docling",
                                    metadata={"error": str(exc)})
        # Plain text fallback for .txt / .md files
        try:
            text = Path(doc_path).read_text(encoding="utf-8", errors="replace")
            return PerceptionResult(
                raw_text=text,
                confidence=0.95,
                provenance="document:plaintext",
            )
        except Exception as exc:
            return PerceptionResult(raw_text="", confidence=0.0,
                                    provenance="document:unknown",
                                    metadata={"error": str(exc)})

    def _perceive_http(self, doc_path: str) -> PerceptionResult:
        try:
            import requests as _req
        except ImportError:
            raise ImportError("pip install requests")
        try:
            with open(doc_path, "rb") as f:
                resp = _req.post(
                    self._endpoint,
                    files={"file": (Path(doc_path).name, f)},
                    timeout=self._timeout,
                )
                resp.raise_for_status()
                body = resp.json()
                text = body.get("markdown", body.get("text", ""))
                return PerceptionResult(
                    raw_text=text,
                    confidence=0.90,
                    provenance="document:docling-server",
                )
        except Exception as exc:
            logger.warning("DocumentBackend (http) error: %s", exc)
            return PerceptionResult(raw_text="", confidence=0.0,
                                    provenance="document:docling-server",
                                    metadata={"error": str(exc)})


# ---------------------------------------------------------------------------
# TripleExtractor â€” raw text â†’ (subject, relation, object) triples
# ---------------------------------------------------------------------------

_TRIPLE_RE = re.compile(
    r"^\s*([^\|\n]+?)\s*\|\s*([A-Z][A-Z0-9_]*)\s*\|\s*([^\|\n]+?)\s*$",
    re.MULTILINE,
)

_EXTRACT_PROMPT = """\
Extract knowledge graph triples from the text below.
Output one triple per line in exactly this format:
  subject | RELATION | object

Rules:
- Relations must be UPPERCASE_SNAKE_CASE (e.g. LOCATED_IN, CONTAINS, PART_OF)
- Subject and object are concise noun phrases, no articles
- Only include factual, verifiable triples
- Context tag if provided helps scope the subject
- Output ONLY the triples, no explanation

Context: {context}
Text: {text}
"""


def _extract_triples_heuristic(text: str) -> List[Tuple[str, str, str]]:
    """Fast path: parse lines already in 'subject | RELATION | object' format."""
    return [(m.group(1).strip(), m.group(2).strip(), m.group(3).strip())
            for m in _TRIPLE_RE.finditer(text)]


def _extract_triples_llm(
    text: str,
    context: str,
    llm_fn: Callable[[str], str],
) -> List[Tuple[str, str, str]]:
    """LLM-based extraction for unstructured text."""
    prompt = _EXTRACT_PROMPT.format(context=context or "none", text=text[:4000])
    try:
        response = llm_fn(prompt)
        triples = _extract_triples_heuristic(response)
        if not triples:
            # Fallback: try to parse "subject â†’ relation â†’ object" style
            for line in response.splitlines():
                parts = re.split(r"\s*[-â€“â†’|,]\s*", line.strip(), maxsplit=2)
                if len(parts) == 3 and all(p.strip() for p in parts):
                    subj, rel, obj = parts
                    rel = re.sub(r"\s+", "_", rel.strip().upper())
                    triples.append((subj.strip(), rel, obj.strip()))
        return triples
    except Exception as exc:
        logger.warning("LLM triple extraction failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# PerceptionAdapter
# ---------------------------------------------------------------------------

class PerceptionAdapter:
    """
    Bridges perception backends into CEREBRUM's knowledge graph.

    Parameters
    ----------
    graph_adapter        : GraphAdapter to write edges into.
    pipeline             : IngestionPipeline for normalization. If None, uses defaults.
    vision               : VisionBackend instance, or None to disable.
    audio                : AudioBackend instance, or None to disable.
    document             : DocumentBackend instance, or None to disable.
    confidence_threshold : Results below this score are routed through llm_fn.
    llm_fn               : callable(prompt: str) -> str for triple extraction.
                           Any llm_bridge adapter works. Required when
                           confidence_threshold > 0 and backends are in use.
    """

    def __init__(
        self,
        graph_adapter: "GraphAdapter",
        pipeline: Optional["IngestionPipeline"] = None,
        vision: Optional[VisionBackend] = None,
        audio: Optional[AudioBackend] = None,
        document: Optional[DocumentBackend] = None,
        confidence_threshold: float = 0.7,
        llm_fn: Optional[Callable[[str], str]] = None,
    ) -> None:
        self._adapter = graph_adapter
        self._pipeline = pipeline
        self._vision = vision
        self._audio = audio
        self._document = document
        self._threshold = confidence_threshold
        self._llm_fn = llm_fn

    # ------------------------------------------------------------------
    # Public ingestion methods
    # ------------------------------------------------------------------

    def ingest_image(self, image_path: Any, context: str = "") -> List["ProcessedEdge"]:
        """Perceive an image and write extracted triples to the graph."""
        if self._vision is None:
            raise RuntimeError("No VisionBackend configured.")
        result = self._vision.perceive(image_path)
        return self._process_result(result, context)

    def ingest_audio(self, audio_path: Any, context: str = "") -> List["ProcessedEdge"]:
        """Transcribe audio and write extracted triples to the graph."""
        if self._audio is None:
            raise RuntimeError("No AudioBackend configured.")
        result = self._audio.perceive(audio_path)
        return self._process_result(result, context)

    def ingest_document(self, doc_path: Any, context: str = "") -> List["ProcessedEdge"]:
        """Parse a document and write extracted triples to the graph."""
        if self._document is None:
            self._document = DocumentBackend()
        result = self._document.perceive(doc_path)
        return self._process_result(result, context)

    def ingest_text(self, text: str, context: str = "") -> List["ProcessedEdge"]:
        """Extract triples directly from plain text."""
        result = PerceptionResult(
            raw_text=text,
            confidence=1.0,
            provenance="text:direct",
        )
        return self._process_result(result, context)

    # ------------------------------------------------------------------
    # StreamSource bridge
    # ------------------------------------------------------------------

    def as_source(
        self,
        watch_dir: str,
        modality: str = "document",
        poll_interval: float = 2.0,
        context: str = "",
    ) -> "PerceptionStreamSource":
        """
        Return a StreamSource that watches watch_dir for new files and
        ingests them via the specified modality.

        Plugs directly into StreamAdapter.add_source().
        """
        return PerceptionStreamSource(
            adapter=self,
            watch_dir=watch_dir,
            modality=modality,
            poll_interval=poll_interval,
            context=context,
        )

    # ------------------------------------------------------------------
    # Internal pipeline
    # ------------------------------------------------------------------

    def _process_result(
        self,
        result: PerceptionResult,
        context: str,
    ) -> List["ProcessedEdge"]:
        if not result.raw_text:
            logger.debug("Empty perception result from %s", result.provenance)
            return []

        # Route through LLM extractor if confidence is low or text is unstructured
        if result.confidence < self._threshold and self._llm_fn is not None:
            triples = _extract_triples_llm(result.raw_text, context, self._llm_fn)
        else:
            triples = _extract_triples_heuristic(result.raw_text)
            if not triples and self._llm_fn is not None:
                triples = _extract_triples_llm(result.raw_text, context, self._llm_fn)

        if not triples:
            logger.debug("No triples extracted from %s output", result.provenance)
            return []

        return self._apply_to_graph(triples, result.provenance, result.confidence)

    def _apply_to_graph(
        self,
        triples: List[Tuple[str, str, str]],
        provenance: str,
        confidence: float,
    ) -> List["ProcessedEdge"]:
        from core.thalamus import IngestionPipeline

        pipeline = self._pipeline or IngestionPipeline(namespace="perception")
        edges: List["ProcessedEdge"] = []

        for subj, rel, obj in triples:
            if not (subj and rel and obj):
                continue
            # Discard LLM preamble bleed â€” real KG entities are concise
            if len(subj) > 120 or len(obj) > 120:
                continue
            processed = pipeline.process(
                source=subj,
                target=obj,
                relation=rel,
                metadata={"confidence": confidence, "provenance": provenance},
            )
            try:
                self._adapter.add_edge(
                    processed.source,
                    processed.target,
                    processed.relation,
                    confidence=processed.confidence,
                    provenance=processed.provenance,
                )
                edges.append(processed)
            except Exception as exc:
                logger.warning("add_edge failed (%s â†’ %s): %s", subj, obj, exc)

        logger.info(
            "Perceived %d triples from %s (confidence=%.2f)",
            len(edges), provenance, confidence,
        )
        return edges


# ---------------------------------------------------------------------------
# PerceptionStreamSource â€” polls a directory for new files
# ---------------------------------------------------------------------------

_VISION_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
_AUDIO_EXTS  = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus"}
_DOC_EXTS    = {".pdf", ".docx", ".txt", ".md", ".html", ".csv"}


class PerceptionStreamSource:
    """
    Watches a directory for new files and ingests them via PerceptionAdapter.

    Designed as a StreamSource drop-in for StreamAdapter.add_source().
    Uses a seen-files set to avoid re-processing.
    """

    def __init__(
        self,
        adapter: PerceptionAdapter,
        watch_dir: str,
        modality: str = "auto",
        poll_interval: float = 2.0,
        context: str = "",
    ) -> None:
        self._adapter = adapter
        self._watch_dir = Path(watch_dir)
        self._modality = modality
        self._poll_interval = poll_interval
        self._context = context
        self._seen: set = set()
        self._running = True

    def read(self) -> "Iterator":
        from core.stream_engine import StreamEvent

        while self._running:
            try:
                for path in sorted(self._watch_dir.iterdir()):
                    if not path.is_file() or path in self._seen:
                        continue
                    self._seen.add(path)
                    modality = self._modality
                    if modality == "auto":
                        ext = path.suffix.lower()
                        if ext in _VISION_EXTS:
                            modality = "vision"
                        elif ext in _AUDIO_EXTS:
                            modality = "audio"
                        else:
                            modality = "document"
                    try:
                        if modality == "vision":
                            edges = self._adapter.ingest_image(path, self._context)
                        elif modality == "audio":
                            edges = self._adapter.ingest_audio(path, self._context)
                        else:
                            edges = self._adapter.ingest_document(path, self._context)
                        for e in edges:
                            yield StreamEvent(
                                source=e.source,
                                relation=e.relation,
                                target=e.target,
                                timestamp=time.time(),
                                metadata={"provenance": e.provenance,
                                          "perception_file": str(path)},
                            )
                    except Exception as exc:
                        logger.warning("Failed to ingest %s: %s", path, exc)
            except Exception as exc:
                logger.warning("PerceptionStreamSource error: %s", exc)

            time.sleep(self._poll_interval)

    def stop(self) -> None:
        self._running = False

    def __repr__(self) -> str:
        return f"PerceptionStreamSource(dir={self._watch_dir}, modality={self._modality})"
