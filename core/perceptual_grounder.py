"""
PerceptualGrounder — Phase 300.

Converts raw vision detections (YOLO bounding boxes, Florence-2 scene
captions, InsightFace identity matches) into (entity, relation, entity)
CandidateTriple objects suitable for SensoryThalamus ingestion.

Design principles:
  - No model loading here; receives pre-computed detection objects
  - Deterministic: same detections always produce same triples
  - Source tier 2 for YOLO/InsightFace (curated model outputs)
  - Source tier 3 for Florence-2 free-text captions (requires corroboration)
  - All triples carry frame_id and camera_id as provenance
"""
from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from core.knowledge_harvester import CandidateTriple

# ── Detection dataclasses (model-agnostic) ────────────────────────────────────

@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


@dataclass
class ObjectDetection:
    label: str
    confidence: float
    bbox: BoundingBox
    track_id: Optional[int] = None
    class_id: Optional[int] = None


@dataclass
class FaceIdentity:
    name: str
    confidence: float
    bbox: BoundingBox
    embedding: Optional[Any] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    emotion: Optional[str] = None


@dataclass
class SceneCaption:
    caption: str
    region_captions: List[Tuple[BoundingBox, str]] = field(default_factory=list)
    dense_tags: List[str] = field(default_factory=list)


@dataclass
class PerceptualEvent:
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    frame_id: int = 0
    camera_id: str = "obsbot_tiny3"
    ts: float = field(default_factory=time.time)
    detections: List[ObjectDetection] = field(default_factory=list)
    identities: List[FaceIdentity] = field(default_factory=list)
    scene: Optional[SceneCaption] = None
    frame_shape: Optional[Tuple[int, int]] = None  # (height, width)


# ── Spatial helpers ────────────────────────────────────────────────────────────

_PERSON_LABELS = {"person", "man", "woman", "child", "human", "face"}
_FURNITURE     = {"chair", "sofa", "couch", "table", "desk", "bed", "monitor"}
_ELECTRONICS   = {"laptop", "keyboard", "mouse", "phone", "tv", "screen", "camera"}
_STOPWORDS     = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "to", "for", "of", "and", "or", "with", "by", "from",
}


def _slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9_\s]", "", text)
    text = re.sub(r"\s+", "_", text)
    return text[:80]


def _spatial_relation(a: BoundingBox, b: BoundingBox, frame_w: float) -> str:
    """Coarse spatial relation between two bounding boxes."""
    ax, ay = a.center
    bx, by = b.center
    dx, dy = bx - ax, by - ay
    if abs(dx) < frame_w * 0.1 and dy > 0:
        return "is_above"
    if abs(dx) < frame_w * 0.1 and dy < 0:
        return "is_below"
    dist = (dx**2 + dy**2) ** 0.5
    if dist < frame_w * 0.2:
        return "is_near"
    return "is_in_same_scene_as"


# ── PerceptualGrounder ────────────────────────────────────────────────────────

class PerceptualGrounder:
    """
    Converts a PerceptualEvent into CandidateTriple objects.

    Usage
    -----
        grounder = PerceptualGrounder(camera_id="obsbot_tiny3")
        triples  = grounder.ground(event)
        # triples is List[CandidateTriple] — pass to SensoryThalamus
    """

    def __init__(
        self,
        camera_id: str = "obsbot_tiny3",
        scene_label: str = "workspace",
        min_detection_confidence: float = 0.4,
        min_identity_confidence: float = 0.6,
        emit_spatial_relations: bool = True,
    ) -> None:
        self._camera_id   = camera_id
        self._scene_label = scene_label
        self._min_det_conf  = min_detection_confidence
        self._min_id_conf   = min_identity_confidence
        self._emit_spatial  = emit_spatial_relations

    def ground(self, event: PerceptualEvent) -> List[CandidateTriple]:
        triples: List[CandidateTriple] = []
        frame_w = float(event.frame_shape[1]) if event.frame_shape else 1920.0
        prov    = f"camera:{self._camera_id}:frame:{event.frame_id}"

        # 1. Object detections
        valid_dets = [
            d for d in event.detections
            if d.confidence >= self._min_det_conf
        ]
        for det in valid_dets:
            triples.extend(self._detection_triples(det, prov))

        # 2. Face / person identities
        valid_ids = [
            i for i in event.identities
            if i.confidence >= self._min_id_conf
        ]
        for identity in valid_ids:
            triples.extend(self._identity_triples(identity, prov))

        # 3. Spatial relations between detected objects
        if self._emit_spatial and len(valid_dets) > 1:
            triples.extend(
                self._spatial_triples(valid_dets, frame_w, prov)
            )

        # 4. Scene caption → free-text triples (tier 3 — needs corroboration)
        if event.scene and event.scene.caption:
            triples.extend(self._caption_triples(event.scene, prov))

        return triples

    # ── Detection → triples ───────────────────────────────────────────────────

    def _detection_triples(
        self, det: ObjectDetection, prov: str
    ) -> List[CandidateTriple]:
        label = _slugify(det.label)
        node  = f"{label}_{det.track_id}" if det.track_id is not None else label
        triples = [
            self._make(node, "IS_IN_SCENE", self._scene_label,
                       det.label, self._scene_label, prov, det.confidence, tier=2),
            self._make(node, "IS_A", label,
                       det.label, label, prov, det.confidence, tier=2),
        ]
        return triples

    def _identity_triples(
        self, identity: FaceIdentity, prov: str
    ) -> List[CandidateTriple]:
        name   = _slugify(identity.name)
        triples = [
            self._make(name, "IS_PRESENT_IN", self._scene_label,
                       identity.name, self._scene_label, prov,
                       identity.confidence, tier=2),
            self._make(name, "IS_A", "person",
                       identity.name, "person", prov,
                       identity.confidence, tier=2),
        ]
        if identity.emotion:
            triples.append(
                self._make(name, "EXPRESSES", _slugify(identity.emotion),
                           identity.name, identity.emotion, prov,
                           identity.confidence * 0.8, tier=2)
            )
        if identity.age is not None:
            age_band = _age_band(identity.age)
            triples.append(
                self._make(name, "AGE_GROUP", age_band,
                           identity.name, age_band, prov,
                           identity.confidence * 0.7, tier=2)
            )
        return triples

    def _spatial_triples(
        self,
        dets: List[ObjectDetection],
        frame_w: float,
        prov: str,
    ) -> List[CandidateTriple]:
        triples = []
        for i, a in enumerate(dets):
            for b in dets[i + 1:]:
                if a.label == b.label:
                    continue
                rel   = _spatial_relation(a.bbox, b.bbox, frame_w)
                la    = _slugify(a.label)
                lb    = _slugify(b.label)
                na    = f"{la}_{a.track_id}" if a.track_id is not None else la
                nb    = f"{lb}_{b.track_id}" if b.track_id is not None else lb
                conf  = min(a.confidence, b.confidence) * 0.9
                triples.append(
                    self._make(na, rel.upper(), nb,
                               a.label, b.label, prov, conf, tier=2)
                )
        return triples

    # ── Florence-2 caption → triples (tier 3) ────────────────────────────────

    def _caption_triples(
        self, scene: SceneCaption, prov: str
    ) -> List[CandidateTriple]:
        triples = []
        # Dense tags → (tag, IS_IN_SCENE, scene_label) tier 3
        for tag in scene.dense_tags:
            node = _slugify(tag)
            if node and node not in _STOPWORDS:
                triples.append(
                    self._make(node, "IS_IN_SCENE", self._scene_label,
                               tag, self._scene_label, prov, 0.6, tier=3)
                )
        # Region captions → (region_entity, DESCRIBED_AS, caption_slug) tier 3
        for bbox, cap in scene.region_captions:
            tokens = [t for t in cap.lower().split() if t not in _STOPWORDS]
            if len(tokens) >= 2:
                subj = _slugify(tokens[0])
                obj  = _slugify("_".join(tokens[1:3]))
                if subj and obj:
                    triples.append(
                        self._make(subj, "DESCRIBED_AS", obj,
                                   tokens[0], "_".join(tokens[1:3]),
                                   prov, 0.55, tier=3)
                    )
        return triples

    # ── Factory ───────────────────────────────────────────────────────────────

    def _make(
        self,
        source: str,
        relation: str,
        target: str,
        source_name: str,
        target_name: str,
        prov: str,
        confidence: float,
        tier: int,
    ) -> CandidateTriple:
        return CandidateTriple(
            triple_id         = str(uuid.uuid4()),
            source            = source,
            relation          = relation,
            target            = target,
            source_name       = source_name,
            target_name       = target_name,
            source_url        = prov,
            source_tier       = tier,
            confidence        = round(min(1.0, max(0.0, confidence)), 4),
            corroborating_sources = [],
            raw               = {"provenance": prov},
        )


# ── Utility ───────────────────────────────────────────────────────────────────

def _age_band(age: int) -> str:
    if age < 13:   return "child"
    if age < 18:   return "teenager"
    if age < 30:   return "young_adult"
    if age < 60:   return "adult"
    return "senior"
