"""
VisionPipeline — Phase 300.

Camera capture loop for the OBSBOT Tiny 3 (or any UVC-compatible camera).
Runs YOLOv11, Florence-2, and InsightFace locally on the GPU, assembles a
PerceptualEvent per processed frame, and feeds it to SensoryThalamus.

All three model dependencies are optional and loaded lazily:
  - ultralytics  (YOLOv11)
  - transformers (Florence-2)
  - insightface  (ArcFace identity)
  - opencv-python (camera capture)

If any dependency is missing the pipeline degrades gracefully:
  - Missing opencv  → snapshot() raises RuntimeError; start() logs and exits
  - Missing YOLO    → detections list is empty
  - Missing Florence-2 → scene caption is None
  - Missing InsightFace → identities list is empty

Install all vision deps:
    pip install cerebrum-kg-core[vision]
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.perceptual_grounder import (
    BoundingBox,
    FaceIdentity,
    ObjectDetection,
    PerceptualEvent,
    SceneCaption,
)

logger = logging.getLogger(__name__)

_FLORENCE_TASK_CAPTION      = "<MORE_DETAILED_CAPTION>"
_FLORENCE_TASK_DENSE_REGION = "<DENSE_REGION_CAPTION>"
_FLORENCE_TASK_OD           = "<OD>"


# ── Status dataclass ───────────────────────────────────────────────────────────

@dataclass
class PipelineStatus:
    running: bool = False
    frames_captured: int = 0
    frames_processed: int = 0
    current_fps: float = 0.0
    models_loaded: Dict[str, bool] = field(default_factory=dict)
    camera_id: int = 0
    resolution: Tuple[int, int] = (1920, 1080)
    last_error: str = ""


# ── VisionPipeline ────────────────────────────────────────────────────────────

class VisionPipeline:
    """
    Real-time camera ingestion pipeline for CEREBRUM perceptual grounding.

    Parameters
    ----------
    thalamus
        SensoryThalamus to receive PerceptualEvent objects.
    grounder
        PerceptualGrounder (for consistency; thalamus also holds one, but
        callers may want to reuse the same grounder instance).
    camera_id
        OpenCV device index. OBSBOT Tiny 3 is typically 0 or 1 on Windows.
        Use `python -c "import cv2; [print(i) for i in range(5) if cv2.VideoCapture(i).isOpened()]"`
        to discover the correct index.
    resolution
        Capture resolution (width, height). OBSBOT supports 3840x2160@30 or
        1920x1080@60. 1080p recommended for real-time inference.
    fps_target
        Target capture FPS (may be limited by camera or USB bandwidth).
    frame_skip
        Process 1-in-N captured frames through the ML models. Set to 1 for
        every frame, 30 for roughly one graph update per second at 30fps.
    yolo_model
        Ultralytics YOLO model name or path. "yolo11n.pt" is fastest;
        "yolo11s.pt" is a good accuracy/speed tradeoff on RTX 5090.
    florence_model
        HuggingFace model ID for Florence-2. "microsoft/Florence-2-base" is
        recommended (270M params, ~80ms/frame on RTX 5090).
    insightface_model
        InsightFace model pack name. "buffalo_sc" is lightweight; "buffalo_l"
        has higher accuracy for identity matching.
    device
        PyTorch device string: "cuda" for RTX 5090, "cpu" for testing.
    """

    def __init__(
        self,
        thalamus: Any,
        grounder: Any,
        camera_id: int = 0,
        resolution: Tuple[int, int] = (1920, 1080),
        fps_target: int = 30,
        frame_skip: int = 5,
        yolo_model: str = "yolo11n.pt",
        florence_model: str = "microsoft/Florence-2-base",
        insightface_model: str = "buffalo_sc",
        device: str = "cuda",
    ) -> None:
        self._thalamus       = thalamus
        self._grounder       = grounder
        self._camera_id      = camera_id
        self._resolution     = resolution
        self._fps_target     = fps_target
        self._frame_skip     = max(1, frame_skip)
        self._yolo_name      = yolo_model
        self._florence_name  = florence_model
        self._insightface_name = insightface_model
        self._device         = device

        self._yolo           = None
        self._florence_model = None
        self._florence_proc  = None
        self._face_analysis  = None

        self._status = PipelineStatus(
            camera_id  = camera_id,
            resolution = resolution,
            models_loaded = {"yolo": False, "florence": False, "insightface": False},
        )

        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._frame_counter = 0
        self._fps_counter = 0
        self._fps_ts = time.time()

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start background capture + inference loop. Non-blocking."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="vision-pipeline",
            daemon=True,
        )
        self._thread.start()
        logger.info("VisionPipeline: started on camera %d.", self._camera_id)

    def stop(self) -> None:
        """Stop the capture loop. Blocks until thread exits (max 5s)."""
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        self._status.running = False
        logger.info("VisionPipeline: stopped. %d frames processed.",
                    self._status.frames_processed)

    def snapshot(self) -> PerceptualEvent:
        """
        Capture a single frame synchronously. Does not require start().
        Useful for testing and one-shot REST endpoints.
        Raises RuntimeError if OpenCV is not installed or camera unavailable.
        """
        cv2 = _require_cv2()
        cap = cv2.VideoCapture(self._camera_id)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(
                f"VisionPipeline: cannot open camera {self._camera_id}. "
                "Check USB connection and device index."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._resolution[1])
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            raise RuntimeError(f"VisionPipeline: failed to read frame from camera {self._camera_id}.")
        self._ensure_models()
        return self._process_frame(frame, frame_id=0)

    def status(self) -> dict:
        s = self._status
        return {
            "running":          s.running,
            "frames_captured":  s.frames_captured,
            "frames_processed": s.frames_processed,
            "current_fps":      round(s.current_fps, 2),
            "models_loaded":    dict(s.models_loaded),
            "camera_id":        s.camera_id,
            "resolution":       s.resolution,
            "last_error":       s.last_error,
        }

    # ── Capture loop ───────────────────────────────────────────────────────────

    def _capture_loop(self) -> None:
        cv2 = _try_import_cv2()
        if cv2 is None:
            logger.error("VisionPipeline: opencv-python not installed. "
                         "Run: pip install 'cerebrum-kg-core[vision]'")
            return

        cap = cv2.VideoCapture(self._camera_id)
        if not cap.isOpened():
            logger.error("VisionPipeline: cannot open camera %d.", self._camera_id)
            self._status.last_error = f"Cannot open camera {self._camera_id}"
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._resolution[1])
        cap.set(cv2.CAP_PROP_FPS, self._fps_target)

        self._ensure_models()
        self._status.running = True
        logger.info("VisionPipeline: camera opened at %dx%d.",
                    self._resolution[0], self._resolution[1])

        frame_interval = 1.0 / self._fps_target
        try:
            while not self._stop_evt.is_set():
                t0 = time.time()
                ret, frame = cap.read()
                if not ret:
                    logger.warning("VisionPipeline: dropped frame %d.", self._frame_counter)
                    time.sleep(0.01)
                    continue

                self._frame_counter += 1
                self._status.frames_captured = self._frame_counter
                self._update_fps()

                if self._frame_counter % self._frame_skip != 0:
                    continue

                try:
                    event = self._process_frame(frame, frame_id=self._frame_counter)
                    self._thalamus.ingest(event)
                    self._status.frames_processed += 1
                except Exception:
                    logger.exception("VisionPipeline: error processing frame %d.",
                                     self._frame_counter)

                # Pace to fps_target (rough — doesn't account for inference time)
                elapsed = time.time() - t0
                sleep_t = frame_interval - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)
        finally:
            cap.release()
            self._status.running = False

    # ── Frame processing ───────────────────────────────────────────────────────

    def _process_frame(self, frame: np.ndarray, frame_id: int = 0) -> PerceptualEvent:
        h, w = frame.shape[:2]
        event = PerceptualEvent(
            frame_id   = frame_id,
            camera_id  = f"obsbot_tiny3_dev{self._camera_id}",
            ts         = time.time(),
            frame_shape= (h, w),
        )

        # YOLO detections
        if self._yolo is not None:
            event.detections = self._run_yolo(frame)

        # InsightFace identity + emotion
        if self._face_analysis is not None:
            event.identities = self._run_insightface(frame)

        # Florence-2 scene caption (only when previous detections found something)
        if self._florence_model is not None and (event.detections or event.identities):
            event.scene = self._run_florence(frame)

        return event

    # ── YOLO inference ─────────────────────────────────────────────────────────

    def _run_yolo(self, frame: np.ndarray) -> List[ObjectDetection]:
        try:
            results = self._yolo.track(frame, persist=True, verbose=False)
            dets: List[ObjectDetection] = []
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    xyxy  = box.xyxy[0].tolist()
                    conf  = float(box.conf[0])
                    cls   = int(box.cls[0])
                    label = self._yolo.names.get(cls, str(cls))
                    tid   = int(box.id[0]) if box.id is not None else None
                    dets.append(ObjectDetection(
                        label      = label,
                        confidence = conf,
                        bbox       = BoundingBox(*xyxy),
                        track_id   = tid,
                        class_id   = cls,
                    ))
            return dets
        except Exception:
            logger.debug("VisionPipeline: YOLO inference error.", exc_info=True)
            return []

    # ── InsightFace inference ──────────────────────────────────────────────────

    def _run_insightface(self, frame: np.ndarray) -> List[FaceIdentity]:
        try:
            faces = self._face_analysis.get(frame)
            identities: List[FaceIdentity] = []
            for face in faces:
                bbox = BoundingBox(
                    float(face.bbox[0]), float(face.bbox[1]),
                    float(face.bbox[2]), float(face.bbox[3]),
                )
                identities.append(FaceIdentity(
                    name       = getattr(face, "name", "unknown_person"),
                    confidence = float(face.det_score),
                    bbox       = bbox,
                    embedding  = face.embedding if hasattr(face, "embedding") else None,
                    age        = int(face.age) if hasattr(face, "age") and face.age else None,
                    gender     = face.gender if hasattr(face, "gender") else None,
                ))
            return identities
        except Exception:
            logger.debug("VisionPipeline: InsightFace inference error.", exc_info=True)
            return []

    # ── Florence-2 inference ───────────────────────────────────────────────────

    def _run_florence(self, frame: np.ndarray) -> Optional[SceneCaption]:
        try:
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(frame[..., ::-1])  # BGR → RGB

            # Dense caption
            caption = self._florence_infer(pil_img, _FLORENCE_TASK_CAPTION)

            # Object detection tags
            od_result = self._florence_infer(pil_img, _FLORENCE_TASK_OD)
            tags: List[str] = []
            if isinstance(od_result, dict):
                tags = [item.get("label", "") for item in od_result.get("labels", [])]
            elif isinstance(od_result, str):
                tags = [t.strip() for t in od_result.split(",") if t.strip()]

            return SceneCaption(
                caption      = caption if isinstance(caption, str) else str(caption),
                dense_tags   = [t for t in tags if t],
                region_captions = [],
            )
        except Exception:
            logger.debug("VisionPipeline: Florence-2 inference error.", exc_info=True)
            return None

    def _florence_infer(self, image: Any, task: str) -> Any:
        import torch
        inputs = self._florence_proc(
            text=task, images=image, return_tensors="pt"
        ).to(self._device)
        with torch.no_grad():
            ids = self._florence_model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=256,
                num_beams=3,
            )
        result = self._florence_proc.batch_decode(ids, skip_special_tokens=False)[0]
        parsed = self._florence_proc.post_process_generation(
            result, task=task, image_size=(image.width, image.height)
        )
        return parsed.get(task, result)

    # ── Lazy model loading ─────────────────────────────────────────────────────

    def _ensure_models(self) -> None:
        self._load_yolo()
        self._load_florence()
        self._load_insightface()

    def _load_yolo(self) -> None:
        if self._yolo is not None:
            return
        try:
            from ultralytics import YOLO
            self._yolo = YOLO(self._yolo_name)
            if self._device == "cuda":
                self._yolo.to("cuda")
            self._status.models_loaded["yolo"] = True
            logger.info("VisionPipeline: YOLOv11 loaded (%s).", self._yolo_name)
        except ImportError:
            logger.warning("VisionPipeline: ultralytics not installed — YOLO disabled. "
                           "Run: pip install ultralytics")
        except Exception as e:
            logger.warning("VisionPipeline: YOLO load failed: %s", e)

    def _load_florence(self) -> None:
        if self._florence_model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
            self._florence_proc  = AutoProcessor.from_pretrained(
                self._florence_name, trust_remote_code=True
            )
            self._florence_model = AutoModelForCausalLM.from_pretrained(
                self._florence_name,
                torch_dtype = torch.float16 if self._device == "cuda" else torch.float32,
                trust_remote_code = True,
            ).to(self._device)
            self._florence_model.eval()
            self._status.models_loaded["florence"] = True
            logger.info("VisionPipeline: Florence-2 loaded (%s).", self._florence_name)
        except ImportError:
            logger.warning("VisionPipeline: transformers not installed — Florence-2 disabled. "
                           "Run: pip install transformers")
        except Exception as e:
            logger.warning("VisionPipeline: Florence-2 load failed: %s", e)

    def _load_insightface(self) -> None:
        if self._face_analysis is not None:
            return
        try:
            from insightface.app import FaceAnalysis
            self._face_analysis = FaceAnalysis(
                name=self._insightface_name,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self._face_analysis.prepare(ctx_id=0)
            self._status.models_loaded["insightface"] = True
            logger.info("VisionPipeline: InsightFace loaded (%s).", self._insightface_name)
        except ImportError:
            logger.warning("VisionPipeline: insightface not installed — face recognition disabled. "
                           "Run: pip install insightface onnxruntime-gpu")
        except Exception as e:
            logger.warning("VisionPipeline: InsightFace load failed: %s", e)

    # ── FPS tracking ───────────────────────────────────────────────────────────

    def _update_fps(self) -> None:
        self._fps_counter += 1
        now = time.time()
        elapsed = now - self._fps_ts
        if elapsed >= 2.0:
            self._status.current_fps = self._fps_counter / elapsed
            self._fps_counter = 0
            self._fps_ts = now


# ── Helpers ────────────────────────────────────────────────────────────────────

def _try_import_cv2() -> Optional[Any]:
    try:
        import cv2
        return cv2
    except ImportError:
        return None


def _require_cv2() -> Any:
    cv2 = _try_import_cv2()
    if cv2 is None:
        raise RuntimeError(
            "opencv-python is required for camera capture. "
            "Run: pip install 'cerebrum-kg-core[vision]'"
        )
    return cv2


def list_cameras(max_index: int = 5) -> List[int]:
    """Return device indices for all available cameras."""
    cv2 = _try_import_cv2()
    if cv2 is None:
        return []
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            found.append(i)
            cap.release()
    return found
