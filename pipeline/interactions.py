from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import tempfile
import urllib.request

import cv2
import numpy as np

from core.types import DetectionResult, HandPose, Interaction

try:
    import mediapipe as mp
except Exception:
    mp = None


@dataclass
class InteractionConfig:
    enabled: bool = False
    max_hands: int = 2
    contact_threshold: float = 0.2
    bbox_padding: int = 12
    min_detect_conf: float = 0.5
    min_track_conf: float = 0.5


class InteractionEngine:
    """Optional MediaPipe-based hand-to-object interaction inference."""

    def __init__(self, config: InteractionConfig):
        self.config = config
        self.available = False
        self._hands = None
        self.error: str | None = None
        if not config.enabled:
            return
        if mp is None:
            self.error = "mediapipe import failed"
            return
        try:
            self._hands = self._create_tasks_landmarker()
            self.available = True
        except Exception as exc:
            self.error = str(exc)
            self.available = False

    def _create_tasks_landmarker(self):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        task_path = os.environ.get("MEDIAPIPE_HAND_LANDMARKER_PATH", "").strip()
        if not task_path:
            for candidate in (
                Path("hand_landmarker.task"),
                Path("models") / "hand_landmarker.task",
                Path("assets") / "hand_landmarker.task",
            ):
                if candidate.exists():
                    task_path = str(candidate)
                    break

        if not task_path:
            model_dir = Path(tempfile.gettempdir()) / "world2data_models"
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / "hand_landmarker.task"
            if not model_path.exists():
                url = (
                    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                    "hand_landmarker/float16/1/hand_landmarker.task"
                )
                try:
                    urllib.request.urlretrieve(url, model_path)
                except Exception as exc:
                    raise RuntimeError(
                        "Could not download hand_landmarker.task. "
                        "Set MEDIAPIPE_HAND_LANDMARKER_PATH to a local .task file."
                    ) from exc
            task_path = str(model_path)

        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=task_path),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=self.config.max_hands,
            min_hand_detection_confidence=self.config.min_detect_conf,
            min_hand_presence_confidence=self.config.min_track_conf,
            min_tracking_confidence=self.config.min_track_conf,
        )
        return vision.HandLandmarker.create_from_options(options)

    def close(self) -> None:
        if self._hands is not None:
            close_fn = getattr(self._hands, "close", None)
            if callable(close_fn):
                close_fn()
            self._hands = None

    def process(
        self, frame: np.ndarray, detection: DetectionResult | None
    ) -> tuple[list[HandPose], list[Interaction]]:
        if not self.available or self._hands is None:
            return [], []

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        hand_poses: list[HandPose] = []
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self._hands.detect(mp_image)
        landmarks_list = getattr(results, "hand_landmarks", None) or []
        handedness_list = getattr(results, "handedness", None) or []
        if not landmarks_list:
            return [], []

        for idx, landmarks in enumerate(landmarks_list):
            points: list[tuple[int, int]] = []
            xs: list[int] = []
            ys: list[int] = []
            for lm in landmarks:
                x = int(np.clip(lm.x * w, 0, w - 1))
                y = int(np.clip(lm.y * h, 0, h - 1))
                points.append((x, y))
                xs.append(x)
                ys.append(y)

            handedness = "unknown"
            score = 0.0
            if idx < len(handedness_list) and handedness_list[idx]:
                cat = handedness_list[idx][0]
                handedness = getattr(cat, "category_name", "unknown") or "unknown"
                score = float(getattr(cat, "score", 0.0) or 0.0)

            hand_poses.append(
                HandPose(
                    hand_id=f"hand_{idx + 1}",
                    handedness=handedness,
                    score=score,
                    bbox=(min(xs), min(ys), max(xs), max(ys)),
                    keypoints=points,
                )
            )

        interactions = self._compute_interactions(hand_poses, detection)
        return hand_poses, interactions

    def _compute_interactions(
        self, hands: list[HandPose], detection: DetectionResult | None
    ) -> list[Interaction]:
        if detection is None or not detection.boxes:
            return []

        interactions: list[Interaction] = []
        pad = self.config.bbox_padding

        for hand in hands:
            best: Interaction | None = None
            best_score = 0.0

            for box_idx, box in enumerate(detection.boxes):
                x1 = int(box.x1) - pad
                y1 = int(box.y1) - pad
                x2 = int(box.x2) + pad
                y2 = int(box.y2) + pad

                inside = 0
                for x, y in hand.keypoints:
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        inside += 1
                score = inside / max(len(hand.keypoints), 1)

                if score > best_score:
                    best_score = score
                    best = Interaction(
                        hand_id=hand.hand_id,
                        target_class=box.class_name,
                        target_index=box_idx,
                        contact_score=score,
                    )

            if best is not None and best.contact_score >= self.config.contact_threshold:
                interactions.append(best)

        return interactions
