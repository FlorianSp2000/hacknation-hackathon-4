from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np

from core.types import DetectionResult, SegmentationResult, VLMResult, TrackingResult


class Detector(ABC):
    @abstractmethod
    def load(self) -> None:
        """Load model weights. Must call before predict."""
        ...

    @abstractmethod
    def predict(self, frame: np.ndarray) -> DetectionResult:
        """frame: BGR numpy (H,W,3). Returns DetectionResult."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Free GPU memory."""
        ...


class Segmenter(ABC):
    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def predict(self, frame: np.ndarray, text_prompt: str | None = None) -> SegmentationResult:
        """frame: BGR numpy (H,W,3). text_prompt: open-vocab query."""
        ...

    @abstractmethod
    def unload(self) -> None: ...


class VLM(ABC):
    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def predict(self, frame: np.ndarray, prompt: str) -> VLMResult:
        """frame: BGR numpy (H,W,3). prompt: text instruction."""
        ...

    @abstractmethod
    def unload(self) -> None: ...


class Tracker(ABC):
    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def update(self, frame: np.ndarray) -> TrackingResult:
        """Process frame. Maintains internal state across calls."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset tracker state (e.g. for new video)."""
        ...

    @abstractmethod
    def unload(self) -> None: ...


