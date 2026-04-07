"""Post-processing for prediction smoothing and confirmation."""

from collections import Counter, deque
from typing import Optional

from src.config import INFERENCE_CONFIG
from src.utils import setup_logger

logger = setup_logger(__name__)


class PostProcessor:
    """Smooths and confirms gesture predictions over time."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or INFERENCE_CONFIG
        self.buffer: deque = deque(maxlen=self.config["prediction_buffer_size"])
        self.confirmed_gesture: Optional[str] = None
        self.confirmed_confidence: float = 0.0

    def update(self, label: str, confidence: float) -> Optional[str]:
        """
        Update buffer and return confirmed gesture if threshold met.

        Args:
            label: Predicted gesture label.
            confidence: Prediction confidence.

        Returns:
            Confirmed gesture label if confidence threshold is met, else None.
        """
        if confidence >= self.config["confidence_threshold"]:
            self.buffer.append(label)

        if len(self.buffer) == self.buffer.maxlen:
            most_common, count = Counter(self.buffer).most_common(1)[0]
            ratio = count / len(self.buffer)
            if ratio >= self.config["confirmation_threshold"]:
                self.confirmed_gesture = most_common
                self.confirmed_confidence = confidence
                return most_common

        return None

    def get_stable_prediction(self) -> Optional[str]:
        """Return the most stable prediction from recent history."""
        if not self.buffer:
            return None
        most_common = Counter(self.buffer).most_common(1)[0][0]
        return most_common

    def reset(self) -> None:
        """Reset post-processor state."""
        self.buffer.clear()
        self.confirmed_gesture = None
        self.confirmed_confidence = 0.0
