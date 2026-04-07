"""Hand detection and landmark extraction using MediaPipe."""

from typing import List, Optional, Tuple

import numpy as np

from src.config import MEDIAPIPE_CONFIG
from src.utils import extract_landmark_features, setup_logger

logger = setup_logger(__name__)


class HandDetector:
    """Detects hands and extracts landmarks using MediaPipe."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or MEDIAPIPE_CONFIG
        import cv2
        import mediapipe as mp
        self.cv2 = cv2
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(**self.config)

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[list]]:
        """
        Detect hands in a BGR frame.

        Args:
            frame: BGR image from OpenCV.

        Returns:
            Tuple of (annotated_frame, list of landmark features or None).
        """
        rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        annotated = frame.copy()
        features_list = []

        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated,
                    hand_lm,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style(),
                )
                features = extract_landmark_features(hand_lm)
                features_list.append(features)

        return annotated, features_list if features_list else None

    def detect_single_hand(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Detect the primary hand and return its features."""
        annotated, features_list = self.detect(frame)
        if features_list:
            return annotated, features_list[0]
        return annotated, None

    def close(self) -> None:
        """Release MediaPipe resources."""
        self.hands.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
