"""Data collection module for sign language gestures."""

import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.config import (
    COLLECTION_CONFIG,
    GESTURES_DIR,
    NUM_LANDMARKS,
)
from src.utils import ensure_dir, setup_logger, extract_landmark_features

logger = setup_logger(__name__)


class GestureCollector:
    """Collects hand gesture data from webcam using MediaPipe."""

    def __init__(
        self,
        gestures_dir: Path = GESTURES_DIR,
        samples_per_class: int = COLLECTION_CONFIG["samples_per_class"],
        countdown: int = COLLECTION_CONFIG["countdown_seconds"],
    ):
        self.gestures_dir = Path(gestures_dir)
        self.samples_per_class = samples_per_class
        self.countdown = countdown

    def collect_gesture(
        self, gesture_label: str, num_samples: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Collect gesture samples from webcam.

        Args:
            gesture_label: Label for the gesture to collect.
            num_samples: Number of samples to collect (uses default if None).

        Returns:
            List of landmark feature arrays.
        """
        import cv2
        import mediapipe as mp

        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        num_samples = num_samples or self.samples_per_class
        samples = []
        save_dir = self.gestures_dir / gesture_label
        ensure_dir(save_dir)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Cannot open webcam")
            return samples

        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        ) as hands:
            self._countdown(cap, gesture_label)
            count = 0
            logger.info(f"Collecting {num_samples} samples for gesture '{gesture_label}'")

            while count < num_samples:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)

                if results.multi_hand_landmarks:
                    hand_lm = results.multi_hand_landmarks[0]
                    features = extract_landmark_features(hand_lm)
                    samples.append(features)
                    count += 1

                    mp_drawing.draw_landmarks(
                        frame,
                        hand_lm,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

                progress = f"Collecting '{gesture_label}': {count}/{num_samples}"
                cv2.putText(
                    frame, progress, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                )
                cv2.imshow("Data Collection", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Collection interrupted by user")
                    break

        cap.release()
        cv2.destroyAllWindows()

        if samples:
            self._save_samples(samples, gesture_label)
            logger.info(f"Saved {len(samples)} samples for gesture '{gesture_label}'")

        return samples

    def _countdown(self, cap, gesture_label: str) -> None:
        """Show countdown before data collection starts."""
        import cv2
        start_time = time.time()
        while time.time() - start_time < self.countdown:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            remaining = self.countdown - int(time.time() - start_time)
            cv2.putText(
                frame,
                f"Get ready for '{gesture_label}': {remaining}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
            cv2.imshow("Data Collection", frame)
            cv2.waitKey(1)

    def _save_samples(self, samples: List[np.ndarray], gesture_label: str) -> None:
        """Save collected samples to CSV file."""
        save_dir = self.gestures_dir / gesture_label
        ensure_dir(save_dir)
        timestamp = int(time.time())
        csv_path = save_dir / f"{gesture_label}_{timestamp}.csv"

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = [f"lm_{i}_{ax}" for i in range(NUM_LANDMARKS) for ax in ["x", "y", "z"]]
            writer.writerow(header + ["label"])
            for sample in samples:
                writer.writerow(list(sample) + [gesture_label])

    def collect_all_gestures(
        self, gesture_list: Optional[List[str]] = None
    ) -> Dict[str, List[np.ndarray]]:
        """Collect data for multiple gestures sequentially."""
        from src.config import GESTURE_CLASSES
        gesture_list = gesture_list or GESTURE_CLASSES
        all_data: Dict[str, List[np.ndarray]] = {}

        for gesture in gesture_list:
            logger.info(f"Starting collection for gesture: {gesture}")
            samples = self.collect_gesture(gesture)
            all_data[gesture] = samples

        return all_data

    def load_existing_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load all existing collected data from CSV files."""
        X, y = [], []

        if not self.gestures_dir.exists():
            logger.warning(f"Gestures directory not found: {self.gestures_dir}")
            return np.array(X), np.array(y)

        for gesture_dir in sorted(self.gestures_dir.iterdir()):
            if not gesture_dir.is_dir():
                continue
            label = gesture_dir.name
            for csv_file in gesture_dir.glob("*.csv"):
                try:
                    with open(csv_file, "r") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            features = [float(row[k]) for k in row if k != "label"]
                            X.append(features)
                            y.append(label)
                except Exception as e:
                    logger.error(f"Error reading {csv_file}: {e}")

        if X:
            logger.info(f"Loaded {len(X)} samples across {len(set(y))} classes")

        return np.array(X, dtype=np.float32), np.array(y)
