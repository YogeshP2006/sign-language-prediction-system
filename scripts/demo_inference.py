"""CLI script for real-time sign language gesture demo."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2

from src.config import TRAINED_MODELS_DIR, PROCESSED_DATA_DIR
from src.inference.hand_detector import HandDetector
from src.inference.predictor import GesturePredictor
from src.inference.post_processor import PostProcessor
from src.utils import setup_logger

logger = setup_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Real-time sign language gesture demo")
    parser.add_argument("--model", type=str, default="random_forest", help="Model name")
    parser.add_argument("--threshold", type=float, default=0.7, help="Confidence threshold")
    args = parser.parse_args()

    suffix = ".h5" if args.model in ("cnn", "lstm") else ".pkl"
    model_path = TRAINED_MODELS_DIR / f"{args.model}_model{suffix}"

    if not model_path.exists():
        logger.error(f"Model not found: {model_path}. Train models first.")
        sys.exit(1)

    preprocessor_path = PROCESSED_DATA_DIR / "preprocessor.pkl"
    label_mapping_path = PROCESSED_DATA_DIR / "label_mapping.json"

    predictor = GesturePredictor(
        model_path=model_path,
        preprocessor_path=preprocessor_path if preprocessor_path.exists() else None,
        label_mapping_path=label_mapping_path if label_mapping_path.exists() else None,
    )
    post_processor = PostProcessor()
    detector = HandDetector()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Cannot open webcam")
        sys.exit(1)

    logger.info("Starting real-time inference. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        annotated, features = detector.detect_single_hand(frame)

        if features is not None:
            label, confidence, _ = predictor.predict_frame(features)
            confirmed = post_processor.update(label, confidence)
            display_label = confirmed or label
            color = (0, 255, 0) if confidence >= args.threshold else (0, 165, 255)
            cv2.putText(
                annotated, f"{display_label} ({confidence:.2f})",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2
            )
        else:
            cv2.putText(
                annotated, "No hand detected", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2
            )

        cv2.imshow("Sign Language Demo (Q to quit)", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()
