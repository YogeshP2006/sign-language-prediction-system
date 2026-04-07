"""CLI script for collecting gesture data from webcam."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import GESTURE_CLASSES
from src.data.collector import GestureCollector
from src.utils import setup_logger

logger = setup_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Collect sign language gesture data")
    parser.add_argument("--gesture", type=str, help="Gesture label (e.g., A)")
    parser.add_argument("--samples", type=int, default=200, help="Samples to collect")
    parser.add_argument("--all", action="store_true", help="Collect all A-Z gestures")
    args = parser.parse_args()

    collector = GestureCollector(samples_per_class=args.samples)

    if args.all:
        logger.info("Collecting all gestures A-Z")
        collector.collect_all_gestures()
    elif args.gesture:
        label = args.gesture.upper()
        if label not in GESTURE_CLASSES:
            logger.error(f"Unknown gesture: {label}. Supported: {GESTURE_CLASSES}")
            sys.exit(1)
        collector.collect_gesture(label, args.samples)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
