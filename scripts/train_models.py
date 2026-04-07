"""CLI script for training sign language classification models."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.collector import GestureCollector
from src.data.preprocessor import DataPreprocessor
from src.models.model_trainer import ModelTrainer, MODEL_REGISTRY
from src.utils import setup_logger

logger = setup_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train sign language classification models")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=list(MODEL_REGISTRY.keys()) + ["all"],
        help="Model to train",
    )
    parser.add_argument("--no-augment", action="store_true", help="Disable data augmentation")
    args = parser.parse_args()

    logger.info("Loading collected gesture data...")
    collector = GestureCollector()
    X, y = collector.load_existing_data()

    if len(X) == 0:
        logger.error("No data found. Run collect_data.py first.")
        sys.exit(1)

    logger.info(f"Loaded {len(X)} samples")

    preprocessor = DataPreprocessor()
    dataset = preprocessor.prepare_dataset(
        X, y, augment=not args.no_augment, save_dir=None
    )

    trainer = ModelTrainer()
    model_names = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]

    results = trainer.train_all(
        dataset["X_train"],
        dataset["y_train"],
        dataset["X_val"],
        dataset["y_val"],
        model_names=model_names,
    )

    trainer.save_results()
    logger.info(f"Training complete: {results}")


if __name__ == "__main__":
    main()
