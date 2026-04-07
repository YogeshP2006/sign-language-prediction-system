"""CLI script for evaluating trained models."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.collector import GestureCollector
from src.data.preprocessor import DataPreprocessor
from src.models.model_trainer import ModelTrainer, MODEL_REGISTRY
from src.utils.metrics import compute_metrics, print_metrics
from src.utils import setup_logger

logger = setup_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained sign language models")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=list(MODEL_REGISTRY.keys()) + ["all"],
    )
    args = parser.parse_args()

    collector = GestureCollector()
    X, y = collector.load_existing_data()

    if len(X) == 0:
        logger.error("No data found.")
        sys.exit(1)

    preprocessor = DataPreprocessor()
    dataset = preprocessor.prepare_dataset(X, y)

    trainer = ModelTrainer()
    model_names = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]
    results = trainer.evaluate_all(dataset["X_test"], dataset["y_test"], model_names)

    print("\n=== Evaluation Results ===")
    for model_name, metrics in results.items():
        print(f"\n--- {model_name} ---")
        if "error" not in metrics:
            print_metrics(metrics)
        else:
            print(f"Error: {metrics['error']}")


if __name__ == "__main__":
    main()
