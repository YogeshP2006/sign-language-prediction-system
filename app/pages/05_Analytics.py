"""Analytics dashboard page."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

from src.config import TRAINED_MODELS_DIR

st.title("📊 Analytics")
st.markdown("Model performance metrics and dataset statistics.")

results_path = TRAINED_MODELS_DIR / "training_results.json"
if results_path.exists():
    with open(results_path) as f:
        results = json.load(f)

    st.markdown("### Model Comparison")
    valid_models = [m for m in results if "error" not in results[m]]
    accuracies = [
        results[m].get("val_accuracy", results[m].get("train_accuracy", 0))
        for m in valid_models
    ]

    if valid_models:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(valid_models, accuracies, color="steelblue")
        ax.set_ylabel("Accuracy")
        ax.set_title("Model Accuracy Comparison")
        ax.set_ylim(0, 1.1)
        st.pyplot(fig)
        plt.close(fig)
else:
    st.info("No training results available yet. Train models first.")
