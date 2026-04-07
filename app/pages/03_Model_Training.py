"""Model training page."""

import json
from pathlib import Path

import streamlit as st

from src.config import PROCESSED_DATA_DIR, TRAINED_MODELS_DIR

st.title("🧠 Model Training")
st.markdown("Train machine learning models on collected gesture data.")

st.info("Run the training script from the command line for full control:")
st.code("python scripts/train_models.py --model all", language="bash")

results_path = TRAINED_MODELS_DIR / "training_results.json"
if results_path.exists():
    st.markdown("### Latest Training Results")
    with open(results_path) as f:
        results = json.load(f)
    for model_name, metrics in results.items():
        with st.expander(f"📊 {model_name}"):
            for key, value in metrics.items():
                if key != "model_path":
                    st.write(
                        f"**{key}**: {value:.4f}"
                        if isinstance(value, float)
                        else f"**{key}**: {value}"
                    )
else:
    st.warning("No training results found. Train models using the script above.")
