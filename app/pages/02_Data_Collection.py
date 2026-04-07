"""Data collection page for the Sign Language Prediction System."""

import streamlit as st
from src.config import GESTURE_CLASSES

st.title("📷 Data Collection")
st.markdown("Collect hand gesture samples for model training.")

st.warning("⚠️ Data collection requires a webcam. Run the CLI script for best results:")
st.code("python scripts/collect_data.py --gesture A --samples 100", language="bash")

st.markdown("### Supported Gestures")
st.write(", ".join(GESTURE_CLASSES))

st.markdown("### Instructions")
st.markdown(
    "1. Select a gesture class\n"
    "2. Run the collection script in terminal\n"
    "3. Hold your hand sign steady in front of the webcam\n"
    "4. Press **Q** to stop collection early"
)
