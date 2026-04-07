"""Home page for the Sign Language Prediction System."""

import streamlit as st
from pathlib import Path
from src.config import GESTURE_CLASSES, TRAINED_MODELS_DIR, GESTURES_DIR

st.title("🏠 Home")
st.markdown("Welcome to the Sign Language Prediction System!")

st.markdown("### System Status")
col1, col2, col3 = st.columns(3)

with col1:
    model_files = list(TRAINED_MODELS_DIR.glob("*.pkl")) + list(TRAINED_MODELS_DIR.glob("*.h5"))
    st.metric("Trained Models", len(model_files))

with col2:
    gesture_dirs = [d for d in GESTURES_DIR.iterdir() if d.is_dir()] if GESTURES_DIR.exists() else []
    st.metric("Gesture Classes Collected", len(gesture_dirs))

with col3:
    st.metric("Supported Classes", len(GESTURE_CLASSES))

st.markdown("### Supported Gestures")
st.write(" | ".join(GESTURE_CLASSES))
