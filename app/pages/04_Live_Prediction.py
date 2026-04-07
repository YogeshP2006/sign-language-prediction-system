"""Live prediction page for real-time gesture recognition."""

import streamlit as st

st.title("🎥 Live Prediction")
st.markdown("Real-time sign language gesture recognition.")

st.info("For best performance, use the CLI demo script:")
st.code("python scripts/demo_inference.py --model random_forest", language="bash")

st.markdown("### How to Use")
st.markdown(
    "1. Train models first (see Model Training page)\n"
    "2. Run the demo script from your terminal\n"
    "3. Show hand gestures to the webcam\n"
    "4. Predictions will appear on screen with confidence scores"
)
