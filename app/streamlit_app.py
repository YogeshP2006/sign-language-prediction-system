"""Main Streamlit application for the Sign Language Prediction System."""

import streamlit as st
from src.config import UI_CONFIG

st.set_page_config(
    page_title=UI_CONFIG["page_title"],
    page_icon=UI_CONFIG["page_icon"],
    layout=UI_CONFIG["layout"],
    initial_sidebar_state="expanded",
)

st.title("🤟 Sign Language Prediction System")
st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    st.info("**Step 1: Collect Data**\nUse the Data Collection page to gather gesture samples.")
with col2:
    st.info("**Step 2: Train Models**\nTrain ML models on the Model Training page.")
with col3:
    st.info("**Step 3: Predict**\nRun live predictions from the Live Prediction page.")

st.markdown("---")
st.markdown("### About")
st.markdown(
    "This system uses **MediaPipe** for hand landmark detection and multiple "
    "ML models (Random Forest, SVM, CNN, LSTM) for gesture recognition. "
    "Navigate using the sidebar to access different features."
)

st.markdown("### Quick Start")
st.code(
    "# Install dependencies\npip install -r requirements.txt\n\n"
    "# Run the app\nstreamlit run app/streamlit_app.py",
    language="bash",
)
