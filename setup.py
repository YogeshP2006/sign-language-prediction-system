from setuptools import setup, find_packages

setup(
    name="sign-language-prediction-system",
    version="1.0.0",
    description="Complete end-to-end Sign Language Prediction System using ML and Computer Vision",
    author="YogeshP2006",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "opencv-python>=4.8.0",
        "mediapipe>=0.10.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "tensorflow>=2.15.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "streamlit>=1.28.0",
        "pyttsx3>=2.90",
        "joblib>=1.3.0",
        "scipy>=1.11.0",
        "Pillow>=10.0.0",
        "tqdm>=4.65.0",
    ],
)
