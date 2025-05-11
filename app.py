import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from utils import extract_features_from_audio

# Load trained model
model_path = 'model/music_genre_classifier.pkl'
model = joblib.load(model_path)

# Streamlit UI
st.set_page_config(page_title="Music Genre Classifier", layout="centered")
st.title("🎵 Music Genre Classifier")
st.caption("Upload a `.wav` file and get the predicted genre with confidence.")

uploaded_file = st.file_uploader("🎧 Upload a `.wav` file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    with st.spinner("Extracting features and making prediction..."):
        temp_file_path = "temp.wav"
        with open(temp_file_path, 'wb') as f:
            f.write(uploaded_file.read())

        # Visualize waveform
        st.subheader("📊 Audio Waveform")
        y, sr = librosa.load(temp_file_path)
        fig, ax = plt.subplots(figsize=(10, 3))
        librosa.display.waveshow(y, sr=sr, ax=ax)
        ax.set_title("Waveform")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        st.pyplot(fig)

        # Feature Extraction
        features = extract_features_from_audio(temp_file_path)
        os.remove(temp_file_path)

    if features is not None:
        proba = model.predict_proba([features])[0]
        max_proba = max(proba)
        predicted_label = model.classes_[np.argmax(proba)]

        # Confidence threshold
        threshold = 0.6
        if max_proba < threshold:
            st.warning("❓ Predicted Genre: Unknown (Low Confidence)")
        else:
            st.success(f"✅ Predicted Genre: **{predicted_label}** (Confidence: {max_proba:.2f})")

        # Show all probabilities
        st.subheader("📈 Prediction Confidence")
        for genre, p in zip(model.classes_, proba):
            st.write(f"- **{genre}**: {p:.2f}")
    else:
        st.error("❌ Feature extraction failed. Make sure the file is a valid `.wav` audio.")
