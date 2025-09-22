import streamlit as st
import sounddevice as sd
from features import extract_mfcc_features
import numpy as np

st.header("Live Voice Detection")

if st.button("Record 2 sec Audio"):
    duration = 2  # seconds
    sr = 16000
    st.info("Recording...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    audio = audio.flatten()
    st.success("Recording done!")

    # Extract features and predict
    feat = extract_mfcc_features(audio, sr)
    feat_scaled = scaler.transform([feat])
    raw_prob = clf.predict_proba(feat_scaled)[0][1]
    smoothed_prob = 0.5 * raw_prob + 0.5 * 0.5

    st.write(f"Probability of fake voice: {smoothed_prob:.2f}")