import streamlit as st
import joblib
import os
import tempfile
from features import extract_mfcc_features
import numpy as np

st.set_page_config(page_title="VeraVoice Demo", layout="centered")
st.title("VeraVoice — Voice Deepfake Detector (Hackathon MVP)")

MODEL_PATH = os.path.join("models", "svm_mfcc.pkl")
if not os.path.exists(MODEL_PATH):
    st.error("Model not found. Ask Person A to put models/svm_mfcc.pkl in models/")
else:
    clf = joblib.load(MODEL_PATH)
    uploaded = st.file_uploader("Upload audio (wav/mp3)", type=["wav", "mp3"])
    if uploaded is not None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        try:
            feat = extract_mfcc_features(tmp_path)
            probs = clf.predict_proba(feat.reshape(1, -1))[0]
            classes = list(getattr(clf, "classes_", range(len(probs))))
            # robustly pick index for 'fake' label
            fake_idx = None
            for i, c in enumerate(classes):
                if str(c).lower() in ("1", "fake", "true"):
                    fake_idx = i
                    break
            if fake_idx is None:
                fake_idx = 1 if len(classes) > 1 else 0
            fake_prob = probs[fake_idx]
            st.audio(tmp_path)
            st.metric("Fake probability", f"{fake_prob*100:.1f}%")
        except Exception as e:
            st.error(f"Error processing file: {e}")
        finally:
            os.remove(tmp_path)