import streamlit as st
import joblib
import os
import tempfile
import numpy as np
import queue
import soundfile as sf

from features import extract_mfcc_features
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase

# ---------------------------------------------------
# Streamlit UI setup
# ---------------------------------------------------
st.set_page_config(page_title="VeraVoice Demo", layout="centered")
st.title("VeraVoice — Voice Deepfake Detector (Hackathon MVP)")

# Load trained SVM model
MODEL_PATH = os.path.join("models", "svm_mfcc.pkl")
clf = joblib.load(MODEL_PATH)

# ---------------------------------------------------
# Modes: Upload vs Live Mic
# ---------------------------------------------------
mode = st.radio("Choose input mode:", ["📂 Upload File", "🎙️ Live Mic"])

# ---------------------------------------------------
# Upload file mode
# ---------------------------------------------------
if mode == "📂 Upload File":
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])
    if uploaded_file:
        # Save temp file
        tmp_path = tempfile.mktemp(suffix=".wav")
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            feat = extract_mfcc_features(tmp_path, sr=16000)
            probs = clf.predict_proba(feat.reshape(1, -1))[0]
            classes = list(getattr(clf, "classes_", range(len(probs))))

            # Find fake index
            fake_idx = None
            for i, c in enumerate(classes):
                if str(c).lower() in ("1", "fake", "true"):
                    fake_idx = i
                    break
            if fake_idx is None:
                fake_idx = 1 if len(classes) > 1 else 0

            fake_prob = probs[fake_idx]

            st.audio(uploaded_file)
            st.metric("Fake probability", f"{fake_prob*100:.1f}%")

        except Exception as e:
            st.error(f"Error processing file: {e}")
        finally:
            os.remove(tmp_path)
# ---------------------------------------------------
# Live mic mode
# ---------------------------------------------------
elif mode == "🎙️ Live Mic":
    class AudioProcessor(AudioProcessorBase):
        def __init__(self):
            self.recorded_frames = []

        def recv_audio_frame(self, frame):
            audio = frame.to_ndarray().flatten().astype(np.float32)
            self.recorded_frames.append(audio)
            return frame

    ctx = webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDRECV,
        audio_receiver_size=256,
        media_stream_constraints={"audio": True, "video": False},
        audio_processor_factory=AudioProcessor,
    )

    if ctx.audio_processor:
        if st.button("Stop & Analyze Recording"):
            if len(ctx.audio_processor.recorded_frames) == 0:
                st.warning("⚠️ No audio captured. Please speak into your mic before stopping.")
            else:
                # Concatenate recorded chunks
                audio_data = np.concatenate(ctx.audio_processor.recorded_frames, axis=0)

                # Save to temporary WAV file so it matches upload pipeline
                tmp_path = tempfile.mktemp(suffix=".wav")
                sf.write(tmp_path, audio_data, 16000)  # 16kHz same as training

                try:
                    feat = extract_mfcc_features(tmp_path, sr=16000)
                    probs = clf.predict_proba(feat.reshape(1, -1))[0]
                    classes = list(getattr(clf, "classes_", range(len(probs))))

                    # Find fake index
                    fake_idx = None
                    for i, c in enumerate(classes):
                        if str(c).lower() in ("1", "fake", "true"):
                            fake_idx = i
                            break
                    if fake_idx is None:
                        fake_idx = 1 if len(classes) > 1 else 0

                    fake_prob = probs[fake_idx]

                    st.audio(tmp_path)
                    st.metric("Fake probability (live)", f"{fake_prob*100:.1f}%")

                except Exception as e:
                    st.error(f"Error analyzing mic input: {e}")
                finally:
                    os.remove(tmp_path)

