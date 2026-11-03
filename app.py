"""VeraVoice demo: is this voice a real person or a synthetic clone?"""

import io
import json
import os

import numpy as np
import soundfile as sf
import streamlit as st

from veravoice.audio import load_audio

MODEL_DIR = os.environ.get("VERAVOICE_MODEL", "model")

st.set_page_config(page_title="VeraVoice", page_icon="🎙️", layout="centered")
st.title("VeraVoice")
st.caption("Proof of human for voice: scores a clip for signs of text-to-speech or voice cloning.")


@st.cache_resource(show_spinner="Loading WavLM (first run downloads about 380 MB)...")
def get_detector():
    from veravoice.detector import Detector

    return Detector(MODEL_DIR)


if not os.path.exists(os.path.join(MODEL_DIR, "config.json")):
    st.error(
        f"No trained model found in `{MODEL_DIR}/`. Train one with `notebooks/kaggle_train.ipynb` "
        "(see the README), or set VERAVOICE_MODEL to a bundle folder."
    )
    st.stop()

detector = get_detector()
metrics = detector.config.get("metrics", {})

upload_tab, mic_tab = st.tabs(["Upload a clip", "Record with your mic"])
audio_bytes = None
with upload_tab:
    f = st.file_uploader("WAV, FLAC, MP3, M4A or OGG", type=["wav", "flac", "mp3", "m4a", "ogg"])
    if f:
        audio_bytes = f.getvalue()
with mic_tab:
    rec = st.audio_input("Speak for 3 to 10 seconds")
    if rec:
        audio_bytes = rec.getvalue()

if audio_bytes:
    st.audio(audio_bytes)
    with st.spinner("Analyzing..."):
        try:
            y, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        except Exception:
            # Formats libsndfile can't read (m4a, some mp3) go through librosa/audioread.
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".audio") as tmp:
                tmp.write(audio_bytes)
                tmp.flush()
                y, sr = load_audio(tmp.name), 16_000
        if len(y) < sr // 2:
            st.warning("That clip is under half a second. Record or upload at least 2 seconds of speech.")
            st.stop()
        result = detector.predict_waveform(np.asarray(y), sr)

    icon = {"likely human": "✅", "likely synthetic": "🚩", "uncertain": "🤔"}[result.verdict]
    st.subheader(f"{icon} {result.verdict.capitalize()}")
    st.metric("Synthetic-speech score", f"{result.spoof_probability * 100:.0f}%")
    if len(result.chunk_probabilities) > 1:
        st.caption("Score per 4-second window")
        st.bar_chart(result.chunk_probabilities)

with st.expander("How it works and how well it does"):
    st.markdown(
        "Audio is resampled to 16 kHz, trimmed and normalized, then passed through a frozen "
        "[WavLM](https://huggingface.co/microsoft/wavlm-base-plus) model. A small classifier learns "
        "which WavLM layers carry the artefacts that vocoders leave behind."
    )
    if metrics:
        st.json(json.loads(json.dumps(metrics)))
    st.markdown(
        "Treat the score as a signal, not proof. Heavy compression, phone audio and background "
        "music all make the job harder."
    )
