"""Streamlit app - upload or camera photo. Uses the SAME CurrencyRecognitionPipeline as webcam.py."""
import json

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageOps

import config
from pipeline import CurrencyRecognitionPipeline, PipelineNotReady
from viz import annotate_still

st.set_page_config(page_title="Currency Note Recognizer", page_icon="💵", layout="centered")


@st.cache_resource
def get_pipeline():
    return CurrencyRecognitionPipeline()


def speak(text: str):
    components.html(f"<script>const u=new SpeechSynthesisUtterance({json.dumps(text)});u.lang='en-IN';"
                    "window.speechSynthesis.cancel();window.speechSynthesis.speak(u);</script>", height=0)


st.title("💵 Currency Note Recognizer")
st.caption("Note localisation → banknote/not-banknote detector → ResNet50 denomination classifier. "
           "Faces, objects and backgrounds are rejected as “No currency detected”.")

try:
    pipe = get_pipeline()
except PipelineNotReady as e:
    st.error(str(e))
    st.stop()

voice = st.sidebar.toggle("🔊 Speak the result", value=True)
st.sidebar.markdown(f"**Supported:** {', '.join(pipe.supported)}")
st.sidebar.markdown("**Tips:** one note per photo, flat, good light, the whole note visible.")
st.sidebar.caption("For live video run `python webcam.py` (adds temporal smoothing).")

tab_up, tab_cam = st.tabs(["📁 Upload", "📷 Camera"])
with tab_up:
    up = st.file_uploader("Upload a note image", type=["jpg", "jpeg", "png", "webp"])
with tab_cam:
    cam = st.camera_input("Take a photo of the note")

src = cam or up
if src:
    img = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    with st.spinner("Analysing..."):
        r = pipe.analyze_pil(img)
    st.image(annotate_still(img, r), use_container_width=True)

    if r.status == "ok":
        st.success(f"### {r.name}  —  {r.confidence:.1%} confident")
        st.info(r.description)
        if voice:
            speak(r.description)
    elif r.status == "no_currency":
        st.error("### No currency detected")
        st.caption(r.reason)
    elif r.status == "unidentified":
        st.warning(f"### {r.message}")
        st.write(r.description)
    else:
        st.warning(f"### {r.message}")

    with st.expander("Details (scores and thresholds)"):
        st.json({"status": r.status, "detector_score": r.detector_score, "margin": r.margin, "entropy": r.entropy,
                 "diagnostics": r.diagnostics, "reason": r.reason,
                 "thresholds": {"detector": config.DETECTOR_THRESHOLD, "confidence": config.CONFIDENCE_THRESHOLD,
                                "margin": config.MARGIN_THRESHOLD, "entropy_max": config.ENTROPY_THRESHOLD}})
        if r.top3:
            st.bar_chart(pd.DataFrame(r.top3, columns=["note", "probability"]).set_index("note"))
