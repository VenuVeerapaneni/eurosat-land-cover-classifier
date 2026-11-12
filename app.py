from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image

from predict import load_model, predict_image

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "outputs" / "best_model.pt"

st.set_page_config(page_title="EuroSAT Explorer", page_icon="🛰️", layout="wide")
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #08111f 0%, #10263a 55%, #153a42 100%); color: #f3f7fb; }
.block-container { max-width: 1120px; padding-top: 2.6rem; }
.hero { padding: 1.5rem 1.8rem; border: 1px solid #ffffff20; border-radius: 20px;
        background: #ffffff0b; margin-bottom: 1.2rem; }
.hero h1 { margin: 0; font-size: 2.5rem; color: #f4f8fb; letter-spacing: -0.04em; }
.hero p { color: #b8c7d3; margin: .5rem 0 0; font-size: 1.05rem; }
.result { padding: 1.1rem 1.3rem; border-radius: 16px; background: #ffffff10;
          border: 1px solid #ffffff20; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>EuroSAT Explorer</h1>
  <p>Upload a satellite image patch and explore the model’s land-cover prediction.</p>
</div>
""", unsafe_allow_html=True)

if not CHECKPOINT.exists():
    st.error("Model checkpoint is missing. Run the training instructions in README.md first.")
    st.stop()

@st.cache_resource
def get_model():
    return load_model(str(CHECKPOINT))

uploaded = st.file_uploader(
    "Choose a satellite image",
    type=["jpg", "jpeg", "png"],
    help="For meaningful predictions, use a satellite patch similar to the EuroSAT dataset.",
)

st.caption("The model was trained on 64×64 RGB Sentinel-2 patches. Predictions on ordinary photos may be unreliable.")

if uploaded is not None:
    try:
        with Image.open(uploaded) as image:
            image.load()
            original, label, confidence, cam = predict_image(image, *get_model())

        left, right = st.columns(2, gap="large")
        with left:
            st.subheader("Uploaded image")
            st.image(original, use_container_width=True)
        with right:
            st.subheader("Model attention")
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(np.asarray(original) / 255.0)
            ax.imshow(cam, cmap="jet", alpha=0.38)
            ax.axis("off")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.markdown(
            f'<div class="result"><div style="color:#b8c7d3">PREDICTED LAND-COVER CLASS</div>'
            f'<div style="font-size:2rem;font-weight:700;margin:.2rem 0">{label}</div>'
            f'<div style="color:#b8c7d3">Model confidence: <b>{confidence:.1%}</b></div></div>',
            unsafe_allow_html=True,
        )
    except Exception as exc:
        st.error(f"Could not read or predict this image: {exc}")
else:
    st.info("Choose a JPG or PNG to see the predicted class, confidence, and Grad-CAM visualization.")
