import io
import os
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

APP_TITLE = "YOLO Object Detection (Streamlit)"
MODEL_PATH = "best.pt"
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


@st.cache_resource
def load_model():
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Put best.pt in the repo root (same folder as app.py), "
            "or change MODEL_PATH."
        )
    return YOLO(MODEL_PATH)


def is_allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTS


def pil_to_bytes(img: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    st.header("Settings")
    conf_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
    iou_threshold = st.slider("IoU threshold", 0.0, 1.0, 0.45, 0.01)

# Load model once
try:
    model = load_model()
except Exception as e:
    st.error(str(e))
    st.stop()

uploaded = st.file_uploader(
    "Upload an image",
    type=[ext.replace(".", "") for ext in sorted(ALLOWED_EXTS)],
)

if uploaded is None:
    st.info("Upload an image to run detection.")
    st.stop()

if not is_allowed(uploaded.name):
    st.error(f"Unsupported file type. Allowed: {sorted(ALLOWED_EXTS)}")
    st.stop()

# Read image
image = Image.open(uploaded).convert("RGB")
image_np = np.array(image)

col1, col2 = st.columns(2, gap="large")
with col1:
    st.subheader("Original")
    st.image(image, use_container_width=True)

# Run inference
with st.spinner("Running YOLO inference..."):
    results = model.predict(
        source=image_np,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False,
    )

# Annotated image: Ultralytics plot returns BGR ndarray
annotated_bgr = results[0].plot()
annotated_rgb = annotated_bgr[..., ::-1]
annotated_pil = Image.fromarray(annotated_rgb)

# Extract detections
detections = []
for b in results[0].boxes:
    x1, y1, x2, y2 = b.xyxy[0].tolist()
    conf = float(b.conf[0].item())
    cls = int(b.cls[0].item())
    detections.append(
        {
            "box": [x1, y1, x2, y2],
            "confidence": conf,
            "class_id": cls,
            "class_name": model.names.get(cls, str(cls)),
        }
    )

with col2:
    st.subheader("Result (Annotated)")
    st.image(annotated_pil, use_container_width=True)

st.subheader("Detections (JSON)")
st.json(detections)

# Downloads
st.subheader("Download")
annotated_bytes = pil_to_bytes(annotated_pil, fmt="PNG")
st.download_button(
    label="Download annotated image (PNG)",
    data=annotated_bytes,
    file_name=f"result_{Path(uploaded.name).stem}.png",
    mime="image/png",
)

json_bytes = io.BytesIO()
json_bytes.write(str(detections).encode("utf-8"))
st.download_button(
    label="Download detections (TXT/JSON-like)",
    data=json_bytes.getvalue(),
    file_name=f"detections_{Path(uploaded.name).stem}.txt",
    mime="text/plain",
)
