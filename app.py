import io
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import cv2  # pip install opencv-python-headless

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


def apply_filter_to_rgb(rgb: np.ndarray, mode: str,
                        h_min=0, s_min=0, v_min=0,
                        h_max=179, s_max=255, v_max=255) -> np.ndarray:
    """Apply filter to an RGB image and return RGB image (uint8)."""
    if mode == "none":
        return rgb

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if mode == "grayscale":
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        out = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

    if mode == "hsv_range":
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
        upper = np.array([h_max, s_max, v_max], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        filtered = cv2.bitwise_and(bgr, bgr, mask=mask)
        return cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)

    return rgb


def summarize_detections(detections: list[dict]) -> str:
    if not detections:
        return "No objects detected."
    counts = {}
    for d in detections:
        name = d["class_name"]
        counts[name] = counts.get(name, 0) + 1
    parts = [f"{v} {k}" for k, v in sorted(counts.items(), key=lambda x: (-x[1], x[0]))]
    return "Detected " + ", ".join(parts) + "."


def make_tts_mp3(text: str) -> bytes | None:
    """Optional: requires `pip install gTTS` + internet."""
    try:
        from gtts import gTTS
        buf = io.BytesIO()
        gTTS(text=text, lang="en").write_to_fp(buf)
        return buf.getvalue()
    except Exception:
        return None


# ---------------- UI ----------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

# session state toggles
if "show_filter" not in st.session_state:
    st.session_state.show_filter = False
if "play_audio" not in st.session_state:
    st.session_state.play_audio = False
if "filter_mode" not in st.session_state:
    st.session_state.filter_mode = "none"

with st.sidebar:
    st.header("Settings")
    conf_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
    iou_threshold = st.slider("IoU threshold", 0.0, 1.0, 0.45, 0.01)

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
annotated_rgb = annotated_bgr[..., ::-1]  # BGR->RGB

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

    # If filter is active, show filtered annotated image; otherwise show normal annotated image
    display_rgb = annotated_rgb

    if st.session_state.show_filter:
        mode = st.session_state.filter_mode
        # default HSV range values (if you select HSV filter, sliders appear below)
        h_min = st.session_state.get("h_min", 0)
        s_min = st.session_state.get("s_min", 0)
        v_min = st.session_state.get("v_min", 0)
        h_max = st.session_state.get("h_max", 179)
        s_max = st.session_state.get("s_max", 255)
        v_max = st.session_state.get("v_max", 255)

        display_rgb = apply_filter_to_rgb(
            annotated_rgb, mode, h_min, s_min, v_min, h_max, s_max, v_max
        )

    st.image(Image.fromarray(display_rgb), use_container_width=True)

    # ---- Buttons below image ----
    b1, b2 = st.columns(2)
    with b1:
        if st.button("Filter", use_container_width=True):
            st.session_state.show_filter = not st.session_state.show_filter
            st.session_state.play_audio = False
    with b2:
        if st.button("Audio", use_container_width=True):
            st.session_state.play_audio = True
            st.session_state.show_filter = False

    # ---- Filter options shown only when Filter is ON ----
    if st.session_state.show_filter:
        st.markdown("### Filter Options (Annotated image only)")
        mode = st.selectbox(
            "Filter mode",
            ["none", "grayscale", "hsv_range"],
            index=["none", "grayscale", "hsv_range"].index(st.session_state.filter_mode),
        )
        st.session_state.filter_mode = mode

        if mode == "hsv_range":
            c1, c2, c3 = st.columns(3)
            with c1:
                st.session_state.h_min = st.slider("H min", 0, 179, st.session_state.get("h_min", 0))
                st.session_state.h_max = st.slider("H max", 0, 179, st.session_state.get("h_max", 179))
            with c2:
                st.session_state.s_min = st.slider("S min", 0, 255, st.session_state.get("s_min", 0))
                st.session_state.s_max = st.slider("S max", 0, 255, st.session_state.get("s_max", 255))
            with c3:
                st.session_state.v_min = st.slider("V min", 0, 255, st.session_state.get("v_min", 0))
                st.session_state.v_max = st.slider("V max", 0, 255, st.session_state.get("v_max", 255))

    # ---- Audio panel ----
    if st.session_state.play_audio:
        st.markdown("### Audio")
        summary = summarize_detections(detections)
        st.write(summary)

        mp3 = make_tts_mp3(summary)
        if mp3 is None:
            st.warning("Audio not available. Install gTTS: `pip install gTTS` (needs internet).")
        else:
            st.audio(mp3, format="audio/mp3")

st.subheader("Detections (JSON)")
st.json(detections)

# Downloads (download what is currently displayed in the right panel)
st.subheader("Download")
download_img = Image.fromarray(display_rgb) if "display_rgb" in locals() else Image.fromarray(annotated_rgb)
download_bytes = pil_to_bytes(download_img, fmt="PNG")

st.download_button(
    label="Download displayed image (PNG)",
    data=download_bytes,
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
