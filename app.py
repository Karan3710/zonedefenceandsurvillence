import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image, ExifTags
import datetime
import os   # ✅ FIX 1

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    try:
        model_path = os.path.join(os.getcwd(), "best.pt")
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# ✅ LOAD MODEL
model = load_model()

if model is None:
    st.error("❌ Model not loaded. Make sure best.pt is present.")
    st.stop()

# ---------------- DAY/NIGHT DETECTION ----------------
def detect_day_night(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)

    if brightness > 100:
        return "DAY", brightness
    elif brightness > 60:
        return "DIM", brightness
    else:
        return "NIGHT", brightness

# ---------------- SMART THREAT ENGINE ----------------
def get_smart_threat(image, img, results, model):  # ✅ FIX 2

    day_night, brightness = detect_day_night(img)
    hour = datetime.datetime.now().hour

    score = 1 if day_night == "DAY" else 2 if day_night == "DIM" else 3

    if results[0].boxes is not None:
        for box in results[0].boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label == "person":
                score += 2
            elif label == "vehicle":
                score += 1

    if score <= 2:
        threat = "🟢 LOW"
    elif score <= 4:
        threat = "🟡 MEDIUM"
    elif score <= 6:
        threat = "🟠 HIGH"
    else:
        threat = "🔴 CRITICAL"

    return threat, hour, day_night, brightness

# ---------------- UI ----------------
st.set_page_config(page_title="AI Border Surveillance", layout="wide")

st.title("🚨 AI Border Surveillance System")

confidence = st.sidebar.slider("Confidence", 0.1, 1.0, 0.5)

# ---------------- IMAGE ----------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img = np.array(image)

    st.image(image, width='stretch')

    results = model(img, conf=confidence)
    annotated = results[0].plot()

    st.image(annotated, width='stretch')

    threat, hour, day_night, brightness = get_smart_threat(
        image, img, results, model  # ✅ FIX 2 applied
    )

    st.metric("Threat", threat)

# ---------------- WEBCAM ----------------
run = st.sidebar.checkbox("Start Webcam")

FRAME_WINDOW = st.empty()

if run:
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=confidence)
        annotated = results[0].plot()

        threat, hour, day_night, brightness = get_smart_threat(
            None, frame, results, model  # ✅ FIX 2 applied
        )

        FRAME_WINDOW.image(annotated, channels="BGR")

        if not st.session_state.get("Start Webcam", False):
            break

    cap.release()
