import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image, ExifTags
import datetime

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    try:
        return YOLO(r"C:\Users\karan\Downloads\zonedetection\best.pt")   # keep best.pt in same folder
    except:
        return None

model = load_model()

if model is None:
    st.error("❌ Model not loaded. Make sure best.pt is in same folder.")
    st.stop()

# ---------------- EXTRACT IMAGE TIME ----------------
def get_image_time(image):
    try:
        exif = image._getexif()
        if exif is not None:
            for tag, value in exif.items():
                decoded = ExifTags.TAGS.get(tag, tag)
                if decoded == "DateTimeOriginal":
                    time_str = value.split(" ")[1]
                    hour = int(time_str.split(":")[0])
                    return hour
    except:
        pass
    return None

# ---------------- DAY/NIGHT DETECTION ----------------
def detect_day_night(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Normalize brightness
    brightness = np.mean(gray)

    # Better thresholds (IMPORTANT)
    if brightness > 100:
        return "DAY", brightness
    elif brightness > 60:
        return "DIM", brightness   # new middle state
    else:
        return "NIGHT", brightness

# ---------------- SMART THREAT ENGINE ----------------
def get_smart_threat(image, img, results):

    # Day/Night detection
    day_night, brightness = detect_day_night(img)

    # Get time (fallback system)
    import datetime
    hour = datetime.datetime.now().hour

    # Base score
    if day_night == "DAY":
        score = 1
    elif day_night == "DIM":
        score = 2
    else:
        score = 3

    # Detection scoring
    if results[0].boxes is not None:
        for box in results[0].boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label == "person":
                score += 2
            elif label == "vehicle":
                score += 1

    # Final logic
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
st.markdown("Smart Detection using **AI + Time + Brightness Intelligence**")

# Sidebar
st.sidebar.header("⚙️ Settings")
confidence = st.sidebar.slider("Confidence", 0.1, 1.0, 0.5)

# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img = np.array(image)

    st.subheader("📷 Uploaded Image")
    st.image(image, width='stretch')

    # Prediction
    results = model(img, conf=confidence)
    annotated = results[0].plot()

    st.subheader("🎯 Detection Results")
    st.image(annotated, width='stretch')

    # Threat Analysis
    threat, hour, day_night, brightness = get_smart_threat(image, img, results)

    st.subheader("🚨 Threat Level")
    st.metric("Threat", threat)

    # Alert
    if threat == "🔴 CRITICAL":
        st.error("🚨 ALERT: CRITICAL THREAT DETECTED!")

    # Intelligence Info
    st.subheader("🧠 Intelligence Info")
    st.write(f"🌙 Mode: {day_night}")
    st.write(f"💡 Brightness: {brightness:.2f}")

    # Detection Details
    st.subheader("📊 Detection Details")

    if results[0].boxes is not None:
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf_score = float(box.conf[0])
            st.write(f"🔹 {model.names[cls]} → {conf_score:.2f}")
    else:
        st.write("No objects detected")

# ---------------- WEBCAM ----------------

run = st.sidebar.checkbox("Start Webcam", key="webcam_toggle")

FRAME_WINDOW = st.empty()

if run:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Webcam not working")
            break

        # YOLO detection
        results = model(frame, conf=confidence)
        annotated = results[0].plot()

        # Threat detection
        threat, hour, day_night, brightness = get_smart_threat(
            image=None, img=frame, results=results
        )

        # Show frame
        FRAME_WINDOW.image(annotated, channels="BGR", width="stretch")

        # Show threat
        st.sidebar.metric("Threat Level", threat)

        # Small delay (IMPORTANT)
        import time
        time.sleep(0.03)

        # STOP condition (safe way)
        if not st.session_state.get("webcam_toggle", False):
            break

    cap.release()

else:
    st.write("📷 Webcam stopped")