import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import datetime
import os
import time

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    try:
        general_model = YOLO("yolov8n.pt")   # person, vehicle
        custom_model = YOLO(os.path.join(os.getcwd(), "best.pt"))  # your model
        return general_model, custom_model
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None, None

general_model, custom_model = load_models()

if general_model is None:
    st.error("❌ Models not loaded")
    st.stop()

# ---------------- SMART THREAT ENGINE ----------------
def get_smart_threat(img, results_general, model):

    # -------- TIME --------
    hour = datetime.datetime.now().hour

    if 6 <= hour < 18:
        time_mode = "DAY"
        base_score = 1
    elif 18 <= hour < 22:
        time_mode = "EVENING"
        base_score = 2
    else:
        time_mode = "NIGHT"
        base_score = 3

    score = base_score

    # -------- BRIGHTNESS --------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)

    if brightness < 50:
        score += 1
    elif brightness > 180:
        score -= 1

    # -------- OBJECT COUNT (GENERAL MODEL) --------
   person_count = 0
vehicle_count = 0

vehicle_classes = ["car", "truck", "bus", "motorcycle", "bicycle"]

if results_general[0].boxes is not None:
    for box in results_general[0].boxes:

        cls = int(box.cls[0].item())   # ✅ FIX
        conf_score = float(box.conf[0].item())  # ✅ FIX
        label = general_model.names[cls].lower()  # ✅ USE GENERAL MODEL

        # DEBUG (IMPORTANT)
        st.write("Detected:", label, "Confidence:", conf_score)

        if conf_score < 0.25:
            continue

        if label == "person":
            person_count += 1
        elif label in vehicle_classes:
            vehicle_count += 1
    # -------- LOGIC --------
    if person_count >= 3:
        score += 4
    elif person_count >= 1:
        score += 2

    if vehicle_count >= 2:
        score += 2
    elif vehicle_count == 1:
        score += 1

    if time_mode == "NIGHT" and person_count > 0:
        score += 3

    # -------- FINAL THREAT --------
    if score <= 2:
        threat = "🟢 LOW"
    elif score <= 5:
        threat = "🟡 MEDIUM"
    elif score <= 8:
        threat = "🟠 HIGH"
    else:
        threat = "🔴 CRITICAL"

    return threat, hour, time_mode, brightness, person_count, vehicle_count

# ---------------- UI ----------------
st.set_page_config(page_title="AI Border Surveillance", layout="wide")

st.title("🚨 AI Border Surveillance System")
st.markdown("Hybrid AI: General Detection + Custom Border Intelligence")

# Sidebar
st.sidebar.header("⚙️ Settings")
confidence = st.sidebar.slider("Confidence", 0.1, 1.0, 0.25)

# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)

    # ✅ FIX COLOR (correct indent)
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    st.image(image, use_container_width=True)

    # ✅ LOWER CONFIDENCE (recommended)
    results_general = general_model(img, conf=confidence)

    # ✅ DEBUG
    st.write("Detections:", results_general[0].boxes)

    annotated = results_general[0].plot()

    st.image(annotated, use_container_width=True)
    # -------- THREAT --------
    threat, hour, time_mode, brightness, p_count, v_count = get_smart_threat(
        img, results_general, general_model
    )

    st.subheader("🚨 Threat Level")
    st.metric("Threat", threat)

    if threat == "🔴 CRITICAL":
        st.error("🚨 CRITICAL ALERT: Intrusion detected!")
    elif threat == "🟠 HIGH":
        st.warning("⚠️ HIGH RISK activity detected")

    # -------- INFO --------
    st.subheader("🧠 Intelligence Info")
    st.write(f"🕒 Time: {hour}:00 ({time_mode})")
    st.write(f"💡 Brightness: {brightness:.2f}")
    st.write(f"👤 Persons: {p_count}")
    st.write(f"🚗 Vehicles: {v_count}")

# ---------------- WEBCAM ----------------
run = st.sidebar.checkbox("Start Webcam")

FRAME_WINDOW = st.empty()

if run:
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Webcam not working")
            break

        results_general = general_model(frame, conf=confidence)
        annotated = results_general[0].plot()

        threat, hour, time_mode, brightness, p_count, v_count = get_smart_threat(
            frame, results_general, general_model
        )

        FRAME_WINDOW.image(annotated, channels="BGR", use_container_width=True)

        st.sidebar.metric("Threat Level", threat)

        time.sleep(0.03)

        if not st.session_state.get("Start Webcam", False):
            break

    cap.release()

else:
    st.write("📷 Webcam stopped")
