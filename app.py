import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import datetime
import os
import time

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    try:
        model_path = os.path.join(os.getcwd(), "best.pt")
        return YOLO(model_path)
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()

if model is None:
    st.error("❌ Model not loaded. Make sure best.pt is present.")
    st.stop()

# ---------------- SMART THREAT ENGINE ----------------
def get_smart_threat(img, results, model):

    # -------- TIME (PRIMARY FACTOR) --------
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

    # -------- BRIGHTNESS (SECONDARY) --------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)

    if brightness < 50:
        score += 1
    elif brightness > 180:
        score -= 1

    # -------- OBJECT COUNT (FIXED) --------
    person_count = 0
    vehicle_count = 0

    vehicle_classes = ["car", "truck", "bus", "motorcycle", "bicycle"]

    if results[0].boxes is not None:
        for box in results[0].boxes:
            cls = int(box.cls[0])
            label = model.names[cls].lower()
            conf_score = float(box.conf[0])

            # Ignore weak detections
            if conf_score < 0.4:
                continue

            if label == "person":
                person_count += 1
            elif label in vehicle_classes:
                vehicle_count += 1

    # -------- INTELLIGENCE LOGIC --------
    if person_count >= 3:
        score += 4
    elif person_count >= 1:
        score += 2

    if vehicle_count >= 2:
        score += 2
    elif vehicle_count == 1:
        score += 1

    # NIGHT + PERSON = CRITICAL BOOST
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
    st.image(image, use_container_width=True)

    # Prediction
    results = model(img, conf=confidence)
    annotated = results[0].plot()

    st.subheader("🎯 Detection Results")
    st.image(annotated, use_container_width=True)

    # Threat Analysis
    threat, hour, time_mode, brightness, p_count, v_count = get_smart_threat(
        img, results, model
    )

    st.subheader("🚨 Threat Level")
    st.metric("Threat", threat)

    if threat == "🔴 CRITICAL":
        st.error("🚨 CRITICAL ALERT: Intrusion detected!")
    elif threat == "🟠 HIGH":
        st.warning("⚠️ HIGH RISK activity detected")

    # Info Panel
    st.subheader("🧠 Surveillance Intelligence")
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

        results = model(frame, conf=confidence)
        annotated = results[0].plot()

        threat, hour, time_mode, brightness, p_count, v_count = get_smart_threat(
            frame, results, model
        )

        FRAME_WINDOW.image(annotated, channels="BGR", use_container_width=True)

        st.sidebar.metric("Threat Level", threat)

        time.sleep(0.03)

        if not st.session_state.get("Start Webcam", False):
            break

    cap.release()

else:
    st.write("📷 Webcam stopped")
