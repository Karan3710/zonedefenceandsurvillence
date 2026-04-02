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
        general_model = YOLO("yolov8n.pt")   # person + vehicle detection
        return general_model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

general_model = load_models()

if general_model is None:
    st.error("❌ Model not loaded")
    st.stop()

# ---------------- SMART THREAT ENGINE ----------------
def get_smart_threat(img, results, model):

    # -------- BRIGHTNESS BASED DAY/NIGHT --------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)

    if brightness < 70:
        time_mode = "NIGHT"
        score = 3
    elif brightness < 120:
        time_mode = "DIM"
        score = 2
    else:
        time_mode = "DAY"
        score = 1

    # -------- OBJECT COUNT --------
    person_count = 0
    vehicle_count = 0

    vehicle_classes = ["car", "truck", "bus", "motorcycle", "bicycle"]

    if results[0].boxes is not None:
        for box in results[0].boxes:
            cls = int(box.cls[0].item())
            conf_score = float(box.conf[0].item())
            label = model.names[cls].lower()

            if conf_score < 0.25:
                continue

            if label == "person":
                person_count += 1
            elif label in vehicle_classes:
                vehicle_count += 1

    # -------- THREAT LOGIC --------
    if person_count >= 1:
        score += 3

    if vehicle_count >= 1:
        score += 1

    # NIGHT BOOST
    if time_mode == "NIGHT":
        score += 2

    # CRITICAL NIGHT INTRUSION
    if time_mode == "NIGHT" and person_count > 0:
        score += 4

    # -------- FINAL THREAT --------
    if score <= 2:
        threat = "🟢 LOW"
    elif score <= 5:
        threat = "🟡 MEDIUM"
    elif score <= 8:
        threat = "🟠 HIGH"
    else:
        threat = "🔴 CRITICAL"

    return threat, time_mode, brightness, person_count, vehicle_count

# ---------------- UI ----------------
st.set_page_config(page_title="AI Border Surveillance", layout="wide")

st.title("🚨 AI Border Surveillance System")
st.markdown("Smart Detection using **AI + Night Intelligence**")

# Sidebar
st.sidebar.header("⚙️ Settings")
confidence = st.sidebar.slider("Confidence", 0.1, 1.0, 0.25)

# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)

    # ✅ Convert to OpenCV format
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    st.subheader("📷 Uploaded Image")
    st.image(image, use_container_width=True)

    # -------- DETECTION --------
    results = general_model(img, conf=confidence)

    annotated = results[0].plot()

    st.subheader("🎯 Detection Results")
    st.image(annotated, use_container_width=True)

    # -------- THREAT --------
    threat, time_mode, brightness, p_count, v_count = get_smart_threat(
        img, results, general_model
    )

    st.subheader("🚨 Threat Level")
    st.metric("Threat", threat)

    if threat == "🔴 CRITICAL":
        st.error("🚨 CRITICAL ALERT: Intrusion detected!")
    elif threat == "🟠 HIGH":
        st.warning("⚠️ HIGH RISK activity detected")

    # -------- INFO --------
    st.subheader("🧠 Intelligence Info")
    st.write(f"🌙 Mode: {time_mode}")
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

        results = general_model(frame, conf=confidence)
        annotated = results[0].plot()

        threat, time_mode, brightness, p_count, v_count = get_smart_threat(
            frame, results, general_model
        )

        FRAME_WINDOW.image(annotated, channels="BGR", use_container_width=True)

        st.sidebar.metric("Threat Level", threat)

        time.sleep(0.03)

        if not st.session_state.get("Start Webcam", False):
            break

    cap.release()

else:
    st.write("📷 Webcam stopped")
