import streamlit as st
import cv2
import numpy as np
from gtts import gTTS
import tempfile
from PIL import Image

# ------------------------------------------------------
# SIMPLE COLOR NAMES DICTIONARY (BEGINNER FRIENDLY)
# ------------------------------------------------------
COLOR_NAMES = {
    (255, 0, 0): "Red",
    (0, 255, 0): "Green",
    (0, 0, 255): "Blue",
    (255, 255, 0): "Yellow",
    (255, 165, 0): "Orange",
    (128, 0, 128): "Purple",
    (0, 255, 255): "Cyan",
    (255, 192, 203): "Pink",
    (165, 42, 42): "Brown",
    (0, 0, 0): "Black",
    (255, 255, 255): "White",
    (128, 128, 128): "Gray"
}

def get_closest_color_name(B, G, R):
    min_dist = float("inf")
    closest_name = "Unknown"
    for (r, g, b), name in COLOR_NAMES.items():
        dist = (r - R)**2 + (g - G)**2 + (b - B)**2
        if dist < min_dist:
            min_dist = dist
            closest_name = name
    return closest_name


# ------------------------------------------------------
# SPEAK COLOR NAME
# ------------------------------------------------------
def speak(text):
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        return fp.name


# ------------------------------------------------------
# STREAMLIT UI
# ------------------------------------------------------
st.title("🎨 Color Blindness Assist Tool")
st.write("Beginner-friendly tool to detect and speak colors.")

tab1, tab2 = st.tabs(["🟢 Webcam Color Detector", "🖼️ Image Upload & Cursor Color Viewer"])

# ------------------------------------------------------
# TAB 1 — WEBCAM COLOR DETECTOR
# ------------------------------------------------------
with tab1:
    st.subheader("Webcam Color Detection")

    run_cam = st.checkbox("Start Webcam")

    if run_cam:
        cam = cv2.VideoCapture(0)

        if not cam.isOpened():
            st.error("Webcam not found!")
        else:
            frame_window = st.image([])

            while run_cam:
                ret, frame = cam.read()
                if not ret:
                    break

                # Take center pixel
                h, w, _ = frame.shape
                cx, cy = w//2, h//2
                B, G, R = frame[cy, cx]

                color_name = get_closest_color_name(B, G, R)

                # Draw circle + text
                cv2.circle(frame, (cx, cy), 8, (0, 0, 0), -1)
                cv2.putText(frame, color_name, (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_window.image(frame)

                # Speak
                if st.button("🔊 Speak Color"):
                    mp3_path = speak(color_name)
                    st.audio(mp3_path)

            cam.release()

# ------------------------------------------------------
# TAB 2 — IMAGE UPLOAD MODE
# ------------------------------------------------------
with tab2:
    st.subheader("Upload Image & Hover to Detect Color")

    uploaded = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        img_np = np.array(img)

        st.image(img, caption="Move cursor over image")

        st.write("Click anywhere on the image to detect color:")

        # Streamlit cannot detect real mouse hover
        # so user enters X, Y manually
        x = st.number_input("Enter X pixel:", min_value=0, max_value=img_np.shape[1]-1)
        y = st.number_input("Enter Y pixel:", min_value=0, max_value=img_np.shape[0]-1)

        if st.button("Detect Color"):
            R, G, B = img_np[int(y), int(x)]
            color_name = get_closest_color_name(B, G, R)

            st.success(f"Detected Color: **{color_name}**")
            st.write(f"RGB = ({R}, {G}, {B})")

            if st.button("🔊 Speak"):
                mp3_path = speak(color_name)
                st.audio(mp3_path)
