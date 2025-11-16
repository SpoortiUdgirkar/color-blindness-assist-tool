import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Color Blind Assistant", layout="wide")
st.title("🎨 Color Blindness Assistant Tool")

st.write("This tool helps color-blind users identify colors accurately using:")
st.write("✔ Real-time webcam")
st.write("✔ Uploading images")
st.write("✔ Click detection")
st.write("✔ Color-blind safe filters")

# -------------------------------------------
# COLOR NAME DETECTOR
# -------------------------------------------
def get_color_name(r, g, b):
    colors = {
        "Red": (255, 0, 0),
        "Green": (0, 255, 0),
        "Blue": (0, 0, 255),
        "Yellow": (255, 255, 0),
        "Orange": (255, 165, 0),
        "Purple": (128, 0, 128),
        "Pink": (255, 105, 180),
        "Brown": (150, 75, 0),
        "Grey": (128, 128, 128),
        "White": (255, 255, 255),
        "Black": (0, 0, 0)
    }

    min_dist = float("inf")
    cname = "Unknown"

    for name, (cr, cg, cb) in colors.items():
        dist = (r-cr)**2 + (g-cg)**2 + (b-cb)**2
        if dist < min_dist:
            min_dist = dist
            cname = name

    return cname

# -------------------------------------------
# COLOR-BLIND SIMULATION MODES
# -------------------------------------------
def apply_colorblind_filter(img, mode):
    matrix = {
        "None": np.eye(3),
        "Protanopia (Red Weak)": np.array([[0.566, 0.433, 0.0],
                                           [0.558, 0.442, 0.0],
                                           [0.0,   0.242, 0.758]]),
        "Deuteranopia (Green Weak)": np.array([[0.625, 0.375, 0.0],
                                               [0.7,   0.3,   0.0],
                                               [0.0,   0.3,   0.7]]),
        "Tritanopia (Blue Weak)": np.array([[0.95, 0.05, 0.0],
                                            [0.0,  0.433, 0.567],
                                            [0.0,  0.475, 0.525]])
    }

    mat = matrix[mode]
    img = img.astype(float) / 255.0
    filtered = img.dot(mat.T)
    filtered = np.clip(filtered, 0, 1)
    return (filtered * 255).astype("uint8")

# -------------------------------------------
# IMAGE UPLOAD
# -------------------------------------------
st.header("📁 Upload Image for Color Detection")

uploaded = st.file_uploader("Upload JPG/PNG image", type=["jpg", "jpeg", "png"])
colorblind_mode = st.selectbox("Choose Color-Blind Mode", 
                               ["None", "Protanopia (Red Weak)", "Deuteranopia (Green Weak)", "Tritanopia (Blue Weak)"])

if uploaded:
    img = Image.open(uploaded)
    img = np.array(img)

    filtered_img = apply_colorblind_filter(img, colorblind_mode)
    st.image(filtered_img, caption=f"Filtered View: {colorblind_mode}", use_column_width=True)

    st.write("Click on the image to detect color (using Streamlit image coordinate tool coming soon).")

# -------------------------------------------
# REAL-TIME WEBCAM
# -------------------------------------------
st.header("📷 Real-Time Webcam Color Detection")

start_cam = st.checkbox("Start Camera")

if start_cam:
    webcam_frame = st.empty()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("Camera not found.")
            break

        h, w = frame.shape[:2]
        cx, cy = w//2, h//2

        b, g, r = frame[cy, cx]
        cname = get_color_name(r, g, b)

        cv2.circle(frame, (cx, cy), 5, (0, 0, 0), -1)
        cv2.putText(frame, f"{cname}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        filtered_frame = apply_colorblind_filter(frame, colorblind_mode)
        webcam_frame.image(filtered_frame, channels="BGR")

        if not start_cam:
            break

    cap.release()

