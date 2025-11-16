import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Color Blind Assistant", layout="wide")
st.title("🎨 Color Blindness Assistant Tool")

st.write("This tool helps color-blind users identify colors accurately using:")
st.write("✔ Real-time webcam")
st.write("✔ Uploading images")
st.write("✔ Color-blind assistance messages")

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
# COLOR-BLIND ASSISTANCE
# -------------------------------------------
def assist_colorblind(r, g, b, colorblind_type):
    cname = get_color_name(r, g, b)
    
    if colorblind_type == "Protanopia (Red Weak)" and cname in ["Red", "Pink", "Orange"]:
        advise = f"This is {cname} (red may appear dull)"
    elif colorblind_type == "Deuteranopia (Green Weak)" and cname in ["Green", "Yellow"]:
        advise = f"This is {cname} (green may be hard to see)"
    elif colorblind_type == "Tritanopia (Blue Weak)" and cname in ["Blue", "Purple"]:
        advise = f"This is {cname} (blue may appear confusing)"
    else:
        advise = f"This is {cname}"
    
    return advise

# -------------------------------------------
# COLOR-BLIND FILTER (optional for enhancement)
# -------------------------------------------
def apply_colorblind_filter(img, mode):
    # Ensure RGB only (remove alpha channel if present)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    # Convert to float and normalize
    img = img.astype("float32") / 255.0

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

    # Apply matrix transformation
    filtered = img @ mat.T

    # Clip and convert back
    filtered = np.clip(filtered, 0, 1)
    return (filtered * 255).astype("uint8")

# -------------------------------------------
# IMAGE UPLOAD
# -------------------------------------------
st.header("📁 Upload Image for Color Detection")

uploaded = st.file_uploader("Upload JPG/PNG image", type=["jpg", "jpeg", "png"])
colorblind_mode = st.selectbox("Choose Color-Blind Type", 
                               ["None", "Protanopia (Red Weak)", "Deuteranopia (Green Weak)", "Tritanopia (Blue Weak)"])

if uploaded:
    img = Image.open(uploaded)
    img = np.array(img)

    # Apply color-blind filter (optional)
    filtered_img = apply_colorblind_filter(img, colorblind_mode)
    st.image(filtered_img, caption=f"Filtered View: {colorblind_mode}", use_column_width=True)

    # Detect center pixel color for assistance
    h, w = filtered_img.shape[:2]
    cx, cy = w//2, h//2
    b, g, r = filtered_img[cy, cx]
    advice = assist_colorblind(r, g, b, colorblind_mode)
    st.write("🔹", advice)

# -------------------------------------------
# REAL-TIME WEBCAM
# -------------------------------------------
st.header("📷 Real-Time Webcam Color Detection")

start_cam = st.checkbox("Start Camera")

if start_cam:
    webcam_frame = st.empty()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while start_cam:
        ret, frame = cap.read()
        if not ret:
            st.write("Camera not found.")
            break

        h, w = frame.shape[:2]
        cx, cy = w//2, h//2

        b, g, r = frame[cy, cx]
        advice = assist_colorblind(r, g, b, colorblind_mode)

        # Overlay circle and text
        cv2.circle(frame, (cx, cy), 10, (0, 0, 0), -1)
        cv2.putText(frame, advice, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        # Apply color-blind filter for enhancement (optional)
        filtered_frame = apply_colorblind_filter(frame, colorblind_mode)
        webcam_frame.image(filtered_frame, channels="BGR")

    cap.release()
