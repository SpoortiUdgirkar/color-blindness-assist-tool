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
        advice = f"This is {cname} (red may appear dull)"
    elif colorblind_type == "Deuteranopia (Green Weak)" and cname in ["Green", "Yellow"]:
        advice = f"This is {cname} (green may be hard to see)"
    elif colorblind_type == "Tritanopia (Blue Weak)" and cname in ["Blue", "Purple"]:
        advice = f"This is {cname} (blue may appear confusing)"
    else:
        advice = f"This is {cname}"
    
    return advice

# -------------------------------------------
# IMAGE UPLOAD
# -------------------------------------------
st.header("📁 Upload Image for Color Detection")

uploaded = st.file_uploader("Upload JPG/PNG image", type=["jpg", "jpeg", "png"])
colorblind_type = st.selectbox("Choose Your Color-Blind Type", 
                               ["None", "Protanopia (Red Weak)", "Deuteranopia (Green Weak)", "Tritanopia (Blue Weak)"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")  # ensure 3 channels
    img = np.array(img)

    # Show original image (do not simulate color-blind)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Detect center pixel color for assistance
    h, w = img.shape[:2]
    cx, cy = w//2, h//2
    b, g, r = img[cy, cx]
    advice = assist_colorblind(r, g, b, colorblind_type)
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
        advice = assist_colorblind(r, g, b, colorblind_type)

        # Overlay circle and text
        cv2.circle(frame, (cx, cy), 10, (0, 0, 0), -1)
        cv2.putText(frame, advice, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        # Show real webcam feed
        webcam_frame.image(frame, channels="BGR")

    cap.release()
