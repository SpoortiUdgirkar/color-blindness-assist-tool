import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Color Blind Assistant", layout="wide")
st.title("🎨 Color Blindness Assistant Tool")
st.write("This tool helps color-blind users identify colors accurately using:")
st.write("✔ Side-by-side comparison (simulated vs corrected)")
st.write("✔ Uploaded images")
st.write("✔ Real-time webcam")

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
# COLOR-BLIND FILTER (simulation)
# -------------------------------------------
def apply_colorblind_filter(img, mode):
    if img.shape[2] == 4:
        img = img[:, :, :3]
    img = img.astype("float32") / 255.0
    matrix = {
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
    mat = matrix.get(mode, np.eye(3))
    filtered = img @ mat.T
    filtered = np.clip(filtered, 0, 1)
    return (filtered * 255).astype("uint8")

# -------------------------------------------
# COLOR-BLIND TYPE SELECTION
# -------------------------------------------
colorblind_type = st.selectbox("Choose Your Color-Blind Type", 
                               ["None", "Protanopia (Red Weak)", "Deuteranopia (Green Weak)", "Tritanopia (Blue Weak)"])

# -------------------------------------------
# IMAGE UPLOAD
# -------------------------------------------
st.header("📁 Upload Image for Comparison")

uploaded = st.file_uploader("Upload JPG/PNG image", type=["jpg", "jpeg", "png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    img = np.array(img)

    # Generate simulated color-blind image
    if colorblind_type != "None":
        simulated_img = apply_colorblind_filter(img, colorblind_type)
    else:
        simulated_img = img.copy()

    # Detect center pixel
    h, w = img.shape[:2]
    cx, cy = w//2, h//2
    b, g, r = img[cy, cx]
    advice = assist_colorblind(r, g, b, colorblind_type)

    # Display side-by-side
    col1, col2 = st.columns(2)
    with col1:
        st.image(simulated_img, caption=f"Simulated {colorblind_type}")
    with col2:
        st.image(img, caption="Original / Corrected Colors")
        st.write("🔹 Center Pixel Advice:", advice)

# -------------------------------------------
# REAL-TIME WEBCAM
# -------------------------------------------
st.header("📷 Real-Time Webcam Comparison")

start_cam = st.checkbox("Start Camera")
if start_cam:
    webcam_frame = st.empty()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while start_cam:
        ret, frame = cap.read()
        if not ret:
            st.write("Camera not found.")
            break

        # Convert to RGB for PIL/Streamlit compatibility
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Simulated image
        if colorblind_type != "None":
            simulated_frame = apply_colorblind_filter(frame_rgb, colorblind_type)
        else:
            simulated_frame = frame_rgb.copy()

        # Detect center pixel
        h, w = frame_rgb.shape[:2]
        cx, cy = w//2, h//2
        b, g, r = frame_rgb[cy, cx]
        advice = assist_colorblind(r, g, b, colorblind_type)

        # Draw circle on original frame
        cv2.circle(frame_rgb, (cx, cy), 10, (0, 0, 0), -1)
        cv2.putText(frame_rgb, advice, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show side-by-side in Streamlit
        col1, col2 = st.columns(2)
        with col1:
            col1.image(simulated_frame, caption=f"Simulated {colorblind_type}")
        with col2:
            col2.image(frame_rgb, caption="Original / Corrected Colors")
            col2.write("🔹 Center Pixel Advice:", advice)

    cap.release()
