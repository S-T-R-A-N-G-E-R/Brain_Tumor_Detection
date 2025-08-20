import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import os
import torch
import pandas as pd

# --- Page config ---
st.set_page_config(
    page_title="NeuroScan AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CSS (adds glass look, fixes slider container, caps image size) ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

body, .stApp {
  font-family: 'Inter', sans-serif;
  background: radial-gradient(circle at 20% 20%, #0f0f1c, #0a0a0f);
  color: #fff;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* Reusable glass card */
.glass-card {
  background: rgba(255, 255, 255, 0.05);
  backdrop-filter: blur(20px);
  border-radius: 20px;
  padding: 2rem;
  border: 1px solid rgba(255, 255, 255, 0.08);
  box-shadow: 0 8px 30px rgba(0,0,0,0.4);
  transition: all 0.3s ease;
}
.glass-card:hover { transform: translateY(-4px); box-shadow: 0 16px 40px rgba(0,0,0,0.5); }

/* Hero */
.hero { text-align: center; padding: 4rem 2rem; }
.hero h1 {
  font-size: 3.6rem; font-weight: 700;
  background: linear-gradient(90deg, #7b61ff, #4bc0c8);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  margin-bottom: 0.75rem;
}
.hero p { color: rgba(255,255,255,0.75); font-size: 1.1rem; max-width: 720px; margin: 0 auto; }

/* Upload area card */
.upload-area {
  border: 2px dashed rgba(123, 97, 255, 0.5);
  border-radius: 16px; padding: 2.5rem 1.75rem; text-align: center;
  background: rgba(123, 97, 255, 0.05); transition: 0.3s;
}
.upload-area:hover { background: rgba(123, 97, 255, 0.1); border-color: rgba(123,97,255,0.9); }

/* Buttons */
.stButton > button {
  width: 100%; background: linear-gradient(135deg, #7b61ff, #4bc0c8);
  color: #fff; border: none; border-radius: 12px; padding: 0.9rem; font-weight: 600;
  transition: all 0.3s;
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 10px 25px rgba(123,97,255,0.3); }

/* Stats grid */
.stats-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(200px,1fr));
  gap: 1.25rem; margin: 1.5rem 0;
}
.stat { text-align: center; padding: 1.25rem; border-radius: 16px; background: rgba(255,255,255,0.07); }
.stat h2 {
  font-size: 1.8rem; margin: 0.2rem 0;
  background: linear-gradient(90deg,#7b61ff,#4bc0c8);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.stat p { color: rgba(255,255,255,0.75); margin: 0; }

/* Make images sane-sized and centered */
.stImage > img, .stImage img {
  max-width: 720px !important; width: 100% !important; height: auto !important;
  margin: 0 auto !important; display: block !important; border-radius: 16px;
  box-shadow: 0 8px 24px rgba(0,0,0,0.35);
}

/* Style the expander as a glass card (so slider sits inside) */
div[data-testid="stExpander"] {
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 16px;
  box-shadow: 0 8px 24px rgba(0,0,0,0.35);
}
div[data-testid="stExpander"] > details { padding: 0.5rem 0.75rem; }
div[data-testid="stExpander"] summary { font-weight: 600; }

/* Responsive tweaks */
@media (max-width: 768px) {
  .hero h1 { font-size: 2.4rem; }
}
</style>
""", unsafe_allow_html=True)

# --- Model loader (robust path resolution) ---
@st.cache_resource
def load_model():
    candidate_paths = [
        '/Users/heythere/Christ/Image_Video/CAC_Project/brain_tumor_model.pt',  # original path
        'brain_tumor_model.pt'  # fallback to local
    ]
    model_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            model_path = p
            break
    if model_path is None:
        st.error("Model not found. Checked:\n- " + "\n- ".join(candidate_paths))
        return None
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = YOLO(model_path)
    model.to(device)
    return model

def predict_image(image, model, conf):
    img_array = np.array(image)
    results = model.predict(img_array, conf=conf, iou=0.5, max_det=100, verbose=False)
    annotated = results[0].plot()
    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls)
        detections.append({
            "Class": model.names[cls_id],
            "Confidence": f"{float(box.conf):.2%}"
        })
    return annotated, detections

# --- UI ---
st.markdown(
    "<div class='hero'><h1>NeuroScan AI</h1>"
    "<p>Advanced brain tumor detection powered by YOLOv8. Upload an MRI scan for instant analysis.</p></div>",
    unsafe_allow_html=True
)

with st.container():
    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.markdown("<div class='glass-card'><div class='upload-area'>🧠<br><br><b>Upload MRI Image</b><br><small>Drag & drop JPG/PNG</small></div></div>", unsafe_allow_html=True)
        file = st.file_uploader(
            "Upload Image",  # non-empty label (for accessibility)
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"  # hides it from UI
        )


    with col2:
        # Use an EXPANDER to guarantee the slider lives inside a visible, bordered container
        with st.expander("⚙️ Settings", expanded=True):
            conf = st.slider("Confidence Threshold", 0.1, 0.9, 0.25, 0.05, help="Minimum confidence for detections")

# --- Processing & Results ---
if file:
    image = Image.open(file).convert("RGB")
    model = load_model()
    if model:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("📋 Original MRI Scan")
        st.image(image, caption="Uploaded MRI Image")  # width capped by CSS (max 720px)
        st.markdown("</div>", unsafe_allow_html=True)

        run = st.button("🔍 Run AI Detection")
        if run:
            with st.spinner("🧠 Analyzing brain scan..."):
                annotated, detections = predict_image(image, model, conf)

            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("🎯 Detection Results")

            # Stats
            avg_conf = np.mean([float(d['Confidence'][:-1])/100 for d in detections]) if detections else 0
            unique_classes = len(set(d['Class'] for d in detections)) if detections else 0

            st.markdown("<div class='stats-grid'>", unsafe_allow_html=True)
            st.markdown(f"<div class='stat'><h2>{len(detections)}</h2><p>Detections</p></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='stat'><h2>{avg_conf:.0%}</h2><p>Avg Confidence</p></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='stat'><h2>{unique_classes}</h2><p>Tumor Types</p></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Annotated image (width capped by CSS)
            st.image(annotated, caption="AI Detection (Annotated)")

            # Table or success note
            if detections:
                df = pd.DataFrame(detections)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.success("🎉 No tumors detected at the current threshold.")

            st.markdown("</div>", unsafe_allow_html=True)

# --- Footer ---
st.markdown(
    "<div style='text-align:center; color: rgba(255,255,255,0.5); padding: 1.5rem 0;'>"
    "🚀 NeuroScan AI | Powered by YOLOv8 & Streamlit<br>"
    "<small>⚠️ Research use only. Consult qualified medical professionals for diagnosis.</small>"
    "</div>",
    unsafe_allow_html=True
)
