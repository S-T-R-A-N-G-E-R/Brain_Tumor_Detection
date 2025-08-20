import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
import torch
import pandas as pd

# Set page config for a modern look
st.set_page_config(
    page_title="NeuroScan AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for ultra-modern, glassmorphism UI
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
        color: #ffffff;
        font-family: 'Inter', sans-serif;
        min-height: 100vh;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Main Container */
    .main-container {
        max-width: 1400px;
        margin: 0 auto;
        padding: 2rem;
    }
    
    /* Hero Section */
    .hero-section {
        text-align: center;
        margin-bottom: 3rem;
        padding: 3rem 2rem;
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        letter-spacing: -0.02em;
    }
    
    .hero-subtitle {
        font-size: 1.25rem;
        color: rgba(255, 255, 255, 0.7);
        font-weight: 400;
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.6;
    }
    
    /* Control Panel */
    .control-panel {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    }
    
    .control-title {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        color: #ffffff;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Upload Area */
    .upload-area {
        border: 2px dashed rgba(102, 126, 234, 0.5);
        border-radius: 16px;
        padding: 3rem 2rem;
        text-align: center;
        background: rgba(102, 126, 234, 0.05);
        transition: all 0.3s ease;
        margin-bottom: 2rem;
    }
    
    .upload-area:hover {
        border-color: rgba(102, 126, 234, 0.8);
        background: rgba(102, 126, 234, 0.1);
        transform: translateY(-2px);
    }
    
    .upload-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        color: #667eea;
    }
    
    /* Buttons */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 1rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    .stSlider > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Results Section */
    .results-container {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2rem;
        margin-top: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    }
    
    .results-title {
        font-size: 1.75rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        color: #ffffff;
        text-align: center;
    }
    
    /* Image Display */
    .stImage > img {
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease;
    }
    
    .stImage > img:hover {
        transform: scale(1.02);
    }
    
    /* Results Layout */
    .results-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 2rem;
        margin-top: 2rem;
    }
    
    .result-section {
        background: rgba(255, 255, 255, 0.06);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .result-section h4 {
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        color: #ffffff;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .no-detection {
        text-align: center;
        padding: 3rem 2rem;
        background: rgba(76, 175, 80, 0.1);
        border-radius: 16px;
        border: 1px solid rgba(76, 175, 80, 0.2);
    }
    
    .no-detection h4 {
        color: #4CAF50;
        margin-bottom: 1rem;
        font-size: 1.5rem;
    }
    
    .no-detection p {
        color: rgba(255, 255, 255, 0.8);
        margin-bottom: 0.5rem;
        line-height: 1.5;
    }
    
    .no-detection small {
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.9rem;
    }
    
    /* Stats Cards */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .stat-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 2rem 1.5rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.15);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .stat-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .stat-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.2);
        background: rgba(255, 255, 255, 0.12);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        display: block;
    }
    
    .stat-label {
        font-size: 1rem;
        color: rgba(255, 255, 255, 0.8);
        font-weight: 500;
        letter-spacing: 0.5px;
    }
    
    .stat-icon {
        font-size: 1.5rem;
        margin-bottom: 1rem;
        opacity: 0.8;
    }
    
    /* Model Info */
    .model-info {
        background: rgba(102, 126, 234, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(102, 126, 234, 0.2);
        margin-top: 1rem;
    }
    
    .model-info h4 {
        color: #667eea;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    .model-detail {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }
    
    .model-detail span:first-child {
        color: rgba(255, 255, 255, 0.7);
    }
    
    .model-detail span:last-child {
        color: #ffffff;
        font-weight: 500;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: rgba(255, 255, 255, 0.5);
        font-size: 0.9rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        margin-top: 3rem;
    }
    
    /* File Uploader */
    .stFileUploader {
        margin: 0;
    }
    
    .stFileUploader > label {
        display: none;
    }
    
    /* Success/Warning Messages */
    .stSuccess, .stWarning, .stError {
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2.5rem;
        }
        
        .main-container {
            padding: 1rem;
        }
        
        .hero-section {
            padding: 2rem 1rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Initialize model
@st.cache_resource
def load_model():
    model_path = '/Users/heythere/Christ/Image_Video/CAC_Project/brain_tumor_model.pt'
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}")
        return None
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = YOLO(model_path)
    model.to(device)
    return model

# Function to process image and get predictions
def predict_image(image, model, conf_threshold):
    img_array = np.array(image)
    results = model.predict(img_array, conf=conf_threshold, iou=0.5, max_det=100, verbose=False)
    annotated_img = results[0].plot()
    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls)
        cls_name = model.names[cls_id]
        conf = float(box.conf)
        detections.append({
            'Class': cls_name,
            'Confidence': f"{conf:.2%}",
            'Bounding Box': [float(coord) for coord in box.xyxy[0]]
        })
    return annotated_img, detections

# Create results directory
results_dir = "prediction_results"
os.makedirs(results_dir, exist_ok=True)

# Main Layout
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Hero Section
st.markdown("""
    <div class="hero-section">
        <div class="hero-title">NeuroScan AI</div>
        <div class="hero-subtitle">
            Advanced brain tumor detection powered by YOLOv8 deep learning technology.
            Upload your MRI scan to get instant, accurate tumor detection results.
        </div>
    </div>
""", unsafe_allow_html=True)

# Control Panel
st.markdown('<div class="control-panel">', unsafe_allow_html=True)
st.markdown('<div class="control-title">🎛️ Detection Controls</div>', unsafe_allow_html=True)

# Create columns for controls
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
        <div class="upload-area">
            <div class="upload-icon">🧠</div>
            <h3>Upload MRI Image</h3>
            <p>Drag and drop your JPEG or PNG file here</p>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"], label_visibility="hidden")

with col2:
    st.markdown("### ⚙️ Settings")
    conf_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.25, 0.05, 
                              help="Minimum confidence for detections")
    
    # Model info card
    st.markdown("""
        <div class="model-info">
            <h4>🤖 Model Information</h4>
            <div class="model-detail">
                <span>Architecture:</span>
                <span>YOLOv8 Nano</span>
            </div>
            <div class="model-detail">
                <span>Classes:</span>
                <span>Glioma, Meningioma, Pituitary</span>
            </div>
            <div class="model-detail">
                <span>Dataset:</span>
                <span>Kaggle Brain Tumor</span>
            </div>
            <div class="model-detail">
                <span>Device:</span>
                <span>MacBook Air M2 Optimized</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Main Processing Section
if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    
    # Load model
    model = load_model()
    if model is None:
        st.stop()
    
    # Display original image in a nice container
    st.markdown('<div class="results-container">', unsafe_allow_html=True)
    st.markdown('<div class="results-title">📋 Original MRI Scan</div>', unsafe_allow_html=True)
    
    # Center the image
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption="Uploaded MRI Image", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Detection button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔍 Run AI Detection"):
            with st.spinner("🧠 Analyzing brain scan..."):
                # Get predictions
                annotated_img, detections = predict_image(image, model, conf_threshold)
                
                # Results section
                st.markdown('<div class="results-container">', unsafe_allow_html=True)
                st.markdown('<div class="results-title">🎯 Detection Results</div>', unsafe_allow_html=True)
                
                # Stats cards with better layout
                st.markdown('<div class="stats-grid">', unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                        <div class="stat-card">
                            <div class="stat-icon">🎯</div>
                            <div class="stat-number">{len(detections)}</div>
                            <div class="stat-label">Detections Found</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    avg_conf = np.mean([float(d['Confidence'].rstrip('%'))/100 for d in detections]) if detections else 0
                    st.markdown(f"""
                        <div class="stat-card">
                            <div class="stat-icon">📊</div>
                            <div class="stat-number">{avg_conf:.0%}</div>
                            <div class="stat-label">Average Confidence</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    unique_classes = len(set(d['Class'] for d in detections)) if detections else 0
                    st.markdown(f"""
                        <div class="stat-card">
                            <div class="stat-icon">🧬</div>
                            <div class="stat-number">{unique_classes}</div>
                            <div class="stat-label">Tumor Types</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    status = "🟢" if detections else "🟡"
                    status_text = "Positive" if detections else "Negative"
                    st.markdown(f"""
                        <div class="stat-card">
                            <div class="stat-icon">{status}</div>
                            <div class="stat-number">{status_text}</div>
                            <div class="stat-label">Detection Status</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Results display with improved layout
                st.markdown('<div class="results-grid">', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("""
                        <div class="result-section">
                            <h4>🖼️ Annotated Scan</h4>
                        </div>
                    """, unsafe_allow_html=True)
                    st.image(annotated_img, caption="AI Detection Results", use_container_width=True)
                
                with col2:
                    if detections:
                        st.markdown("""
                            <div class="result-section">
                                <h4>📋 Detection Summary</h4>
                            </div>
                        """, unsafe_allow_html=True)
                        df = pd.DataFrame(detections)
                        df_display = df[['Class', 'Confidence']].copy()
                        st.dataframe(df_display, use_container_width=True, hide_index=True)
                    else:
                        st.markdown("""
                            <div class="result-section">
                                <div class="no-detection">
                                    <h4>🎉 Scan Analysis Complete</h4>
                                    <p><strong>No tumors detected!</strong></p>
                                    <p>The AI analysis shows no signs of tumors with the current confidence threshold.</p>
                                    <small>Consider adjusting the confidence threshold if you suspect potential findings.</small>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="footer">
        <p>🚀 NeuroScan AI | Powered by YOLOv8 & Streamlit | Built for Medical Professionals</p>
        <p><small>⚠️ For research purposes only. Always consult qualified medical professionals for diagnosis.</small></p>
    </div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)