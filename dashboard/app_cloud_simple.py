"""
Streamlit Dashboard for Defect Detection - Cloud Version (No OpenCV)
"""
import streamlit as st
import requests
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import time
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Defect Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API endpoint
API_URL = st.secrets.get("API_URL", "https://cv-defect-api.onrender.com")

# Initialize session state
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []

# Title
st.title("🔍 Real-Time Object/Defect Detection System")
st.markdown("### Computer Vision-based Detection using YOLOv8")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    iou_threshold = st.slider("IOU Threshold", 0.0, 1.0, 0.45, 0.05)
    
    if st.button("Check API Health"):
        try:
            response = requests.get(f"{API_URL}/health")
            if response.status_code == 200:
                st.success("✅ API is healthy")
                st.json(response.json())
        except Exception as e:
            st.error(f"❌ API Error: {e}")
    
    st.metric("Total Detections", len(st.session_state.detection_history))
    
    if st.button("Clear History"):
        st.session_state.detection_history = []
        st.rerun()

# Main content
col1, col2 = st.columns(2)

with col1:
    st.header("📤 Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_column_width=True)
        
        if st.button("🔍 Detect Objects", type="primary"):
            with st.spinner("Processing..."):
                try:
                    # Prepare image
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format='JPEG')
                    img_byte_arr.seek(0)
                    
                    # API request
                    files = {"file": ("image.jpg", img_byte_arr, "image/jpeg")}
                    params = {"confidence": confidence, "iou_threshold": iou_threshold}
                    
                    response = requests.post(f"{API_URL}/detect", files=files, params=params)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Store in history
                        st.session_state.detection_history.append({
                            "timestamp": datetime.now(),
                            "result": result,
                            "image": image
                        })
                        
                        # Display results
                        with col2:
                            st.header("🎯 Detection Results")
                            
                            if result.get('model_type') == 'demo':
                                st.warning("⚠️ Demo mode - showing sample detections")
                            
                            # Metrics
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Detections", result.get('total_detections', 0))
                            c2.metric("Time", f"{result.get('processing_time', 0):.3f}s")
                            c3.metric("Model", result.get('model_type', 'unknown'))
                            
                            # Show detections
                            if result.get('detections'):
                                # Create image with boxes using Pillow
                                img_with_boxes = image.copy()
                                draw = ImageDraw.Draw(img_with_boxes)
                                
                                for det in result['detections']:
                                    bbox = det['bbox']
                                    # Draw rectangle
                                    draw.rectangle(
                                        [(bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2'])],
                                        outline="green",
                                        width=3
                                    )
                                    # Add label
                                    label = f"{det['class']} ({det['confidence']:.2f})"
                                    draw.text((bbox['x1'], bbox['y1']-20), label, fill="green")
                                
                                st.image(img_with_boxes, caption="Detected Objects", use_column_width=True)
                                
                                # List detections
                                for i, det in enumerate(result['detections']):
                                    st.write(f"**Object {i+1}:** {det['class']} ({det['confidence']:.2%})")
                            else:
                                st.info("No objects detected")
                    else:
                        st.error(f"API Error: {response.status_code}")
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")

with col2:
    if not uploaded_file:
        st.header("🎯 Detection Results")
        st.info("Upload an image to start detection")

# History
st.header("📜 Recent Detections")
if st.session_state.detection_history:
    for item in reversed(st.session_state.detection_history[-3:]):
        with st.expander(f"{item['timestamp'].strftime('%H:%M:%S')} - {item['result'].get('total_detections', 0)} objects"):
            c1, c2 = st.columns([1, 2])
            with c1:
                st.image(item['image'], width=150)
            with c2:
                if item['result'].get('detections'):
                    for det in item['result']['detections']:
                        st.write(f"- {det['class']} ({det['confidence']:.1%})")
                else:
                    st.write("No detections")

st.markdown("---")
st.markdown("🚀 [GitHub](https://github.com/MrKunalSharma/cv-defect-detection) | Powered by YOLOv8")
