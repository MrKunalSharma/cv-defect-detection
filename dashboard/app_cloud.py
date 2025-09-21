"""
Streamlit Dashboard for Defect Detection - Cloud Version
"""
import streamlit as st
import requests
import numpy as np
from PIL import Image
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

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    div[data-testid="stImage"] {
        text-align: center;
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# API endpoint - use secrets if available, otherwise use deployed URL
try:
    API_URL = st.secrets["API_URL"]
except:
    API_URL = "https://cv-defect-api.onrender.com"

# Initialize session state
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []

# Title and description
st.title("🔍 Real-Time Object/Defect Detection System")
st.markdown("### Computer Vision-based Detection using YOLOv8")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Detection settings
    st.subheader("Detection Parameters")
    confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    iou_threshold = st.slider("IOU Threshold", 0.0, 1.0, 0.45, 0.05)
    
    # API Health Check
    st.subheader("🔌 System Status")
    if st.button("Check API Health"):
        try:
            response = requests.get(f"{API_URL}/health")
            if response.status_code == 200:
                health = response.json()
                if health['status'] == 'healthy':
                    st.success("✅ API is healthy")
                    model_type = health.get('model_type', 'unknown')
                    st.info(f"Model type: {model_type}")
                    st.info(f"API URL: {API_URL}")
            else:
                st.error("❌ API is not responding")
        except Exception as e:
            st.error(f"❌ Cannot connect to API: {str(e)}")
    
    # Statistics
    st.subheader("📊 Statistics")
    st.metric("Total Detections", len(st.session_state.detection_history))
    
    # Clear history
    if st.button("🗑️ Clear History"):
        st.session_state.detection_history = []
        st.rerun()

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload Image")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload an image to detect objects/defects"
    )
    
    # Camera input
    camera_photo = st.camera_input("Or take a photo")
    
    # Use camera photo if available, otherwise use uploaded file
    image_to_process = camera_photo if camera_photo else uploaded_file
    
    if image_to_process:
        # Display original image
        st.subheader("Original Image")
        image = Image.open(image_to_process)
        st.image(image, use_column_width=True)
        
        # Process button
        if st.button("🔍 Detect Objects", type="primary"):
            with st.spinner("Processing..."):
                try:
                    # Prepare image for API
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format='JPEG')
                    img_byte_arr.seek(0)
                    
                    # Send to API
                    files = {"file": ("image.jpg", img_byte_arr, "image/jpeg")}
                    params = {
                        "confidence": confidence,
                        "iou_threshold": iou_threshold
                    }
                    
                    response = requests.post(
                        f"{API_URL}/detect",
                        files=files,
                        params=params
                    )
                    
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
                            
                            # Check if demo mode
                            if result.get('model_type') == 'demo':
                                st.warning("⚠️ Running in demo mode - showing sample detections")
                            
                            # Determine what to show based on response
                            total_detections = result.get('total_detections', 0)
                            model_type = result.get('model_type', 'unknown')
                            
                            # Metrics
                            col2_1, col2_2, col2_3 = st.columns(3)
                            with col2_1:
                                st.metric("Total Detections", total_detections)
                            with col2_2:
                                st.metric("Processing Time", f"{result['processing_time']:.3f}s")
                            with col2_3:
                                st.metric("Model Type", model_type.replace('_', ' ').title())
                            
                            # Detection details
                            if result['detections']:
                                st.subheader("Detected Objects:")
                                
                                # Draw bboxes on image
                                img_array = np.array(image)
                                for detection in result['detections']:
                                    bbox = detection['bbox']
                                    # Draw rectangle
                                    # Rectangle drawing removed for cloud deployment,
                                        (bbox['x2'], bbox['y2']),
                                        (0, 255, 0), 2
                                    )
                                    # Add label
                                    label = f"{detection['class']} ({detection['confidence']:.2f})"
                                    # Text drawing removed for cloud deployment,
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, (0, 255, 0), 2
                                    )
                                
                                # Display annotated image
                                st.image(img_array, use_column_width=True, caption="Annotated Image")
                                
                                # List detections
                                for i, detection in enumerate(result['detections']):
                                    st.write(f"**Detection {i+1}:**")
                                    st.write(f"- Type: {detection['class']}")
                                    st.write(f"- Confidence: {detection['confidence']:.2%}")
                                    st.write("---")
                            else:
                                st.info("No objects detected in the image.")
                                
                    else:
                        st.error(f"Error: {response.status_code}")
                        st.error(response.text)
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.info("Make sure the API is accessible")

with col2:
    if not image_to_process:
        st.header("🎯 Detection Results")
        st.info("Upload an image or take a photo to start detection")

# History section
st.header("📜 Detection History")
if st.session_state.detection_history:
    # Show latest 5 detections
    for item in reversed(st.session_state.detection_history[-5:]):
        with st.expander(f"Detection at {item['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"):
            col_hist1, col_hist2 = st.columns([1, 2])
            with col_hist1:
                st.image(item['image'], width=200)
            with col_hist2:
                result = item['result']
                total_detections = result.get('total_detections', 0)
                st.write(f"**Total Detections:** {total_detections}")
                st.write(f"**Processing Time:** {result['processing_time']:.3f}s")
                if result['detections']:
                    for detection in result['detections']:
                        st.write(f"- {detection['class']} ({detection['confidence']:.2%})")
else:
    st.info("No detections yet. Upload an image to start!")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and YOLOv8 | [GitHub](https://github.com/MrKunalSharma/cv-defect-detection)")

