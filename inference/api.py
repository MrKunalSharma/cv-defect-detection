"""
FastAPI inference server for defect detection
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io
import time
from pathlib import Path
from typing import List, Dict, Any
import sys
sys.path.append('.')

from ultralytics import YOLO
import cv2
import torch

# Set PyTorch to trust the model files
import warnings
warnings.filterwarnings('ignore')

# Initialize FastAPI app
app = FastAPI(
    title="Defect Detection API",
    description="Real-time defect detection using YOLOv8",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# COCO class names (80 classes)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Defect class names (for custom trained models)
DEFECT_CLASSES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled_in_scale', 'scratches']

# Load model
MODEL_PATH = Path("models/yolov8n_pretrained.pt")
IS_DEFECT_MODEL = False  # Set to True when using custom trained model

try:
    if not MODEL_PATH.exists():
        print("Downloading YOLOv8n model...")
        model = YOLO('yolov8n.pt')
        if Path('yolov8n.pt').exists():
            import shutil
            MODEL_PATH.parent.mkdir(exist_ok=True)
            shutil.copy2('yolov8n.pt', MODEL_PATH)
            print(f"Model saved to {MODEL_PATH}")
    
    print("Loading model...")
    model = YOLO(str(MODEL_PATH))
    print("Model loaded successfully!")
    
    # Determine which classes to use
    if IS_DEFECT_MODEL:
        CLASS_NAMES = DEFECT_CLASSES
    else:
        CLASS_NAMES = COCO_CLASSES
    
except Exception as e:
    print(f"Error loading model: {e}")
    print("Using a dummy model for testing...")
    model = None
    CLASS_NAMES = DEFECT_CLASSES


def process_image(image_bytes: bytes) -> np.ndarray:
    """Convert uploaded bytes to numpy array"""
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert('RGB')
    return np.array(image)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Object/Defect Detection API",
        "model_type": "Defect Detection Model" if IS_DEFECT_MODEL else "General Object Detection (COCO)",
        "classes": len(CLASS_NAMES),
        "endpoints": {
            "/": "This message",
            "/health": "Health check",
            "/detect": "POST - Upload image for detection",
            "/docs": "API documentation"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": "defect" if IS_DEFECT_MODEL else "coco"
    }


@app.post("/detect")
async def detect_defects(
    file: UploadFile = File(...),
    confidence: float = 0.25,
    iou_threshold: float = 0.45
):
    """
    Detect objects/defects in uploaded image
    
    Args:
        file: Image file (JPEG, PNG)
        confidence: Confidence threshold (0-1)
        iou_threshold: IOU threshold for NMS (0-1)
    
    Returns:
        JSON with detection results
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    if model is None:
        return JSONResponse(content={
            "status": "error",
            "message": "Model not loaded"
        })
    
    try:
        # Read image
        start_time = time.time()
        image_bytes = await file.read()
        image = process_image(image_bytes)
        
        # Run inference
        results = model(image, conf=confidence, iou=iou_threshold)
        
        # Process results
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # Get class and confidence
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Get class name
                    class_name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"class_{cls}"
                    
                    detections.append({
                        "class": class_name,
                        "class_id": cls,
                        "confidence": round(conf, 4),
                        "bbox": {
                            "x1": int(x1),
                            "y1": int(y1),
                            "x2": int(x2),
                            "y2": int(y2)
                        }
                    })
        
        processing_time = time.time() - start_time
        
        return JSONResponse(content={
            "status": "success",
            "model_type": "defect" if IS_DEFECT_MODEL else "object_detection",
            "detections": detections,
            "total_detections": len(detections),
            "processing_time": round(processing_time, 4),
            "image_shape": image.shape
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
