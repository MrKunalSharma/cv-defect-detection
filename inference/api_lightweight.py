"""
Lightweight FastAPI server for Render deployment
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import time
import os
from typing import List

# Initialize FastAPI app
app = FastAPI(
    title="Defect Detection API - Demo",
    description="Lightweight demo API for defect detection",
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

# Mock detection function (replace with real model later)
def mock_detect(image_bytes: bytes):
    """Mock detection for demo purposes"""
    # In production, this would use the actual model
    return [
        {
            "class": "demo_object",
            "confidence": 0.85,
            "bbox": {"x1": 100, "y1": 100, "x2": 200, "y2": 200}
        }
    ]

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Defect Detection API - Demo Mode",
        "status": "running",
        "note": "This is a lightweight demo. Full model requires local deployment.",
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
        "mode": "demo",
        "timestamp": time.time()
    }

@app.post("/detect")
async def detect_defects(
    file: UploadFile = File(...),
    confidence: float = 0.25,
    iou_threshold: float = 0.45
):
    """
    Detect objects/defects in uploaded image (Demo mode)
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        start_time = time.time()
        image_bytes = await file.read()
        
        # Get image info
        image = Image.open(io.BytesIO(image_bytes))
        width, height = image.size
        
        # Mock detections
        detections = mock_detect(image_bytes)
        
        processing_time = time.time() - start_time
        
        return JSONResponse(content={
            "status": "success",
            "model_type": "demo",
            "detections": detections,
            "total_detections": len(detections),
            "processing_time": round(processing_time, 4),
            "image_shape": [height, width, 3],
            "note": "Demo mode - using mock detections"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
