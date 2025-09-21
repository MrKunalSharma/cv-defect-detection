"""
Download and test YOLOv8 model
"""
from ultralytics import YOLO
import shutil

print("Downloading YOLOv8n model...")
try:
    # This will download and load the model
    model = YOLO('yolov8n.pt')
    print("Model loaded successfully!")
    
    # Save to models directory
    if hasattr(model, 'ckpt_path') and model.ckpt_path:
        shutil.copy2(model.ckpt_path, 'models/yolov8n_pretrained.pt')
        print("Model saved to models/yolov8n_pretrained.pt")
    else:
        # Try alternative method
        model.save('models/yolov8n_pretrained.pt')
        print("Model exported to models/yolov8n_pretrained.pt")
    
    # Test inference
    print("\nTesting model...")
    results = model.predict('data/raw/train/images/scratches_0000.jpg')
    print(f"Test successful! Detected {len(results[0].boxes) if results[0].boxes else 0} objects")
    
except Exception as e:
    print(f"Error: {e}")
