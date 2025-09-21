"""
YOLOv8 Training Script for Defect Detection
"""
import os
from pathlib import Path
from datetime import datetime

import yaml
from ultralytics import YOLO
from utils.config import config


def train_model():
    """Train YOLOv8 model for defect detection"""
    
    # Load configuration
    model_config = config.model
    train_config = config.training
    
    print("=" * 50)
    print("Starting YOLOv8 Training")
    print("=" * 50)
    
    # Initialize model
    model = YOLO(f"{model_config['architecture']}.pt")
    print(f"Loaded {model_config['architecture']} model")
    
    # Training arguments
    train_args = {
        'data': 'data/raw/data.yaml',
        'epochs': train_config['epochs'],
        'imgsz': model_config['input_size'],
        'batch': train_config['batch_size'],
        'device': train_config['device'],
        'workers': 4,
        'project': 'runs/train',
        'name': f"defect_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'exist_ok': True,
        'pretrained': model_config['pretrained'],
        'optimizer': 'Adam',
        'lr0': train_config['learning_rate'],
        'patience': 10,
        'save': True,
        'save_period': -1,  # Save only best and last
        'cache': False,
        'verbose': True,
        'conf': model_config['confidence_threshold'],
        'iou': model_config['iou_threshold'],
    }
    
    print(f"\nTraining configuration:")
    for key, value in train_args.items():
        print(f"  {key}: {value}")
    
    # Start training
    print("\nStarting training...")
    results = model.train(**train_args)
    
    # Save the best model to models directory
    best_model_path = Path(results.save_dir) / 'weights' / 'best.pt'
    if best_model_path.exists():
        save_path = Path('models') / 'best_model.pt'
        best_model_path.rename(save_path)
        print(f"\nBest model saved to: {save_path}")
    
    print("\nTraining completed!")
    print(f"Results saved to: {results.save_dir}")
    
    # Validate the model
    print("\nRunning validation...")
    metrics = model.val()
    
    print("\nValidation metrics:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    
    return model, results


if __name__ == "__main__":
    train_model()
