# Computer Vision Defect Detection System

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-latest-green.svg)](https://github.com/ultralytics/ultralytics)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red.svg)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-teal.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A production-ready, real-time object/defect detection system using Ultralytics YOLOv8 for inference, FastAPI for the backend API, and Streamlit for an interactive dashboard.

## 🌟 Live Demo

- Dashboard: [Streamlit Cloud](https://cv-defect-detection-m3zyjth8vjvwdtpnuoeskn.streamlit.app/)
- API Docs: [Render OpenAPI](https://cv-defect-api.onrender.com/docs)

## 🚀 Features

- Real-time detection with YOLOv8 (COCO by default, custom defects supported)
- REST API with automatic Swagger docs and CORS support
- Streamlit dashboard: upload or camera input, annotated results, history
- Config-driven training and serving via `configs/config.yaml`
- Saves best model to `models/best_model.pt` and reports mAP metrics
- Dockerfiles for local/containerized runs; example cloud config provided

## 🏗️ Architecture

```text
Data Source (Upload/Camera)
        │
        ▼
   FastAPI (inference)
        │ JSON detections
        ▼
 Streamlit Dashboard ──► Visualization (annotated image, metrics)
```

## 📁 Project Structure

```text
cv-defect-detection/
  configs/
    config.yaml              # Project, data, model, training, API settings
  dashboard/
    app.py                   # Streamlit UI (local)
    app_cloud.py             # Streamlit UI (cloud)
    app_cloud_simple.py      # Simple cloud variant
  data/
    raw/                     # YOLO-format sample dataset (train/val)
      data.yaml
  inference/
    api.py                   # FastAPI server (image upload → detections)
    api_render.py            # Cloud variant
    api_lightweight.py       # Lightweight demo variant
  models/                    # Pretrained and trained weights
  training/
    train.py                 # YOLOv8 training entrypoint
  utils/
    config.py                # Config loader
  tests/
    test_api.py              # API test (if used)
  test_model.py              # Download & quick inference test
  Dockerfile                 # App container (if used)
  Dockerfile.api             # API-only container
  Dockerfile.render          # Render deployment container
  render.yaml                # Example Render deployment
  requirements*.txt          # Dependencies (base/API/Streamlit)
```

## 🛠️ Tech Stack

- Computer Vision: YOLOv8 (Ultralytics), PyTorch
- Backend: FastAPI, Uvicorn
- Frontend: Streamlit
- Imaging: OpenCV, Pillow
- Packaging: Docker

## 🔧 Installation

### Prerequisites

- Python 3.9+
- Git
- (Optional) NVIDIA GPU + CUDA for faster training/inference

### Setup

```bash
git clone https://github.com/MrKunalSharma/cv-defect-detection.git
cd cv-defect-detection
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

Quick sanity check (downloads YOLOv8n and tests an image):

```bash
python test_model.py
```

## 🚀 Usage

### Start the API

```bash
uvicorn inference.api:app --host 0.0.0.0 --port 8000 --reload
```

### Launch the Dashboard (separate terminal)

```bash
streamlit run dashboard/app.py
```

Then open:
- Dashboard: `http://localhost:8501`
- API Docs: `http://localhost:8000/docs`

### Using the Dashboard

- Upload an image or capture via camera (local)
- Adjust Confidence and IoU in the sidebar
- Click “Detect Objects” to run inference and view annotated results
- See detection history at the bottom

## 🌐 API Endpoints

- `GET /` – API info and available endpoints
- `GET /health` – Health status and model info
- `POST /detect` – Multipart image upload → JSON detections

Example request:

```bash
curl -X POST "http://localhost:8000/detect?confidence=0.25&iou_threshold=0.45" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/raw/val/images/scratches_0000.jpg"
```

Sample response:

```json
{
  "status": "success",
  "model_type": "object_detection",
  "detections": [
    {
      "class": "person",
      "class_id": 0,
      "confidence": 0.92,
      "bbox": { "x1": 120, "y1": 80, "x2": 380, "y2": 400 }
    }
  ],
  "total_detections": 1,
  "processing_time": 0.015,
  "image_shape": [720, 1280, 3]
}
```

## 🧠 Training

Train a model using `configs/config.yaml`:

```bash
python training/train.py
```

Behavior:
- Loads `architecture` (e.g., `yolov8n`) and dataset from `data/raw/data.yaml`
- Logs under `runs/train/defect_detection_*`
- Saves best weights to `models/best_model.pt`
- Prints validation metrics (mAP50, mAP50-95)

GPU tips:
- Set `training.device` to `0` in `configs/config.yaml` for CUDA
- Tweak `epochs`, `batch_size`, `input_size` to fit your hardware

## 🔁 Switch to Custom Defect Model

After training, point the API to your model and classes:

1. Move/rename best weights to `models/best_model.pt`
2. Edit `inference/api.py`:
   - `IS_DEFECT_MODEL = True`
   - `MODEL_PATH = Path("models/best_model.pt")`
3. Restart the API

The API will return six defect classes: `crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled_in_scale`, `scratches`.

## 🐳 Docker

API-only container:

```bash
docker build -f Dockerfile.api -t defect-api .
docker run -p 8000:8000 defect-api
```

Full app container (if using combined Dockerfile):

```bash
docker build -t defect-app .
docker run -p 8000:8000 -p 8501:8501 defect-app
```

## ⚙️ Configuration

Edit `configs/config.yaml` to customize:
- Project and data paths
- Model parameters (architecture, input size, thresholds)
- Training (epochs, batch size, learning rate, device)
- API settings

## 🧪 Tests

Run tests if present:

```bash
pytest tests/
```

## 📄 License

This project is licensed under the MIT License – see `LICENSE` for details.

## 🙏 Acknowledgments

- Ultralytics YOLOv8
- FastAPI
- Streamlit
- COCO dataset (for pretrained weights)

## 📬 Contact

Kunal Sharma – `kunalsharma13579kunals@gmail.com`

Project: `https://github.com/MrKunalSharma/cv-defect-detection`

If this project helps you, please ⭐ the repo!
