# License Plate Detection with YOLO11

Fine-tune YOLO11 to detect license plates in vehicle images using a Roboflow-annotated dataset.

## Pipeline

```
Roboflow Dataset      YOLO11 Fine-tuning       Deploy
(Plate images)    ->  (yolo11m, 30 epochs)  -> (Roboflow API)
```

## Results (test set)

| Metric | Value |
|--------|-------|
| mAP@50 | 0.973 |
| mAP@50-95 | 0.710 |
| Precision | 0.983 |
| Recall | 0.938 |

## Repository structure

```
plate_detection.ipynb     # Main notebook — training, validation, inference
src/plates_detection/     # Python package (detection utilities)
images/                   # Input images for local inference
models/                   # Downloaded YOLO base weights (yolo11m.pt)
datasets/                 # Roboflow annotated dataset (downloaded on first run)
runs/                     # Ultralytics training outputs (generated)
pyproject.toml            # Dependencies (managed with uv)
.env                      # API keys (not committed)
```

## Setup

Requires Python 3.10 and [uv](https://github.com/astral-sh/uv).

```bash
uv sync
```

Create a `.env` file in the project root:

```
ROBOFLOW_API_KEY=your_roboflow_key
WORKSPACE_NAME=your_workspace_name
PROJECT_NAME=your_project_name
```

- Roboflow key: https://app.roboflow.com/settings/api

## Running

Open `plate_detection.ipynb` and run cells top to bottom. The notebook will:

1. Download the annotated dataset from Roboflow
2. Fine-tune `yolo11m.pt` on the plate detection dataset
3. Evaluate on the held-out test split
4. Run inference with confidence threshold sweep

## Key dependencies

| Package | Purpose |
|---------|---------|
| `ultralytics<=8.3.40` | YOLO11 training and inference |
| `torch` + `torchvision` (CUDA 12.4) | Deep learning backend |
| `roboflow` / `inference` | Dataset download and deployment |
| `supervision` | Detection visualisation |
| `opencv-python` | Image processing |
