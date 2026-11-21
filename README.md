# VisionGuard AI (`industrial-vision-inspector`)

> **Intelligent industrial defect detection** platform built with PatchCore (ResNet18), OpenVINO CPU acceleration, FastAPI, Streamlit, and SQLite inspection history.

## 1. Problem Statement
Automated visual quality inspection on manufacturing lines requires high defect recall, pixel-level localization, low-latency CPU inference, and traceable inspection logs. VisionGuard AI transforms an experimental PatchCore script into a production-ready visual inspection platform.

## 2. Project Overview
This repository extends [Shayan-hab/Industrial-Anomaly-Detection](https://github.com/Shayan-hab/Industrial-Anomaly-Detection) with calibrated scoring, OpenVINO CPU export/inference, severity classification, a FastAPI inspection service, a Streamlit dashboard, and automated tests.

### Upstream vs. Original Contributions
- **Inherited from `Industrial-Anomaly-Detection`:** ResNet18 intermediate feature extraction (`layer2` + `layer3`), PatchCore memory bank concept, and synthetic dataset fallback.
- **Original Additions:**
  - Fixed `configs/config.yaml` and `MVTecDataset` category-path handling.
  - Seeded **k-center greedy coreset subsampling** and **normal-score calibration** (`mean + 3 * std`).
  - **ONNX + OpenVINO IR export** and benchmarked **OpenVINO CPU inference**.
  - **Global-scale heatmap normalization**, contour bounding-box localization, and **defect severity classification** (`OK`, `MINOR`, `MAJOR`, `CRITICAL`).
  - **FastAPI inspection service** (`/health`, `/model/info`, `/inspect`, `/inspect/batch`, `/history`, `/results/{filename}`) with SQLite history.
  - **Streamlit interactive inspection dashboard**, **Dockerfile**, and **Pytest suite**.

## 3. Verified Experimental Results (CPU)
Measured on the synthetic MVTec-style `bottle` validation split (`10` test images: 5 normal, 5 defective) on a 12-core Linux CPU:
- **Training time:** `3.407 s`
- **Calibrated threshold:** `0.7815`
- **ROC-AUC / Accuracy / F1:** `1.0` / `1.0` / `1.0` *(on synthetic dataset; real MVTec AD performance depends on category & lighting)*
- **PyTorch CPU latency:** `56.05 ms / image`
- **OpenVINO CPU latency:** `51.02 ms / image`

## 4. Installation & Quickstart
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1. Generate/download dataset & train + export OpenVINO IR
PYTHONPATH=. python scripts/download_dataset.py
PYTHONPATH=. python scripts/train_and_export.py

# 2. Start FastAPI service
PYTHONPATH=. uvicorn api.main:app --host 0.0.0.0 --port 8000

# 3. Start Streamlit dashboard (in another terminal)
PYTHONPATH=. streamlit run dashboard/app.py
```

## 5. How to Run Tests
```bash
PYTHONPATH=. pytest tests/ -v
```

## 6. Docker Instructions
```bash
docker build -t visionguard-ai .
docker run -p 8000:8000 visionguard-ai
```

## 7. Original Repository Attribution & License
- **Original Upstream Repository:** [Shayan-hab/Industrial-Anomaly-Detection](https://github.com/Shayan-hab/Industrial-Anomaly-Detection) (no `LICENSE` file was present in upstream).
- MVTec AD dataset usage is subject to the [MVTec AD License](https://www.mvtec.com/company/research/datasets/mvtec-ad) when using real MVTec data.
