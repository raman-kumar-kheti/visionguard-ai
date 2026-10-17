"""FastAPI visual quality inspection service for VisionGuard AI."""

from __future__ import annotations

import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
from torchvision import transforms

from src.db.history import InspectionHistoryDB
from src.models.patchcore import PatchCore
from src.pipeline.visualizer import Visualizer, classify_severity

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output"
RESULTS_DIR = OUTPUT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

cfg = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))
IMAGE_SIZE = int(cfg["dataset"]["image_size"])
DEFAULT_THRESHOLD = float(cfg["inference"]["threshold"])

transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

detector = PatchCore(
    backbone_name=cfg["model"]["backbone"],
    pca_components=int(cfg["model"]["pca_components"]),
    coreset_ratio=float(cfg["model"]["coreset_sampling_ratio"]),
    model_version=str(cfg["model"].get("version", "1.0.0")),
)
visualizer = Visualizer()
history_db = InspectionHistoryDB(OUTPUT_DIR / "inspection_history.db")
active_backend = "pytorch-cpu"


def _load_artifacts() -> None:
    global active_backend
    ckpt = OUTPUT_DIR / "memory_bank.pkl"
    if ckpt.exists():
        detector.load(ckpt)
        xml_path = OUTPUT_DIR / "feature_extractor.xml"
        if cfg["inference"].get("use_openvino", True) and xml_path.exists():
            try:
                detector.load_openvino(xml_path)
                active_backend = "openvino-cpu"
            except Exception:
                active_backend = "pytorch-cpu"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _load_artifacts()
    yield


app = FastAPI(
    title="VisionGuard AI",
    version="1.0.0",
    description="Intelligent industrial visual defect detection platform.",
    lifespan=lifespan,
)


def _inspect_single_bytes(raw_bytes: bytes, filename: str, threshold: float | None) -> dict[str, Any]:
    if detector.memory_bank is None:
        _load_artifacts()
    if detector.memory_bank is None:
        raise HTTPException(status_code=503, detail="Model not trained. Run scripts/train_and_export.py first.")

    np_arr = np.frombuffer(raw_bytes, np.uint8)
    bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {filename}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized_rgb = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE))
    tensor = transform(Image.fromarray(rgb)).unsqueeze(0)

    scores, heatmaps, latency_ms = detector.predict(tensor)
    score = round(float(scores[0]), 4)
    thr = threshold if threshold is not None else detector.calibrated_threshold
    status, severity, confidence = classify_severity(score, thr)

    overlay, bboxes = visualizer.apply_heatmap(
        resized_rgb,
        heatmaps[0],
        threshold=thr,
        vmax=detector.score_vmax,
    )
    out_name = f"inspect_{uuid.uuid4().hex[:10]}.png"
    out_path = RESULTS_DIR / out_name
    cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    record = {
        "filename": filename,
        "status": status,
        "severity": severity,
        "anomaly_score": score,
        "threshold": round(float(thr), 4),
        "confidence": confidence,
        "latency_ms": latency_ms,
        "backend": active_backend,
        "model_version": detector.model_version,
        "bboxes": bboxes,
        "output_image": f"/results/{out_name}",
    }
    record["inspection_id"] = history_db.record(record)
    return record


@app.get("/health", tags=["System"])
def health() -> dict[str, Any]:
    return {
        "status": "healthy" if detector.memory_bank is not None else "degraded",
        "model_loaded": detector.memory_bank is not None,
        "backend": active_backend,
        "model_version": detector.model_version,
    }


@app.get("/model/info", tags=["Model"])
def model_info() -> dict[str, Any]:
    metrics_path = OUTPUT_DIR / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    return {
        "model_version": detector.model_version,
        "backbone": detector.backbone_name,
        "backend": active_backend,
        "calibrated_threshold": detector.calibrated_threshold,
        "metrics": metrics,
    }


@app.post("/inspect", tags=["Inspection"])
async def inspect_image(
    file: UploadFile = File(...),
    threshold: float | None = Query(default=None, ge=0.0),
) -> dict[str, Any]:
    content = await file.read()
    return _inspect_single_bytes(content, file.filename or "upload.png", threshold)


@app.post("/inspect/batch", tags=["Inspection"])
async def inspect_batch(
    files: list[UploadFile] = File(...),
    threshold: float | None = Query(default=None, ge=0.0),
) -> dict[str, Any]:
    results = []
    for f in files:
        content = await f.read()
        results.append(_inspect_single_bytes(content, f.filename or "upload.png", threshold))
    ng_count = sum(1 for r in results if r["status"] == "NG")
    return {
        "total_images": len(results),
        "ok_count": len(results) - ng_count,
        "ng_count": ng_count,
        "results": results,
    }


@app.get("/history", tags=["History"])
def inspection_history(limit: int = 50) -> list[dict[str, Any]]:
    return history_db.list_recent(limit=limit)


@app.get("/results/{filename}", tags=["Results"])
def get_result_image(filename: str) -> FileResponse:
    path = RESULTS_DIR / Path(filename).name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Result image not found.")
    return FileResponse(path=str(path), media_type="image/png")
