"""Streamlit interactive visual inspection dashboard for VisionGuard AI."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from torchvision import transforms

from src.models.patchcore import PatchCore
from src.pipeline.visualizer import Visualizer, classify_severity

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"

st.set_page_config(page_title="VisionGuard AI", layout="wide")
st.title("VisionGuard AI — Industrial Visual Quality Inspector")

ckpt = OUTPUT_DIR / "memory_bank.pkl"
if not ckpt.exists():
    st.warning("No trained memory bank found. Run `python scripts/train_and_export.py` first.")
    st.stop()

detector = PatchCore().load(ckpt)
visualizer = Visualizer()
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

with st.sidebar:
    st.header("Inspection Settings")
    threshold = st.slider(
        "Anomaly Threshold",
        min_value=0.2,
        max_value=5.0,
        value=float(detector.calibrated_threshold),
        step=0.05,
    )
    metrics_path = OUTPUT_DIR / "metrics.json"
    if metrics_path.exists():
        st.subheader("Verified Model Metrics")
        st.json(json.loads(metrics_path.read_text(encoding="utf-8")))

uploaded = st.file_uploader("Upload product image (PNG/JPG)", type=["png", "jpg", "jpeg"])
if uploaded is not None:
    raw = np.frombuffer(uploaded.read(), np.uint8)
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (224, 224))
    tensor = transform(Image.fromarray(rgb)).unsqueeze(0)

    scores, heatmaps, latency_ms = detector.predict(tensor)
    score = float(scores[0])
    status, severity, confidence = classify_severity(score, threshold)
    overlay, bboxes = visualizer.apply_heatmap(resized, heatmaps[0], threshold=threshold, vmax=detector.score_vmax)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Decision", status)
    c2.metric("Severity", severity)
    c3.metric("Anomaly Score", f"{score:.4f}")
    c4.metric("Latency", f"{latency_ms:.2f} ms")

    col_left, col_right = st.columns(2)
    col_left.image(resized, caption="Original Image")
    col_right.image(overlay, caption=f"Defect Localization Heatmap (Confidence: {confidence:.2%})")
