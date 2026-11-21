"""Tests for VisionGuard AI PatchCore, visualizer, OpenVINO export, and FastAPI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from api.main import app
from src.pipeline.visualizer import classify_severity


def test_severity_classification() -> None:
    status_ok, sev_ok, conf_ok = classify_severity(0.8, threshold=1.5)
    assert status_ok == "OK"
    assert sev_ok == "NONE"
    assert 0.5 <= conf_ok <= 1.0

    status_ng, sev_ng, conf_ng = classify_severity(3.0, threshold=1.5)
    assert status_ng == "NG"
    assert sev_ng == "CRITICAL"
    assert 0.5 <= conf_ng <= 1.0


def test_health_and_model_info() -> None:
    with TestClient(app) as client:
        res = client.get("/health")
        assert res.status_code == 200
        assert res.json()["status"] == "healthy"

        info = client.get("/model/info")
        assert info.status_code == 200
        assert info.json()["backbone"] == "resnet18"


def test_single_batch_inspection_and_history() -> None:
    root = Path(__file__).resolve().parents[1]
    good_img = next((root / "dataset" / "bottle" / "test" / "good").glob("*.png"))
    bad_img = next((root / "dataset" / "bottle" / "test" / "anomaly").glob("*.png"))

    with TestClient(app) as client:
        with open(good_img, "rb") as fg:
            res_ok = client.post("/inspect", files={"file": ("good.png", fg, "image/png")})
        assert res_ok.status_code == 200
        assert res_ok.json()["status"] == "OK"

        with open(bad_img, "rb") as fb:
            res_ng = client.post("/inspect", files={"file": ("bad.png", fb, "image/png")})
        assert res_ng.status_code == 200
        assert res_ng.json()["status"] == "NG"
        assert res_ng.json()["anomaly_score"] > res_ok.json()["anomaly_score"]

        with open(good_img, "rb") as fg, open(bad_img, "rb") as fb:
            batch_res = client.post(
                "/inspect/batch",
                files=[
                    ("files", ("good.png", fg, "image/png")),
                    ("files", ("bad.png", fb, "image/png")),
                ],
            )
        assert batch_res.status_code == 200
        assert batch_res.json()["total_images"] == 2
        assert batch_res.json()["ng_count"] >= 1

        hist = client.get("/history")
        assert hist.status_code == 200
        assert len(hist.json()) >= 4
