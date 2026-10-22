"""Train PatchCore, evaluate on test split, export OpenVINO IR, and benchmark CPU latency."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset_loader import get_dataloader
from src.models.patchcore import PatchCore


def main() -> dict:
    config_path = ROOT / "configs" / "config.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    ds_cfg = cfg["dataset"]
    m_cfg = cfg["model"]
    out_dir = ROOT / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader = get_dataloader(
        root=ROOT / ds_cfg["path"],
        category=ds_cfg["category"],
        split="train",
        image_size=ds_cfg["image_size"],
        batch_size=8,
    )
    test_loader = get_dataloader(
        root=ROOT / ds_cfg["path"],
        category=ds_cfg["category"],
        split="test",
        image_size=ds_cfg["image_size"],
        batch_size=8,
    )

    model = PatchCore(
        backbone_name=m_cfg["backbone"],
        pca_components=m_cfg["pca_components"],
        coreset_ratio=m_cfg["coreset_sampling_ratio"],
        model_version=m_cfg.get("version", "1.0.0"),
    )
    t_train = time.perf_counter()
    model.fit(train_loader)
    train_seconds = round(time.perf_counter() - t_train, 3)

    # Evaluate PyTorch CPU
    y_true, y_scores, pt_latencies = [], [], []
    for images, labels, _, _ in test_loader:
        scores, _, lat_ms = model.predict(images)
        y_true.extend(labels.numpy().tolist())
        y_scores.extend(scores.tolist())
        pt_latencies.append(lat_ms / max(len(labels), 1))

    y_pred = [1 if s >= model.calibrated_threshold else 0 for s in y_scores]
    roc_auc = round(float(roc_auc_score(y_true, y_scores)), 4) if len(set(y_true)) > 1 else 1.0
    accuracy = round(float(accuracy_score(y_true, y_pred)), 4)
    f1 = round(float(f1_score(y_true, y_pred, zero_division=0)), 4)

    model.save(out_dir)

    # Export OpenVINO IR & benchmark CPU latency
    ov_latency_ms = None
    try:
        xml_path = model.export_openvino(out_dir, image_size=ds_cfg["image_size"])
        model.load_openvino(xml_path)
        ov_latencies = []
        for images, labels, _, _ in test_loader:
            _, _, lat_ms = model.predict(images)
            ov_latencies.append(lat_ms / max(len(labels), 1))
        ov_latency_ms = round(float(np.mean(ov_latencies)), 2)
    except Exception as exc:
        print(f"OpenVINO export/benchmark skipped: {exc}")

    metrics = {
        "model_version": model.model_version,
        "backbone": model.backbone_name,
        "category": ds_cfg["category"],
        "train_seconds": train_seconds,
        "calibrated_threshold": model.calibrated_threshold,
        "score_vmax": model.score_vmax,
        "test_samples": len(y_true),
        "roc_auc": roc_auc,
        "accuracy": accuracy,
        "f1_score": f1,
        "pytorch_cpu_ms_per_image": round(float(np.mean(pt_latencies)), 2),
        "openvino_cpu_ms_per_image": ov_latency_ms,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Training & evaluation complete:", json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    main()
