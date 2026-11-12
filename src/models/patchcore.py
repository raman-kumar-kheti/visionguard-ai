"""CPU-optimized PatchCore anomaly detection with greedy coreset, calibration, and OpenVINO export."""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from torchvision import models


class _FeatureBackbone(nn.Module):
    """Frozen ResNet18 intermediate feature extractor (layer2 + layer3)."""

    def __init__(self) -> None:
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1)
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        f2 = self.layer2(h)
        f3 = self.layer3(f2)
        f2_avg = F.avg_pool2d(f2, kernel_size=3, stride=1, padding=1)
        f3_avg = F.avg_pool2d(f3, kernel_size=3, stride=1, padding=1)
        f3_up = F.interpolate(f3_avg, size=f2_avg.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([f2_avg, f3_up], dim=1)


def _greedy_coreset_subsample(features: np.ndarray, ratio: float, seed: int = 42) -> np.ndarray:
    """Seeded k-center greedy coreset subsampling for representative memory banks."""
    n = features.shape[0]
    target = max(1, int(n * ratio))
    if target >= n:
        return features

    rng = np.random.default_rng(seed)
    # Pre-sample candidate pool if very large to keep CPU training fast
    if n > 12000:
        pool_idx = rng.choice(n, size=12000, replace=False)
        features = features[pool_idx]
        n = features.shape[0]
        target = min(target, n)

    selected = [int(rng.integers(0, n))]
    min_dist = np.linalg.norm(features - features[selected[0]], axis=1)

    for _ in range(1, target):
        next_idx = int(np.argmax(min_dist))
        selected.append(next_idx)
        dist = np.linalg.norm(features - features[next_idx], axis=1)
        min_dist = np.minimum(min_dist, dist)

    return features[selected]


class PatchCore(nn.Module):
    """PatchCore anomaly detector with optional OpenVINO CPU acceleration."""

    def __init__(
        self,
        backbone_name: str = "resnet18",
        pca_components: int = 64,
        coreset_ratio: float = 0.1,
        model_version: str = "1.0.0",
    ) -> None:
        super().__init__()
        if backbone_name != "resnet18":
            raise NotImplementedError("Only 'resnet18' backbone is currently supported.")
        self.device = torch.device("cpu")
        self.backbone_name = backbone_name
        self.model_version = model_version
        self.coreset_ratio = coreset_ratio
        self.feature_extractor = _FeatureBackbone().to(self.device)
        self.pca: PCA | None = PCA(n_components=pca_components, random_state=42) if pca_components > 0 else None
        self.nn_search = NearestNeighbors(n_neighbors=5, algorithm="ball_tree")
        self.memory_bank: np.ndarray | None = None
        self.calibrated_threshold: float = 1.5
        self.score_vmax: float = 3.0
        self._ov_compiled = None

    @torch.inference_mode()
    def _extract_feature_map(self, x: torch.Tensor) -> np.ndarray:
        if self._ov_compiled is not None:
            out = self._ov_compiled([x.cpu().numpy()])[0]
            return out
        return self.feature_extractor(x.to(self.device)).cpu().numpy()

    def _to_patch_vectors(self, fmap: np.ndarray) -> tuple[np.ndarray, int, int]:
        b, c, h, w = fmap.shape
        patches = np.transpose(fmap, (0, 2, 3, 1)).reshape(-1, c)
        return patches, h, w

    def fit(self, dataloader: Any) -> "PatchCore":
        all_patches: list[np.ndarray] = []
        raw_maps: list[np.ndarray] = []
        for batch in dataloader:
            images = batch[0]
            fmap = self._extract_feature_map(images)
            raw_maps.append(fmap)
            patches, _, _ = self._to_patch_vectors(fmap)
            all_patches.append(patches)

        merged = np.concatenate(all_patches, axis=0)
        if self.pca is not None:
            n_comp = min(self.pca.n_components, merged.shape[0], merged.shape[1])
            self.pca = PCA(n_components=n_comp, random_state=42)
            merged = self.pca.fit_transform(merged)

        self.memory_bank = _greedy_coreset_subsample(merged, self.coreset_ratio)
        self.nn_search.fit(self.memory_bank)

        # Calibrate threshold using normal training image scores (mean + 3 * std)
        normal_scores: list[float] = []
        for fmap in raw_maps:
            b, _, h, w = fmap.shape
            patches, _, _ = self._to_patch_vectors(fmap)
            if self.pca is not None:
                patches = self.pca.transform(patches)
            dists, _ = self.nn_search.kneighbors(patches)
            patch_scores = dists.mean(axis=1).reshape(b, h, w)
            normal_scores.extend(np.max(patch_scores, axis=(1, 2)).tolist())

        mean_s = float(np.mean(normal_scores))
        std_s = float(np.std(normal_scores))
        self.calibrated_threshold = round(mean_s + 3.0 * max(std_s, 0.05), 4)
        self.score_vmax = round(max(self.calibrated_threshold * 2.0, 2.5), 4)
        return self

    def predict(self, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray, float]:
        """Return (image_scores, anomaly_heatmaps, latency_ms)."""
        if self.memory_bank is None:
            raise RuntimeError("Model memory_bank is not initialized. Train or load a model first.")
        t0 = time.perf_counter()
        fmap = self._extract_feature_map(x)
        b, _, h, w = fmap.shape
        patches, _, _ = self._to_patch_vectors(fmap)
        if self.pca is not None:
            patches = self.pca.transform(patches)
        dists, _ = self.nn_search.kneighbors(patches)
        patch_scores = dists.mean(axis=1).reshape(b, h, w)
        image_scores = np.max(patch_scores, axis=(1, 2))
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return image_scores, patch_scores, latency_ms

    def export_openvino(self, output_dir: str | Path, image_size: int = 224) -> Path:
        """Export the frozen ResNet18 feature extractor to ONNX and OpenVINO IR."""
        import openvino as ov

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        onnx_path = out / "feature_extractor.onnx"
        xml_path = out / "feature_extractor.xml"

        dummy = torch.randn(1, 3, image_size, image_size)
        torch.onnx.export(
            self.feature_extractor,
            dummy,
            str(onnx_path),
            input_names=["input"],
            output_names=["features"],
            dynamic_axes={"input": {0: "batch"}, "features": {0: "batch"}},
            opset_version=17,
        )
        ov_model = ov.convert_model(str(onnx_path))
        ov.save_model(ov_model, str(xml_path))
        return xml_path

    def load_openvino(self, xml_path: str | Path) -> bool:
        """Load OpenVINO IR for accelerated CPU inference."""
        import openvino as ov

        core = ov.Core()
        model = core.read_model(str(xml_path))
        self._ov_compiled = core.compile_model(model, "CPU")
        return True

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        ckpt_path = out / "memory_bank.pkl"
        with open(ckpt_path, "wb") as f:
            pickle.dump(
                {
                    "memory_bank": self.memory_bank,
                    "pca": self.pca,
                    "nn_search": self.nn_search,
                    "calibrated_threshold": self.calibrated_threshold,
                    "score_vmax": self.score_vmax,
                    "model_version": self.model_version,
                    "backbone_name": self.backbone_name,
                },
                f,
            )
        return ckpt_path

    def load(self, path: str | Path) -> "PatchCore":
        ckpt = Path(path)
        if ckpt.is_dir():
            ckpt = ckpt / "memory_bank.pkl"
        with open(ckpt, "rb") as f:
            data = pickle.load(f)
        self.memory_bank = data["memory_bank"]
        self.pca = data["pca"]
        self.nn_search = data["nn_search"]
        self.calibrated_threshold = float(data.get("calibrated_threshold", 1.5))
        self.score_vmax = float(data.get("score_vmax", 3.0))
        self.model_version = str(data.get("model_version", "1.0.0"))
        return self
