"""Defect localization, severity classification, and global-scale heatmap visualization."""

from __future__ import annotations

import cv2
import matplotlib
import numpy as np


def classify_severity(score: float, threshold: float) -> tuple[str, str, float]:
    """Return (status_ok_ng, severity_tier, confidence_score)."""
    if score < threshold:
        confidence = min(0.99, max(0.5, 1.0 - (score / max(threshold * 2.0, 1e-6))))
        return "OK", "NONE", round(float(confidence), 4)

    ratio = score / max(threshold, 1e-6)
    confidence = min(0.99, max(0.5, 0.5 + 0.25 * (ratio - 1.0)))
    if ratio >= 1.75:
        severity = "CRITICAL"
    elif ratio >= 1.30:
        severity = "MAJOR"
    else:
        severity = "MINOR"
    return "NG", severity, round(float(confidence), 4)


class Visualizer:
    """Overlays calibrated anomaly heatmaps and defect bounding boxes onto RGB images."""

    def __init__(self, alpha: float = 0.45) -> None:
        self.alpha = alpha
        self.colormap = matplotlib.colormaps["jet"]

    def apply_heatmap(
        self,
        image_np: np.ndarray,
        heatmap_scores: np.ndarray,
        threshold: float,
        vmax: float = 3.0,
    ) -> tuple[np.ndarray, list[dict[str, int]]]:
        """Render global-scale heatmap overlay and detect defect bounding boxes."""
        h_img, w_img = image_np.shape[:2]
        resized_scores = cv2.resize(heatmap_scores.astype(np.float32), (w_img, h_img), interpolation=cv2.INTER_CUBIC)

        # Global-scale normalization prevents normal images from showing false red hotspots.
        norm_map = np.clip(resized_scores / max(vmax, 1e-6), 0.0, 1.0)
        colored = (self.colormap(norm_map)[:, :, :3] * 255).astype(np.uint8)
        overlay = cv2.addWeighted(image_np.astype(np.uint8), 1.0 - self.alpha, colored, self.alpha, 0)

        max_score = float(np.max(heatmap_scores))
        status, severity, _ = classify_severity(max_score, threshold)

        bboxes: list[dict[str, int]] = []
        if status == "NG":
            mask = (resized_scores >= threshold).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) < 25:
                    continue
                x, y, bw, bh = cv2.boundingRect(cnt)
                bboxes.append({"x": int(x), "y": int(y), "w": int(bw), "h": int(bh)})
                cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (255, 50, 50), 2)

        color = (220, 40, 40) if status == "NG" else (40, 180, 90)
        label = f"{status} ({severity}) | Score: {max_score:.2f} / Thr: {threshold:.2f}"
        cv2.putText(overlay, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        return overlay, bboxes
