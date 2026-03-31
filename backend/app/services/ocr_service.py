"""
OCRService
──────────
Two-stage Scene Text Recognition pipeline (mirrors the position paper):
  1. YOLOv8  → detect text regions (bounding boxes)
  2. CRNN    → recognize Persian characters from each crop

Swap in your trained weights; the pipeline logic stays the same.
"""

from __future__ import annotations
import os
import cv2
import numpy as np
from pathlib import Path
from typing import Any

# These imports will resolve once you have the actual model weights.
# YOLOv8 via Ultralytics; CRNN is a custom PyTorch model.
try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from app.models.crnn import CRNNRecognizer     # custom model (see models/crnn.py)
from app.utils.persian_utils import (
    sort_boxes_rtl,
    fix_rtl_order,
    join_persian_words,
)


class OCRService:
    """Wraps detection + recognition into a single `.run()` call."""

    def __init__(
        self,
        yolo_weights: str,
        crnn_weights: str,
        conf_threshold: float = 0.4,
        img_size: int = 640,
    ):
        self.conf_threshold = conf_threshold
        self.img_size       = img_size
        self._yolo          = None
        self._crnn          = None

        self._yolo_path = yolo_weights
        self._crnn_path = crnn_weights

    # ── Lazy-load models so Flask startup stays fast ──────────────────────────
    @property
    def yolo(self):
        if self._yolo is None:
            if not _YOLO_AVAILABLE:
                raise RuntimeError("ultralytics not installed — run `pip install ultralytics`")
            if not Path(self._yolo_path).exists():
                raise FileNotFoundError(f"YOLO weights not found: {self._yolo_path}")
            self._yolo = YOLO(self._yolo_path)
        return self._yolo

    @property
    def crnn(self):
        if self._crnn is None:
            self._crnn = CRNNRecognizer.load(self._crnn_path)
        return self._crnn

    # ── Main pipeline ─────────────────────────────────────────────────────────
    def run(self, img_path: str, job_id: str) -> dict[str, Any]:
        """
        Full STR pipeline.

        Returns
        -------
        {
          "persian_text": str,
          "regions": [{"box": [x1,y1,x2,y2], "confidence": float, "text": str}],
          "annotated_image_url": str,
        }
        """
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Could not read image: {img_path}")

        # Step 1 ── Preprocessing
        preprocessed = self._preprocess(image)

        # Step 2 ── YOLOv8 text-region detection
        detections = self._detect(preprocessed)          # list of (box, conf)

        # Step 3 ── Sort boxes right-to-left (Persian reading order)
        detections = sort_boxes_rtl(detections)

        # Step 4 ── CRNN recognition per crop
        regions = []
        for box, conf in detections:
            x1, y1, x2, y2 = box
            crop  = image[y1:y2, x1:x2]
            text  = self._recognize(crop)
            regions.append({"box": list(box), "confidence": round(conf, 4), "text": text})

        # Step 5 ── Post-process: join words in RTL order
        full_text = join_persian_words([r["text"] for r in regions])

        # Step 6 ── Annotate & save result image
        annotated_path = self._annotate_and_save(image, regions, img_path, job_id)

        return {
            "persian_text": full_text,
            "regions": regions,
            "annotated_image_url": f"/api/ocr/result/{job_id}",
        }

    # ── Internal helpers ──────────────────────────────────────────────────────
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Resize + optional grayscale/threshold. Returns RGB for YOLO."""
        resized = cv2.resize(image, (self.img_size, self.img_size))
        return resized

    def _detect(self, image: np.ndarray) -> list[tuple[list[int], float]]:
        """Run YOLOv8; return list of ([x1,y1,x2,y2], conf) in pixel coords."""
        results = self.yolo.predict(
            image,
            conf=self.conf_threshold,
            imgsz=self.img_size,
            verbose=False,
        )
        boxes = []
        for r in results:
            for b in r.boxes:
                xyxy = b.xyxy[0].cpu().numpy().astype(int).tolist()
                conf = float(b.conf[0].cpu())
                boxes.append((xyxy, conf))
        return boxes

    def _recognize(self, crop: np.ndarray) -> str:
        """Pass a single cropped region through the CRNN recognizer."""
        if crop.size == 0:
            return ""
        return self.crnn.predict(crop)

    def _annotate_and_save(
        self,
        image: np.ndarray,
        regions: list[dict],
        img_path: str,
        job_id: str,
    ) -> str:
        annotated = image.copy()
        for r in regions:
            x1, y1, x2, y2 = r["box"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 100), 2)
            cv2.putText(
                annotated, r["text"],
                (x1, max(y1 - 6, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 100), 2,
            )
        out_path = str(Path(img_path).parent / f"{job_id}_annotated.jpg")
        cv2.imwrite(out_path, annotated)
        return out_path
