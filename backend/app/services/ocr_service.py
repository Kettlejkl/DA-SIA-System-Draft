"""
OCRService
──────────
Two-branch Scene Text Recognition pipeline (mirrors the system diagram):

  Branch A — Word-level (YES path in diagram):
    YOLOv8 → word-level bounding box → CRNN (CNN + BiLSTM + CTC) → char sequence

  Branch B — Char-level (NO path in diagram):
    YOLOv8 → character bounding box  → CNN classifier → single char

  The branch is selected automatically based on the YOLO model's class names:
    - If class names contain "_" (positional suffixes like "ب_initial") → char-level
    - Otherwise → word-level

  Both branches feed into the same RTL post-processing and output format.
"""

from __future__ import annotations

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Any

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

from app.models.crnn import CRNNRecognizer
from app.models.cnn_classifier import CNNClassifier
from app.utils.persian_utils import sort_boxes_rtl, fix_rtl_order, join_persian_words


# ── Tuneable thresholds ────────────────────────────────────────────────────────
LINE_Y_FACTOR   = 0.6
WORD_X_FACTOR   = 2.5
CRNN_PAD_FACTOR = 0.3
CRNN_PAD_MIN    = 20

# ── Char-branch NMS ───────────────────────────────────────────────────────────
# Boxes with IoU above this are considered duplicates; keep the higher-conf one.
CHAR_NMS_IOU_THRESHOLD = 0.3

# ── CNN crop padding ──────────────────────────────────────────────────────────
# YOLO boxes are tight to the character stroke. Resizing a tight crop to 32x32
# fills the frame completely, destroying spatial context (dots, diacritics).
# Adding white padding before resize restores that context and significantly
# improves CNN accuracy — especially for visually similar chars like ز/ق/ر.
# Value is a fraction of the crop's shorter dimension. 0.15 = ~10-15px on a
# typical 70-110px tall character crop.
CNN_CROP_PAD_FACTOR = 0.15

# ── Crop normalisation ────────────────────────────────────────────────────────
# Minimum stddev of the grayscale crop required to attempt Otsu binarisation.
# Crops below this threshold are nearly uniform (featureless) and Otsu would
# produce a meaningless split — we skip binarisation and keep the raw gray.
CNN_BINARIZE_MIN_STD = 20.0

# ── Debug ──────────────────────────────────────────────────────────────────────
DEBUG_CROPS = True    # Set to False to disable saving debug crops
DEBUG_DIR   = "debug_crops"


class OCRService:
    """
    Wraps detection + recognition into a single `.run()` call.
    Automatically selects word-level CRNN or char-level CNN based on
    the YOLO model's class names.
    """

    def __init__(
        self,
        yolo_weights: str,
        crnn_weights: str,
        cnn_weights: str,
        conf_threshold: float = 0.4,
        img_size: int = 640,
        cnn_confidence_threshold: float = 0.4,
    ):
        self.conf_threshold = conf_threshold
        self.img_size       = img_size

        self._yolo: YOLO | None           = None
        self._crnn: CRNNRecognizer | None = None
        self._cnn:  CNNClassifier | None  = None

        self._yolo_path       = yolo_weights
        self._crnn_path       = crnn_weights
        self._cnn_path        = cnn_weights
        self._cnn_conf_thresh = cnn_confidence_threshold

        self._mode: str | None = None   # "char" | "word"

    # ── Lazy loaders ──────────────────────────────────────────────────────────

    @property
    def yolo(self) -> YOLO:
        if self._yolo is None:
            if not _YOLO_AVAILABLE:
                raise RuntimeError("ultralytics not installed — pip install ultralytics")
            if not Path(self._yolo_path).exists():
                raise FileNotFoundError(f"YOLO weights not found: {self._yolo_path}")
            self._yolo = YOLO(self._yolo_path)
            has_positional = any("_" in name for name in self._yolo.names.values())
            self._mode = "char" if has_positional else "word"
            print(f"[OCRService] YOLO loaded — mode={self._mode} — "
                  f"{len(self._yolo.names)} classes")
        return self._yolo

    @property
    def crnn(self) -> CRNNRecognizer:
        if self._crnn is None:
            self._crnn = CRNNRecognizer.load(self._crnn_path)
        return self._crnn

    @property
    def cnn(self) -> CNNClassifier:
        if self._cnn is None:
            self._cnn = CNNClassifier(
                self._cnn_path,
                confidence_threshold=self._cnn_conf_thresh,
            )
        return self._cnn

    # ── Main pipeline ──────────────────────────────────────────────────────────

    def run(self, img_path: str, job_id: str) -> dict[str, Any]:
        """
        Full STR pipeline. Returns:
        {
          "persian_text": str,
          "regions": [{"box", "confidence", "text", "source", ...}],
          "annotated_image_url": str,
          "mode": "char" | "word",
        }
        """
        # Read original image — do NOT resize before cropping
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Could not read image: {img_path}")

        h, w = image.shape[:2]
        print(f"[OCRService] Image size: {w}x{h}")

        raw_detections = self._detect(img_path, self.conf_threshold)
        print(f"[OCRService] Raw detections: {len(raw_detections)}")

        # Branch based on mode
        if self._mode == "char":
            regions = self._run_char_branch(image, raw_detections, w, h)
        else:
            regions = self._run_word_branch(image, raw_detections, w, h)

        # RTL sort + join
        regions_sorted = sorted(regions, key=lambda r: (r["box"][1], -r["box"][0]))
        full_text = join_persian_words([r["text"] for r in regions_sorted])
        print(f"[OCRService] Full text: '{full_text}'")

        # Annotate and save
        self._annotate_and_save(image, regions, img_path, job_id)

        return {
            "persian_text":        full_text,
            "regions":             regions,
            "annotated_image_url": f"/api/ocr/result/{job_id}",
            "mode":                self._mode,
        }

    # ── NMS for character boxes ───────────────────────────────────────────────

    @staticmethod
    def _box_iou(a: list[int], b: list[int]) -> float:
        """Compute IoU between two [x1,y1,x2,y2] boxes."""
        ix1 = max(a[0], b[0])
        iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2])
        iy2 = min(a[3], b[3])

        inter_w = max(0, ix2 - ix1)
        inter_h = max(0, iy2 - iy1)
        inter   = inter_w * inter_h

        if inter == 0:
            return 0.0

        area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
        area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
        union  = area_a + area_b - inter

        return inter / union if union > 0 else 0.0

    def _nms_detections(
        self,
        detections: list[tuple],
        iou_threshold: float = CHAR_NMS_IOU_THRESHOLD,
    ) -> list[tuple]:
        """
        Greedy NMS over (box, conf, cls_idx) detections.
        Sorts by confidence descending, suppresses overlapping boxes.
        """
        if not detections:
            return []

        sorted_dets = sorted(detections, key=lambda d: d[1], reverse=True)
        kept = []

        for det in sorted_dets:
            box = det[0]
            suppressed = any(
                self._box_iou(box, k[0]) > iou_threshold
                for k in kept
            )
            if not suppressed:
                kept.append(det)

        removed = len(detections) - len(kept)
        if removed:
            print(f"[NMS] Removed {removed} duplicate detection(s) "
                  f"(IoU threshold={iou_threshold})")

        return kept

    # ── Crop normalisation ────────────────────────────────────────────────────

    @staticmethod
    def _normalize_crop(crop: np.ndarray) -> np.ndarray:
        """
        Normalise a BGR crop to black-strokes-on-white-background regardless
        of the source image style.  Handles three real-world cases:

          Case 1 — black text on light bg  (ideal, no-op after grayscale)
          Case 2 — white/color text on dark bg  (alphabet chart, dark scene)
          Case 3 — color text on light bg  (styled/decorative fonts)

        Steps
        ─────
        1. Grayscale  — strips color noise; character shape is in luminance.
        2. Invert     — if the background is dark, flip so bg becomes white.
        3. Binarise   — Otsu threshold → pure black strokes on pure white.
                        Skipped when contrast is too low (flat histogram) to
                        avoid a meaningless split on near-uniform crops.

        Returns a 3-channel BGR image ready for CNN input.
        """
        # 1. Grayscale
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # 2. Invert if background is dark
        if gray.mean() <= 127:
            gray = cv2.bitwise_not(gray)
            print(f"    [normalize] dark bg detected — inverted")

        # 3. Binarise with Otsu (skip if crop is nearly uniform / featureless)
        if gray.std() >= CNN_BINARIZE_MIN_STD:
            _, gray = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            print(f"    [normalize] Otsu binarised")
        else:
            print(f"    [normalize] low contrast (std={gray.std():.1f}) — skipped binarise")

        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # ── Branch A: Character-level (CNN classifier) ────────────────────────────

    def _run_char_branch(
        self,
        image: np.ndarray,
        detections: list,
        w: int,
        h: int,
    ) -> list[dict]:
        """
        Each YOLO detection = one character.
        CNN classifier predicts the base Persian character from the crop.

        Pipeline:
          1. Filter to char detections (class name contains "_")
          2. Apply NMS to remove duplicate boxes from YOLO
          3. Crop from original image
          4. Normalise crop (invert dark bg, binarise)
          5. Pad with white border
          6. Batch CNN inference
        """
        class_names = self.yolo.names

        char_dets = [
            d for d in detections
            if "_" in class_names.get(d[2], "")
        ]
        print(f"[Char branch] {len(char_dets)} character detections (pre-NMS)")

        char_dets = self._nms_detections(char_dets, iou_threshold=CHAR_NMS_IOU_THRESHOLD)
        print(f"[Char branch] {len(char_dets)} character detections (post-NMS)")

        if not char_dets:
            return []

        if DEBUG_CROPS:
            os.makedirs(DEBUG_DIR, exist_ok=True)

        crops: list[np.ndarray] = []
        valid: list[tuple]      = []

        for i, det in enumerate(char_dets):
            box, conf_, cls_idx = det
            x1, y1, x2, y2 = box

            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(w, int(x2))
            y2 = min(h, int(y2))

            if x2 <= x1 or y2 <= y1:
                print(f"  [skip] Invalid box ({x1},{y1},{x2},{y2})")
                continue

            crop = image[y1:y2, x1:x2]

            if crop.size == 0:
                print(f"  [skip] Empty crop ({x1},{y1},{x2},{y2})")
                continue

            crop = self._normalize_crop(crop)

            ch, cw = crop.shape[:2]
            pad_px = max(4, int(min(ch, cw) * CNN_CROP_PAD_FACTOR))
            crop = cv2.copyMakeBorder(
                crop, pad_px, pad_px, pad_px, pad_px,
                cv2.BORDER_CONSTANT, value=(255, 255, 255),
            )

            if DEBUG_CROPS:
                yolo_class = class_names.get(cls_idx, "unknown")
                safe_class = yolo_class.replace("/", "_").replace("\\", "_")
                debug_path = os.path.join(
                    DEBUG_DIR,
                    f"crop_{i:02d}_{x1}_{y1}_cls{cls_idx}_{safe_class}.png"
                )
                cv2.imwrite(debug_path, crop)
                print(f"  [debug] Saved: {debug_path}  size={crop.shape[1]}x{crop.shape[0]}")

            crops.append(crop)
            valid.append((box, conf_, cls_idx, x1, y1, x2, y2))

        if not crops:
            print("[Char branch] No valid crops extracted")
            return []

        predictions = self.cnn.predict_batch(crops)

        regions = []
        for (box, conf_, cls_idx, x1, y1, x2, y2), (char, cnn_conf) in zip(
            valid, predictions
        ):
            yolo_class = class_names.get(cls_idx, "?")

            if char == "?" or cnn_conf < self._cnn_conf_thresh:
                yolo_base = yolo_class.split("_")[0]
                print(f"  ({x1},{y1},{x2},{y2}) yolo={yolo_class} "
                      f"cnn='{char}' conf={cnn_conf:.3f} → fallback to yolo_base='{yolo_base}'")
                regions.append({
                    "box":        [x1, y1, x2, y2],
                    "confidence": round(conf_,    4),
                    "cnn_conf":   round(cnn_conf, 4),
                    "text":       yolo_base,
                    "source":     "yolo_class",
                    "char_count": 1,
                    "yolo_class": yolo_class,
                })
            else:
                print(f"  ({x1},{y1},{x2},{y2}) yolo={yolo_class} "
                      f"cnn='{char}' conf={cnn_conf:.3f}")
                regions.append({
                    "box":        [x1, y1, x2, y2],
                    "confidence": round(conf_,    4),
                    "cnn_conf":   round(cnn_conf, 4),
                    "text":       char,
                    "source":     "cnn_classifier",
                    "char_count": 1,
                    "yolo_class": yolo_class,
                })

        return regions

    # ── Branch B: Word-level (CRNN) ───────────────────────────────────────────

    def _run_word_branch(
        self,
        image: np.ndarray,
        detections: list,
        w: int,
        h: int,
    ) -> list[dict]:
        """
        Each YOLO detection = one word/text region.
        CRNN predicts the full character sequence from the crop.
        """
        print(f"[Word branch] {len(detections)} word detections")

        if not detections:
            return []

        heights = [d[0][3] - d[0][1] for d in detections]
        avg_h   = max(float(np.median(heights)), 10.0)
        pad     = max(CRNN_PAD_MIN, int(avg_h * CRNN_PAD_FACTOR))

        detections = sort_boxes_rtl(detections)

        regions = []
        for box, conf_, *_ in detections:
            x1, y1, x2, y2 = box
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            if x2 <= x1 or y2 <= y1:
                continue

            cx1  = max(0, x1 - pad)
            cy1  = max(0, y1 - pad)
            cx2  = min(w, x2 + pad)
            cy2  = min(h, y2 + pad)
            crop = image[cy1:cy2, cx1:cx2]

            text = self._recognize_crnn(crop)
            print(f"  ({x1},{y1},{x2},{y2}) crnn='{text}'")

            regions.append({
                "box":        [x1, y1, x2, y2],
                "confidence": round(conf_, 4),
                "text":       text,
                "source":     "crnn",
                "char_count": len(text),
            })

        return regions

    # ── YOLO detect ───────────────────────────────────────────────────────────

    def _detect_with(
        self,
        model: YOLO,
        img_path: str,
        conf: float,
    ) -> list[tuple[list[int], float, int]]:
        """Run detection with any given YOLO model."""
        results = model.predict(
            img_path,
            conf=conf,
            imgsz=self.img_size,
            verbose=False,
        )
        boxes = []
        for r in results:
            for b in r.boxes:
                xyxy    = b.xyxy[0].cpu().numpy().astype(int).tolist()
                conf_   = float(b.conf[0].cpu())
                cls_idx = int(b.cls[0].cpu())
                boxes.append((xyxy, conf_, cls_idx))
        return boxes

    def _detect(
        self,
        img_path: str,
        conf: float,
    ) -> list[tuple[list[int], float, int]]:
        """Run detection with the default YOLO model."""
        return self._detect_with(self.yolo, img_path, conf)

    # ── CRNN recognize ────────────────────────────────────────────────────────

    def _recognize_crnn(self, crop: np.ndarray) -> str:
        if crop is None or crop.size == 0:
            return ""
        return self.crnn.predict(crop)

    # ── Annotation ────────────────────────────────────────────────────────────

    def _annotate_and_save(
        self,
        image: np.ndarray,
        regions: list[dict],
        img_path: str,
        job_id: str,
    ) -> str:
        SOURCE_COLORS = {
            "cnn_classifier": (0,   220, 80),
            "crnn":           (0,   180, 255),
            "yolo_class":     (0,   200, 255),
        }

        annotated = image.copy()
        for r in regions:
            x1, y1, x2, y2 = r["box"]
            source = r.get("source", "cnn_classifier")
            color  = SOURCE_COLORS.get(source, (0, 220, 80))

            conf_label = ""
            if "cnn_conf" in r:
                conf_label = f" {r['cnn_conf']:.2f}"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                f"{r['text']}{conf_label}",
                (x1, max(y1 - 6, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
            )

        out_path = str(Path(img_path).parent / f"{job_id}_annotated.jpg")
        cv2.imwrite(out_path, annotated)
        return out_path