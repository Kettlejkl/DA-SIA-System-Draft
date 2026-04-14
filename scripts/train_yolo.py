"""
scripts/train_yolo.py
─────────────────────
Fine-tune YOLOv8 for Persian text region detection.

Usage:
  python scripts/train_yolo.py \
      --data  data/persian_text/dataset.yaml \
      --model yolov8s.pt \
      --epochs 80 \
      --out   backend/models/weights/persian_text_detector.pt

dataset.yaml format (standard Ultralytics):
  path: /abs/path/to/dataset
  train: images/train
  val:   images/val
  nc: 1
  names: ['persian_text']
"""

import argparse
from pathlib import Path


def train(args):
    from ultralytics import YOLO

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name="persian_text_detector",
        patience=20,
        save=True,
        device="0" if __import__("torch").cuda.is_available() else "cpu",
    )

    # Copy best weights to the target path
    best = Path(results.save_dir) / "weights" / "best.pt"
    out  = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import shutil; shutil.copy(best, out)
    print(f"Best weights saved → {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data",   required=True,  help="Path to dataset.yaml")
    p.add_argument("--model",  default="yolov8s.pt")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--imgsz",  type=int, default=640)
    p.add_argument("--batch",  type=int, default=16)
    p.add_argument("--out",    default="backend/models/weights/persian_text_detector.pt")
    train(p.parse_args())
