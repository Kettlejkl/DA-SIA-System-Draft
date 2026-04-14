"""
scripts/evaluate.py
────────────────────
Compute Character Error Rate (CER) and Word Accuracy on a test set.

Usage:
  python scripts/evaluate.py \
      --data_dir  data/shotor/crops \
      --labels    data/shotor/test_labels.txt \
      --crnn      backend/models/weights/persian_crnn.pt
"""

import argparse, sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from app.models.crnn import CRNNRecognizer


def levenshtein(s1: str, s2: str) -> int:
    """Standard dynamic-programming edit distance."""
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[j] = min(dp[j-1]+1, prev[j]+1, prev[j-1]+cost)
    return dp[n]


def cer(pred: str, gt: str) -> float:
    if not gt:
        return 0.0 if not pred else 1.0
    return levenshtein(pred, gt) / len(gt)


def evaluate(args):
    recognizer = CRNNRecognizer.load(args.crnn)
    data_dir   = Path(args.data_dir)

    samples = []
    with open(args.labels, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                samples.append((parts[0], parts[1]))

    total_cer, correct_words = 0.0, 0

    for fname, gt_text in samples:
        img = cv2.imread(str(data_dir / fname))
        if img is None:
            continue
        pred = recognizer.predict(img)
        total_cer    += cer(pred, gt_text)
        correct_words += int(pred == gt_text)

    n = len(samples)
    print(f"Samples evaluated : {n}")
    print(f"Mean CER          : {total_cer/n:.4f}  ({total_cer/n*100:.2f}%)")
    print(f"Word Accuracy     : {correct_words/n:.4f}  ({correct_words/n*100:.2f}%)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--labels",   required=True)
    p.add_argument("--crnn",     required=True)
    evaluate(p.parse_args())
