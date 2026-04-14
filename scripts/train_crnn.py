"""
scripts/train_crnn.py
─────────────────────
Standalone training script for the CRNN recognition model.

Usage:
  python scripts/train_crnn.py \
      --data_dir  data/shotor/crops \
      --labels    data/shotor/labels.txt \
      --epochs    50 \
      --out       backend/models/weights/persian_crnn.pt

Expected data layout:
  data/
    shotor/
      crops/
        word_0001.png
        word_0002.png
        ...
      labels.txt       # one line per image: "word_0001.png سلام"
"""

import argparse, os, sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Add backend to sys.path so we can import our model
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from app.models.crnn import _CRNNNet, VOCAB, CHAR2IDX, BLANK_TOKEN

# ── Dataset ───────────────────────────────────────────────────────────────────
class PersianWordDataset(Dataset):
    IMG_H, IMG_W = 32, 128

    def __init__(self, data_dir: str, labels_file: str):
        self.data_dir = Path(data_dir)
        self.samples  = []
        with open(labels_file, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    self.samples.append((parts[0], parts[1]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        img = cv2.imread(str(self.data_dir / fname), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.IMG_W, self.IMG_H)).astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        tensor = torch.tensor(img).unsqueeze(0)  # (1, H, W)

        # Encode label as index list
        encoded = [CHAR2IDX.get(c, 0) for c in label if c in CHAR2IDX]
        return tensor, torch.tensor(encoded, dtype=torch.long)


def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    labels_concat = torch.cat(labels)
    return images, labels_concat, label_lengths


# ── Training loop ─────────────────────────────────────────────────────────────
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    dataset = PersianWordDataset(args.data_dir, args.labels)
    loader  = DataLoader(dataset, batch_size=args.batch, shuffle=True,
                         collate_fn=collate_fn, num_workers=2)

    model   = _CRNNNet(num_classes=len(VOCAB)).to(device)
    optim   = torch.optim.Adam(model.parameters(), lr=args.lr)
    ctc     = nn.CTCLoss(blank=0, zero_infinity=True)

    for epoch in range(1, args.epochs + 1):
        model.train(); total_loss = 0
        for imgs, labels, label_lens in loader:
            imgs   = imgs.to(device)
            labels = labels.to(device)

            logits = model(imgs)          # (T, B, C)
            T, B   = logits.size(0), logits.size(1)
            input_lens = torch.full((B,), T, dtype=torch.long)

            loss = ctc(logits, labels, input_lens, label_lens)
            optim.zero_grad(); loss.backward(); optim.step()
            total_loss += loss.item()

        avg = total_loss / len(loader)
        print(f"Epoch {epoch:03d}/{args.epochs} — loss: {avg:.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--labels",   required=True)
    p.add_argument("--epochs",   type=int,   default=50)
    p.add_argument("--batch",    type=int,   default=32)
    p.add_argument("--lr",       type=float, default=1e-3)
    p.add_argument("--out",      default="backend/models/weights/persian_crnn.pt")
    train(p.parse_args())
