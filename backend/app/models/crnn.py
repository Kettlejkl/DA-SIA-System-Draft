"""
CRNN (Convolutional Recurrent Neural Network) for Persian script recognition.
Architecture: CNN feature extractor → BiLSTM sequence modeller → CTC decoder

Reference: Graves et al. (2006) CTC; Li et al. (2017) CRNN for STR.
"""

from __future__ import annotations
import numpy as np
import cv2

try:
    import torch
    import torch.nn as nn
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

# Persian alphabet + blank token for CTC
PERSIAN_CHARS = (
    " ا ب پ ت ث ج چ ح خ د ذ ر ز ژ س ش ص ض ط ظ ع غ ف ق ک گ ل م ن و ه ی"
    " آ ئ ء أ إ ة لا"
).split()
BLANK_TOKEN = "<blank>"
VOCAB       = [BLANK_TOKEN] + PERSIAN_CHARS
CHAR2IDX    = {c: i for i, c in enumerate(VOCAB)}
IDX2CHAR    = {i: c for c, i in CHAR2IDX.items()}


class _ConvBlock(nn.Module if _TORCH_OK else object):
    def __init__(self, in_ch, out_ch, ks=3, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, ks, padding=ks // 2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class _CRNNNet(nn.Module if _TORCH_OK else object):
    """
    Input : (B, 1, 32, W)  — grayscale, height-normalised crops
    Output: (T, B, num_classes)  — log-softmax for CTC
    """

    def __init__(self, num_classes: int, rnn_hidden: int = 256):
        super().__init__()
        self.cnn = nn.Sequential(
            _ConvBlock(1,  64, pool=True),   # → (B, 64,  16, W/2)
            _ConvBlock(64, 128, pool=True),  # → (B, 128,  8, W/4)
            _ConvBlock(128, 256, pool=False),
            _ConvBlock(256, 256, pool=True), # → (B, 256,  4, W/8)
            _ConvBlock(256, 512, pool=False),
            _ConvBlock(512, 512, pool=True), # → (B, 512,  2, W/16)
            nn.Conv2d(512, 512, (2, 1)),     # squeeze height → (B, 512, 1, W/16)
            nn.ReLU(inplace=True),
        )
        self.rnn = nn.Sequential(
            nn.LSTM(512, rnn_hidden, bidirectional=True, batch_first=False),
        )
        self.fc = nn.Linear(rnn_hidden * 2, num_classes)

    def forward(self, x):
        # CNN
        feat = self.cnn(x)          # (B, C, 1, W')
        feat = feat.squeeze(2)      # (B, C, W')
        feat = feat.permute(2, 0, 1) # (W', B, C)  → sequence-first

        # RNN
        out, _ = self.rnn[0](feat)  # (T, B, hidden*2)

        # Linear + log-softmax
        return torch.nn.functional.log_softmax(self.fc(out), dim=2)


# ── Public wrapper ────────────────────────────────────────────────────────────
class CRNNRecognizer:
    """High-level wrapper: image → Persian string."""

    IMG_H = 32        # fixed height after resizing
    IMG_W = 128       # fixed width (or dynamic — set to None for variable)

    def __init__(self, model: "_CRNNNet | None" = None, device: str = "cpu"):
        self._model  = model
        self._device = device

    @classmethod
    def load(cls, weights_path: str) -> "CRNNRecognizer":
        """Load a trained checkpoint. Returns a dummy instance if no weights yet."""
        if not _TORCH_OK:
            raise RuntimeError("PyTorch not installed")

        import os
        device = "cuda" if torch.cuda.is_available() else "cpu"
        net    = _CRNNNet(num_classes=len(VOCAB))

        if os.path.exists(weights_path):
            state = torch.load(weights_path, map_location=device)
            net.load_state_dict(state)
            net.eval()
        else:
            # No weights yet — model will return placeholder output
            print(f"[CRNNRecognizer] WARNING: weights not found at {weights_path}. "
                  "Running in untrained stub mode.")

        net.to(device)
        return cls(model=net, device=device)

    def predict(self, crop_bgr: np.ndarray) -> str:
        """Recognise text in a single BGR image crop."""
        if self._model is None:
            return "???"  # stub when no weights loaded

        tensor = self._preprocess(crop_bgr)
        with torch.no_grad():
            logits = self._model(tensor)     # (T, 1, num_classes)
        return self._ctc_decode(logits)

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _preprocess(self, img: np.ndarray) -> "torch.Tensor":
        gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (self.IMG_W, self.IMG_H))
        arr     = resized.astype(np.float32) / 255.0
        arr     = (arr - 0.5) / 0.5           # normalise to [-1, 1]
        tensor  = torch.tensor(arr).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        return tensor.to(self._device)

    @staticmethod
    def _ctc_decode(logits: "torch.Tensor") -> str:
        """Greedy CTC decode: argmax at each time step, collapse repeats, strip blank."""
        indices = logits.squeeze(1).argmax(dim=1).tolist()  # (T,)
        chars, prev = [], -1
        for idx in indices:
            if idx != prev and idx != 0:   # 0 = BLANK
                chars.append(IDX2CHAR.get(idx, ""))
            prev = idx
        return "".join(chars)
