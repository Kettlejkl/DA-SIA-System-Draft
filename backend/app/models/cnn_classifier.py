"""
CNNClassifier
-------------
Single-character Persian/Arabic classifier used in Branch A (char-level) of
the OCR pipeline.

  Input:  a cropped BGR image of one character (np.ndarray)
  Output: (predicted_char: str, confidence: float)

  42 Persian/Arabic classes + digits 0-9 (Persian)
  Trained model accuracy: 95.05% (on clean isolated chars)

Architecture matched exactly to scripts/train_cnn_classifier.py (PersianCharCNN):
  b1:   Conv(1->32)   BN ReLU Conv(32->32)   BN ReLU
  b2:   Conv(32->64)  BN ReLU Conv(64->64)   BN ReLU
  b3:   Conv(64->128) BN ReLU Conv(128->128) BN ReLU
  b4:   Conv(128->256)BN ReLU Conv(256->256) BN ReLU
  pool: MaxPool2d(2,2)        -- separate module, applied after b1/b2/b3
  gap:  AdaptiveAvgPool2d(2,2)-- applied after b4
  head: Flatten -> Dropout(0.4) -> Linear(1024->512) -> ReLU
        -> Dropout(0.3) -> Linear(512->42)

Preprocessing matched to training:
  BGR -> grayscale -> resize(32,32,INTER_AREA) -> float32
  -> normalize: (x/255 - 0.5) / 0.5   range [-1, 1]
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Any

# -- Optional backend imports --------------------------------------------------
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    import onnxruntime as ort
    _ONNX_AVAILABLE = True
except ImportError:
    _ONNX_AVAILABLE = False

try:
    from tensorflow import keras
    _KERAS_AVAILABLE = True
except ImportError:
    _KERAS_AVAILABLE = False


# -- Constants -----------------------------------------------------------------
DEFAULT_INPUT_SIZE = (32, 32)
UNKNOWN_CHAR       = "?"
NUM_CLASSES        = 42

# -- Fallback class map (matches checkpoint idx2char exactly) ------------------
IDX2CHAR: dict[int, str] = {
    0:  '\u0627',  # ا  alef
    1:  '\u0628',  # ب  be
    2:  '\u067e',  # پ  pe
    3:  '\u062a',  # ت  te
    4:  '\u062b',  # ث  se
    5:  '\u062c',  # ج  jim
    6:  '\u0686',  # چ  che
    7:  '\u062d',  # ح  he
    8:  '\u062e',  # خ  khe
    9:  '\u062f',  # د  dal
    10: '\u0630',  # ذ  zal
    11: '\u0631',  # ر  re
    12: '\u0632',  # ز  ze
    13: '\u0698',  # ژ  zhe
    14: '\u0633',  # س  sin
    15: '\u0634',  # ش  shin
    16: '\u0635',  # ص  sad
    17: '\u0636',  # ض  zad
    18: '\u0637',  # ط  ta
    19: '\u0638',  # ظ  za
    20: '\u0639',  # ع  eyn
    21: '\u063a',  # غ  gheyn
    22: '\u0641',  # ف  fe
    23: '\u0642',  # ق  ghaf
    24: '\u06a9',  # ک  kaf
    25: '\u06af',  # گ  gaf
    26: '\u0644',  # ل  lam
    27: '\u0645',  # م  mim
    28: '\u0646',  # ن  nun
    29: '\u0648',  # و  waw
    30: '\u0647',  # ه  he
    31: '\u06cc',  # ی  ye
    32: '\u06f0',  # ۰
    33: '\u06f1',  # ۱
    34: '\u06f2',  # ۲
    35: '\u06f3',  # ۳
    36: '\u06f4',  # ۴
    37: '\u06f5',  # ۵
    38: '\u06f6',  # ۶
    39: '\u06f7',  # ۷
    40: '\u06f8',  # ۸
    41: '\u06f9',  # ۹
}

# -- Per-character confidence threshold overrides ------------------------------
# Final thresholds tuned from full empirical debug analysis of real pipeline crops.
#
# UNFIXABLE CONFUSION PAIRS (need retraining with positional form augmentation):
#   ی <-> گ   model predicts گ for ی at 0.69-0.81  [MODEL ISSUE]
#   ه <-> ن   model predicts ن for ه at 0.72-0.75  [MODEL ISSUE]
#   و <-> ق   model predicts ق for و at 0.37-0.95  [MODEL ISSUE]
#   ش <-> ض   model predicts ض for ش at 0.956      [MODEL ISSUE]
#   ظ <-> ط   model predicts ط for ظ at 0.854      [MODEL ISSUE]
#   س <-> ص   model predicts ص for س at 0.935      [MODEL ISSUE]
CHAR_THRESHOLD_OVERRIDES: dict[str, float] = {
    '\u0644': 0.06,   # ل  lam:  very low conf, distinct shape, safe to go very low
    '\u0641': 0.10,   # ف  fe:   inconsistent; 0.08 causes ذ->ف false positives
    '\u0642': 0.40,   # ق  ghaf: و->ق false positive observed; keep high
    '\u0648': 0.28,   # و  waw:  confused with ق [MODEL ISSUE] keep moderate
    '\u0647': 0.20,   # ه  he:   confused with ن [MODEL ISSUE]; lower for MISS recovery
    '\u06cc': 0.50,   # ی  ye:   confused with گ at 0.81 [MODEL ISSUE]; keep HIGH
    '\u06af': 0.11,   # گ  gaf:  initial form drops to 0.365
    '\u0637': 0.28,   # ط  ta:   drops to 0.314 on tight crops
    '\u0638': 0.38,   # ظ  za:   confused with ط [MODEL ISSUE]; keep moderate
    '\u0639': 0.17,   # ع  eyn:  drops to 0.189 when correct
    '\u0628': 0.13,   # ب  be:   drops to 0.159 on tight crops
    '\u062b': 0.19,   # ث  se:   low conf at 32x32
    '\u0645': 0.20,   # م  mim:  low conf; ص was winning due to lower threshold
    '\u0635': 0.32,   # ص  sad:  raised so م can compete
    '\u0632': 0.22,   # ز  ze:   similar stroke family
    '\u0631': 0.20,   # ر  re:   low conf on initial forms
    '\u0698': 0.13,   # ژ  zhe:  very low conf even when correct
    '\u0630': 0.22,   # ذ  zal:  similar to د
    '\u062f': 0.18,   # د  dal:  similar to ذ
    '\u062d': 0.30,   # ح  he-jimi: confused with خ
    '\u062c': 0.30,   # ج  jim:  confused with ح
    '\u06a9': 0.27,   # ک  kaf:  drops to 0.289 on some crops
}


# -- Model Architecture --------------------------------------------------------

def _conv_block(in_ch: int, out_ch: int) -> "nn.Sequential":
    """
    Two Conv->BN->ReLU layers. NO MaxPool inside -- pool is a separate
    module applied in PersianCNN.forward(), matching train_cnn_classifier.py.
    """
    return nn.Sequential(
        nn.Conv2d(in_ch,  out_ch, kernel_size=3, padding=1, bias=False),  # .0
        nn.BatchNorm2d(out_ch),                                            # .1
        nn.ReLU(inplace=True),                                             # .2
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),  # .3
        nn.BatchNorm2d(out_ch),                                            # .4
        nn.ReLU(inplace=True),                                             # .5
    )


class PersianCNN(nn.Module):
    """
    Exact architecture from scripts/train_cnn_classifier.py (PersianCharCNN).

    Forward pass spatial dimensions (input 32x32):
      pool(b1(x))  -> 16x16
      pool(b2(x))  ->  8x8
      pool(b3(x))  ->  4x4
      gap(b4(x))   ->  2x2  (256ch -> 1024 flat)
      head(x)      -> 42 logits
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        self.b1   = _conv_block(1,   32)
        self.b2   = _conv_block(32,  64)
        self.b3   = _conv_block(64,  128)
        self.b4   = _conv_block(128, 256)

        self.pool = nn.MaxPool2d(2, 2)
        self.gap  = nn.AdaptiveAvgPool2d((2, 2))

        self.head = nn.Sequential(
            nn.Flatten(),                 # .0
            nn.Dropout(0.4),              # .1
            nn.Linear(1024, 512),         # .2
            nn.ReLU(inplace=True),        # .3
            nn.Dropout(0.3),              # .4
            nn.Linear(512, num_classes),  # .5
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        x = self.pool(self.b1(x))
        x = self.pool(self.b2(x))
        x = self.pool(self.b3(x))
        x = self.gap(self.b4(x))
        return self.head(x)


# -- Main Class ----------------------------------------------------------------

class CNNClassifier:
    """
    Thin wrapper around PersianCNN for single-character inference.

    Parameters
    ----------
    weights_path : str
        Path to persian_cnn_classifier.pt
    confidence_threshold : float
        Global fallback threshold. Per-char overrides in CHAR_THRESHOLD_OVERRIDES
        take precedence when defined.
    input_size : tuple[int, int]
        (H, W) fed to the model. Default: (32, 32).
    class_map : dict[int, str] | None
        Override IDX2CHAR. If None, loads from checkpoint then falls back
        to hardcoded IDX2CHAR.
    """

    def __init__(
        self,
        weights_path: str,
        confidence_threshold: float = 0.4,
        input_size: tuple[int, int] = DEFAULT_INPUT_SIZE,
        class_map: dict[int, str] | None = None,
    ):
        self.weights_path   = weights_path
        self.conf_threshold = confidence_threshold
        self.input_size     = input_size
        self._model: Any    = None
        self._backend: str  = self._resolve_backend(weights_path)
        self._class_map: dict[int, str] = class_map or {}
        self._load_model()

    # -- Backend resolution ----------------------------------------------------

    @staticmethod
    def _resolve_backend(path: str) -> str:
        suffix = Path(path).suffix.lower()
        if suffix in (".pt", ".pth"):
            return "torch"
        if suffix == ".onnx":
            return "onnx"
        if suffix in (".h5", ".keras"):
            return "keras"
        raise ValueError(
            f"Unrecognised weight file extension '{suffix}'. "
            "Expected .pt/.pth, .onnx, or .h5/.keras"
        )

    # -- Model loading ---------------------------------------------------------

    def _load_model(self) -> None:
        path = self.weights_path

        if not Path(path).exists():
            raise FileNotFoundError(f"CNN weights not found: {path}")

        if self._backend == "torch":
            if not _TORCH_AVAILABLE:
                raise RuntimeError("torch not installed -- pip install torch")

            checkpoint = torch.load(path, map_location="cpu")

            if isinstance(checkpoint, torch.nn.Module):
                self._model = checkpoint
                if not self._class_map:
                    print("[CNNClassifier] WARNING: no idx2char in raw Module -- using fallback")
                    self._class_map = IDX2CHAR

            elif isinstance(checkpoint, dict):
                # Load class map from checkpoint
                if not self._class_map:
                    if "idx2char" in checkpoint:
                        self._class_map = {int(k): v for k, v in checkpoint["idx2char"].items()}
                        print(f"[CNNClassifier] Loaded idx2char from checkpoint ({len(self._class_map)} classes)")
                    elif "char2idx" in checkpoint:
                        self._class_map = {v: k for k, v in checkpoint["char2idx"].items()}
                        print(f"[CNNClassifier] Derived idx2char from char2idx ({len(self._class_map)} classes)")
                    else:
                        print("[CNNClassifier] WARNING: no idx2char in checkpoint -- using fallback IDX2CHAR")
                        self._class_map = IDX2CHAR

                # Extract state_dict
                if "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                elif "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                else:
                    state_dict = checkpoint

                tensor_keys = {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
                if not tensor_keys:
                    raise RuntimeError(
                        "Checkpoint contains no tensor weights.\n"
                        f"Keys: {list(checkpoint.keys())}"
                    )

                # Determine num_classes from checkpoint
                num_classes = NUM_CLASSES
                if "num_classes" in checkpoint:
                    num_classes = int(checkpoint["num_classes"])
                elif "head.5.weight" in state_dict:
                    num_classes = state_dict["head.5.weight"].shape[0]

                model = PersianCNN(num_classes=num_classes)
                missing, unexpected = model.load_state_dict(state_dict, strict=True)

                if missing or unexpected:
                    print("[CNNClassifier] strict=True failed -- retrying strict=False")
                    model = PersianCNN(num_classes=num_classes)
                    missing, unexpected = model.load_state_dict(state_dict, strict=False)
                    if missing:
                        print(f"[CNNClassifier] WARNING missing keys: {missing}")
                    if unexpected:
                        print(f"[CNNClassifier] WARNING unexpected keys: {unexpected}")
                else:
                    print("[CNNClassifier] State dict loaded perfectly (strict=True)")

                self._model = model

            else:
                raise RuntimeError(f"Unrecognised checkpoint type: {type(checkpoint)}")

            self._model.eval()
            print(
                f"[CNNClassifier] Ready -- {len(self._class_map)} classes, "
                f"input {self.input_size}, global threshold {self.conf_threshold}"
            )
            print(f"[CNNClassifier] Per-char overrides active: {len(CHAR_THRESHOLD_OVERRIDES)} chars")

        elif self._backend == "onnx":
            if not _ONNX_AVAILABLE:
                raise RuntimeError("onnxruntime not installed -- pip install onnxruntime")
            self._model       = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            self._input_name  = self._model.get_inputs()[0].name
            self._output_name = self._model.get_outputs()[0].name
            if not self._class_map:
                self._class_map = IDX2CHAR
            print(f"[CNNClassifier] Loaded ONNX model from {path}")

        elif self._backend == "keras":
            if not _KERAS_AVAILABLE:
                raise RuntimeError("tensorflow not installed -- pip install tensorflow")
            self._model = keras.models.load_model(path)
            if not self._class_map:
                self._class_map = IDX2CHAR
            print(f"[CNNClassifier] Loaded Keras model from {path}")

    # -- Preprocessing ---------------------------------------------------------

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        """
        Matches training preprocessing exactly:
          BGR -> gray -> resize(32,32) -> float32 -> [-1,1] normalize
        """
        h, w    = self.input_size
        resized = cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA)
        gray    = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        # Match training normalization: (x/255 - 0.5) / 0.5  ->  [-1, 1]
        norm    = (gray.astype(np.float32) / 255.0 - 0.5) / 0.5
        if self._backend in ("torch", "onnx"):
            return norm[np.newaxis, np.newaxis, :, :]   # (1, 1, H, W)
        else:
            return norm[np.newaxis, :, :, np.newaxis]   # (1, H, W, 1)

    # -- Threshold lookup ------------------------------------------------------

    def _effective_threshold(self, char: str) -> float:
        """Per-char override takes precedence over global threshold."""
        return CHAR_THRESHOLD_OVERRIDES.get(char, self.conf_threshold)

    # -- Single prediction -----------------------------------------------------

    def predict(self, crop: np.ndarray) -> tuple[str, float]:
        """
        Returns (char, confidence).

        Tries top-N candidates in descending probability order and returns
        the first one whose confidence meets its effective threshold.
        This handles cases where top-1 fails its threshold but top-2 is
        the correct character with sufficient confidence.

        Example: ع at prob=0.189 (thr=0.17) would be missed if \u062e wins
        top-1 at prob=0.238 but fails its own threshold of 0.4.
        """
        if crop is None or crop.size == 0:
            return UNKNOWN_CHAR, 0.0

        tensor = self._preprocess(crop)
        logits = self._infer(tensor)
        probs  = self._softmax(logits[0])

        # Sort all candidates by probability descending
        ranked = sorted(enumerate(probs), key=lambda x: -x[1])

        # Return first candidate that meets its own threshold
        for idx, conf in ranked:
            char = self._index_to_char(idx)
            if conf >= self._effective_threshold(char):
                return char, float(conf)

        # Nothing passed — return top-1 with its actual confidence
        top_idx, top_conf = ranked[0]
        return UNKNOWN_CHAR, float(top_conf)

    # -- Batch prediction ------------------------------------------------------

    def predict_batch(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        return [self.predict(c) for c in crops]

    # -- Backend inference -----------------------------------------------------

    def _infer(self, tensor: np.ndarray) -> np.ndarray:
        if self._backend == "torch":
            with torch.no_grad():
                t   = torch.from_numpy(tensor)
                out = self._model(t)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                return out.cpu().numpy()
        elif self._backend == "onnx":
            return self._model.run([self._output_name], {self._input_name: tensor})[0]
        elif self._backend == "keras":
            return self._model.predict(tensor, verbose=0)
        raise RuntimeError(f"Unknown backend: {self._backend}")

    # -- Helpers ---------------------------------------------------------------

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def _index_to_char(self, idx: int) -> str:
        return self._class_map.get(idx, UNKNOWN_CHAR)