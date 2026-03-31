# PersianOCR — TIP Manila · CIT 404A · 2026

> **Team:** Carbonel, Michael Jhon M. · Estacio, Ian Maru D. · Rosales, Andrei Matthew B.
> **Course:** Fundamentals of Prescriptive Analytics (CIT 404A / IT42S3)

End-to-end **Scene Text Recognition** system for Persian script.
Pipeline: **YOLOv8** (text detection) → **CRNN + CTC** (character recognition) → **Translation API** (Persian → English / Tagalog)

---

## Folder Structure

```
persian-ocr/
├── backend/                      # Flask API server
│   ├── run.py                    # Entry point: `python run.py`
│   ├── requirements.txt
│   ├── .env.example              # Copy to .env and fill values
│   └── app/
│       ├── __init__.py           # Flask app factory
│       ├── config.py             # All configuration (env-driven)
│       ├── routes/
│       │   ├── ocr.py            # POST /api/ocr/recognize
│       │   ├── translation.py    # POST /api/translate/
│       │   └── health.py         # GET  /api/health
│       ├── services/
│       │   ├── ocr_service.py    # YOLO + CRNN pipeline
│       │   └── translation_service.py  # LibreTranslate / Google / DeepL
│       ├── models/
│       │   ├── crnn.py           # CRNN architecture + CTC decoder
│       │   └── weights/          # ← put .pt files here after training
│       ├── middleware/
│       │   └── request_logger.py
│       └── utils/
│           ├── file_utils.py
│           └── persian_utils.py  # RTL sort, join, fix order
│
├── frontend/                     # React + Vite
│   ├── index.html
│   ├── vite.config.js            # /api/* proxied to Flask :5000
│   ├── package.json
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       ├── styles/global.css
│       ├── pages/
│       │   └── UploadPage.jsx    # Main page
│       ├── components/
│       │   ├── ImageDropzone.jsx
│       │   ├── LanguageSelector.jsx
│       │   ├── ResultsPanel.jsx
│       │   └── StatusBadge.jsx
│       ├── hooks/
│       │   └── useOcr.js         # OCR + translation state machine
│       └── services/
│           └── api.js            # Axios wrappers for all endpoints
│
├── scripts/
│   ├── train_yolo.py             # Fine-tune YOLOv8 on Persian text data
│   ├── train_crnn.py             # Train CRNN recognizer
│   └── evaluate.py              # Compute CER + Word Accuracy
│
└── docs/
    └── architecture.md           # (fill in your system diagram notes)
```

---

## Quick Start

### 1 — Backend

```bash
cd backend
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # edit translation provider + keys
python run.py                 # Flask starts on :5000
```

> **Note:** YOLOv8 and CRNN weights are not included. The server starts without them
> and returns stub/error responses until you place trained `.pt` files in
> `backend/models/weights/`.

### 2 — Frontend

```bash
cd frontend
npm install
npm run dev     # Vite dev server on :5173, /api/* proxied to Flask
```

Open http://localhost:5173

---

## Training

### Prepare data
- Download the **Shotor dataset** (~120k Persian word images).
- Annotate detection images with [LabelImg](https://github.com/heartexlabs/labelImg) or Roboflow in YOLO format.
- Split 80 / 10 / 10 (train / val / test).

### Train YOLOv8 detector
```bash
python scripts/train_yolo.py \
    --data  data/persian_text/dataset.yaml \
    --model yolov8s.pt \
    --epochs 80
```

### Train CRNN recognizer
```bash
python scripts/train_crnn.py \
    --data_dir  data/shotor/crops \
    --labels    data/shotor/train_labels.txt \
    --epochs    50
```

### Evaluate
```bash
python scripts/evaluate.py \
    --data_dir  data/shotor/crops \
    --labels    data/shotor/test_labels.txt \
    --crnn      backend/models/weights/persian_crnn.pt
```

---

## Translation Providers

Set `TRANSLATION_PROVIDER` in `.env`:

| Value            | Notes                                          |
|------------------|------------------------------------------------|
| `mock`           | No API key needed — returns placeholder text   |
| `libretranslate` | Free / self-hostable. Set `LIBRETRANSLATE_URL` |
| `google`         | Needs `GOOGLE_TRANSLATE_KEY`                   |
| `deepl`          | Needs `DEEPL_API_KEY`                          |

Adding **Tagalog** (`tl`) or any future language requires only a config change —
`SUPPORTED_TARGETS = ["en", "tl"]` — as long as your provider supports it.

---

## API Reference

| Method | Endpoint                | Description                                 |
|--------|-------------------------|---------------------------------------------|
| GET    | `/api/health`           | Liveness check + GPU info                   |
| POST   | `/api/ocr/recognize`    | Upload image → bounding boxes + Persian text |
| GET    | `/api/ocr/result/<id>`  | Fetch annotated result image                |
| POST   | `/api/translate/`       | `{text, target}` → `{translated}`           |
| GET    | `/api/translate/languages` | List supported target codes              |

---

## References

- Abbasi, M. (2025). *Persian OCR with YOLO + CRNN*. DEV Community.
- Gandomkar & Khoramipour (2025). *Omni-font OCR for Persian*. Majlesi J. EE.
- Graves et al. (2006). *CTC*. ICML.
- Shotor dataset — Asadi (2020).
