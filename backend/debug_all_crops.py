"""
Check top-3 predictions for all current debug crops,
focusing on chars that are returning '?' at high confidence.
"""
import cv2
import torch
import numpy as np
import os
from app.models.cnn_classifier import CNNClassifier, IDX2CHAR, CHAR_THRESHOLD_OVERRIDES

cnn = CNNClassifier('models/weights/persian_cnn_classifier.pt')

print(f"Global threshold: {cnn.conf_threshold}")
print(f"Overrides: {CHAR_THRESHOLD_OVERRIDES}\n")
print(f"{'FILE':<45} {'TOP-1':>6} {'CONF':>6}  TOP-3")
print("-" * 90)

for f in sorted(os.listdir('debug_crops')):
    if not f.endswith('.png') or 'model_view' in f or 'padded' in f:
        continue

    img = cv2.imread(f'debug_crops/{f}')
    if img is None:
        continue

    # Get raw probabilities
    tensor = cnn._preprocess(img)
    with torch.no_grad():
        t = torch.from_numpy(tensor)
        out = cnn._model(t)
    probs = torch.softmax(out[0], dim=0).numpy()

    top3 = sorted(enumerate(probs), key=lambda x: -x[1])[:3]
    top1_char = IDX2CHAR.get(top3[0][0], '?')
    top1_conf = top3[0][1]

    # Effective threshold for top-1 char
    eff_thresh = CHAR_THRESHOLD_OVERRIDES.get(top1_char, cnn.conf_threshold)
    result = top1_char if top1_conf >= eff_thresh else '?'

    try:
        expected = f.split('_cls')[1].split('_')[1]
    except:
        expected = '?'

    status = 'OK' if result == expected else ('MISS' if result == '?' else 'WRONG')

    top3_str = '  |  '.join(
        f"{IDX2CHAR.get(i,'?')}={p:.3f}(thr={CHAR_THRESHOLD_OVERRIDES.get(IDX2CHAR.get(i,'?'), cnn.conf_threshold):.2f})"
        for i, p in top3
    )
    print(f"[{status:<5}] exp={expected:2s} got={result!r:4s}  {f[:42]:<42}  {top3_str}")