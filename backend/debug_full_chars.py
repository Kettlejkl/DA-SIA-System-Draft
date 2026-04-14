"""
Full character accuracy report for all 42 Persian/Arabic classes.
Run from: C:\Final Project\DA-SIA-System-Draft-main\backend
Usage: py debug_full_chars.py
"""
import cv2
import torch
import numpy as np
import os
from collections import defaultdict
from app.models.cnn_classifier import CNNClassifier, IDX2CHAR, CHAR_THRESHOLD_OVERRIDES

cnn = CNNClassifier('models/weights/persian_cnn_classifier.pt')

# CHAR2IDX for reverse lookup
CHAR2IDX = {v: k for k, v in IDX2CHAR.items()}

# Collect results per expected character
results = defaultdict(lambda: {'ok': 0, 'wrong': 0, 'miss': 0, 'total': 0, 'wrongs': [], 'misses': []})

print("Processing debug crops...\n")

for f in sorted(os.listdir('debug_crops')):
    if not f.endswith('.png'):
        continue
    if any(x in f for x in ['model_view', 'padded', 'z_padded', 'test_crop']):
        continue

    img = cv2.imread(f'debug_crops/{f}')
    if img is None:
        continue

    # Parse expected char from filename: crop_XX_x_y_clsNN_CHAR_form.png
    try:
        parts = f.split('_cls')[1]          # e.g. "60_ش_isolated.png"
        char_part = parts.split('_')[1]     # e.g. "ش"
        expected = char_part
    except:
        continue

    if expected not in CHAR2IDX:
        continue

    char, conf = cnn.predict(img)
    results[expected]['total'] += 1

    if char == expected:
        results[expected]['ok'] += 1
    elif char == '?':
        results[expected]['miss'] += 1
        results[expected]['misses'].append((conf, f))
    else:
        results[expected]['wrong'] += 1
        results[expected]['wrongs'].append((char, conf, f))

# Print per-character summary
print(f"{'CHAR':<6} {'NAME':<12} {'TOTAL':>5} {'OK':>4} {'MISS':>5} {'WRONG':>6} {'ACC%':>6}  ISSUES")
print("-" * 100)

CHAR_NAMES = {
    'ا':'alef', 'ب':'be', 'پ':'pe', 'ت':'te', 'ث':'se',
    'ج':'jim', 'چ':'che', 'ح':'he', 'خ':'khe', 'د':'dal',
    'ذ':'zal', 'ر':'re', 'ز':'ze', 'ژ':'zhe', 'س':'sin',
    'ش':'shin', 'ص':'sad', 'ض':'zad', 'ط':'ta', 'ظ':'za',
    'ع':'eyn', 'غ':'gheyn', 'ف':'fe', 'ق':'ghaf', 'ک':'kaf',
    'گ':'gaf', 'ل':'lam', 'م':'mim', 'ن':'nun', 'و':'waw',
    'ه':'he2', 'ی':'ye',
    '۰':'0', '۱':'1', '۲':'2', '۳':'3', '۴':'4',
    '۵':'5', '۶':'6', '۷':'7', '۸':'8', '۹':'9',
}

total_ok = total_miss = total_wrong = total_all = 0

for char in sorted(results.keys(), key=lambda c: CHAR2IDX.get(c, 99)):
    r = results[char]
    if r['total'] == 0:
        continue
    acc = 100 * r['ok'] / r['total']
    name = CHAR_NAMES.get(char, '?')
    thr = CHAR_THRESHOLD_OVERRIDES.get(char, cnn.conf_threshold)

    issues = []
    for wrong_char, conf, fname in r['wrongs']:
        issues.append(f"WRONG->{wrong_char}({conf:.2f})")
    for conf, fname in r['misses']:
        issues.append(f"MISS({conf:.2f})")

    issue_str = '  '.join(issues[:3])  # show max 3
    flag = '' if acc == 100 else ('  [MODEL ISSUE]' if r['wrong'] > 0 else '  [THRESHOLD?]')

    print(f"  {char:<4} {name:<12} {r['total']:>5} {r['ok']:>4} {r['miss']:>5} {r['wrong']:>6} {acc:>5.0f}%  thr={thr:.2f}  {issue_str}{flag}")

    total_ok    += r['ok']
    total_miss  += r['miss']
    total_wrong += r['wrong']
    total_all   += r['total']

print("-" * 100)
print(f"  {'TOTAL':<16} {total_all:>5} {total_ok:>4} {total_miss:>5} {total_wrong:>6} {100*total_ok/total_all:>5.0f}%")
print()
print(f"Overall: {total_ok}/{total_all} correct ({100*total_ok/total_all:.1f}%)")
print(f"  Missed (returned ?): {total_miss}")
print(f"  Wrong char returned: {total_wrong}")
print()

# Characters with no test crops
tested = set(results.keys())
all_chars = set(IDX2CHAR.values()) - set('۰۱۲۳۴۵۶۷۸۹')
untested = all_chars - tested
if untested:
    print(f"Characters with NO test crops ({len(untested)}): {' '.join(sorted(untested, key=lambda c: CHAR2IDX.get(c,99)))}")
    print("  Add images containing these characters to get full coverage.")