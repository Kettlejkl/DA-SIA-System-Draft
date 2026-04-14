"""
persian_utils.py
────────────────
Helpers for right-to-left Persian text post-processing.
"""

from __future__ import annotations


def sort_boxes_rtl(
    detections: list[tuple[list[int], float]]
) -> list[tuple[list[int], float]]:
    """
    Sort bounding boxes in Persian reading order (right-to-left, top-to-bottom).
    Groups boxes by approximate row (y-coordinate), then reverses x within each row.
    """
    if not detections:
        return detections

    # Determine row grouping tolerance: 50% of median box height
    heights  = [abs(b[3] - b[1]) for b, _ in detections]
    median_h = sorted(heights)[len(heights) // 2]
    tol      = median_h * 0.5

    # Sort by top-y first
    sorted_d = sorted(detections, key=lambda d: d[0][1])

    rows: list[list[tuple[list[int], float]]] = []
    for det in sorted_d:
        placed = False
        for row in rows:
            if abs(det[0][1] - row[0][0][1]) < tol:
                row.append(det)
                placed = True
                break
        if not placed:
            rows.append([det])

    result = []
    for row in rows:
        # RTL: sort by x descending (rightmost first)
        row_sorted = sorted(row, key=lambda d: d[0][0], reverse=True)
        result.extend(row_sorted)

    return result


def fix_rtl_order(text: str) -> str:
    """
    Reverse a string that was scanned left-to-right back to RTL order.
    (Used when the CRNN outputs characters in scan order.)
    """
    return text[::-1]


def join_persian_words(words: list[str]) -> str:
    """Join recognised word strings with a space (already in RTL order)."""
    return " ".join(w for w in words if w)
