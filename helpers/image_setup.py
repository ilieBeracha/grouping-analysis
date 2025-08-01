import cv2
import numpy as np

def dedup_circles(circles: list[tuple[int, int, int, float]],
                  min_gap: int = 15) -> list[tuple[int, int, int, float]]:
    """Non-max-suppression on circle centres."""
    kept: list[tuple[int, int, int, float]] = []
    for c in circles:
        if all((c[0] - k[0]) ** 2 + (c[1] - k[1]) ** 2 > min_gap ** 2 for k in kept):
            kept.append(c)
    return kept

def contrast_pct(center_val: float, ring_val: float) -> float:
    """
    Percentage darkness drop from ring to centre, 0‑100.
    Higher == darker hole relative to background.
    """
    return max(0.0, min(100.0, (ring_val - center_val) / ring_val * 100.0))


def find_grid_lines(gray_img: np.ndarray) -> np.ndarray:
    """Return x‑positions of the vertical grid lines (one per square)."""
    sobelx = cv2.Sobel(gray_img, cv2.CV_16S, 1, 0, ksize=3)
    absx   = cv2.convertScaleAbs(sobelx)
    _, bw  = cv2.threshold(absx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # sum each column; peaks correspond to vertical lines
    col_sum = bw.sum(axis=0)
    peaks   = np.where(col_sum > 0.6 * col_sum.max())[0]   # adjust 0.6 if needed
    return peaks

def find_a4_or_none(gray: np.ndarray) -> np.ndarray | None:
    H, W = gray.shape
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, None, iterations=2)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    for c in cnts:
        area = cv2.contourArea(c)
        if area < 0.10 * H * W:              # at least 10 % of frame
            break
        peri   = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.015 * peri, True)
        if len(approx) != 4:
            continue
        return approx.reshape(4, 2).astype(np.float32)

    return None

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect