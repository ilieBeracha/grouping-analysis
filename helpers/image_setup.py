import cv2
import numpy as np


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
