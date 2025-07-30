import cv2
import numpy as np
import os

from helpers.image_setup import contrast_pct

def detect_bullets(img, bullet_type, contrast_factor):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=contrast_factor, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    if bullet_type == "7.62mm":
        min_radius, max_radius, param2, min_dist = 11, 17, 16, 25
    else:
        min_radius, max_radius, param2, min_dist = 14, 20, 16, 28

    edge_blur = cv2.GaussianBlur(enhanced, (3, 3), 0)
    validated = []

    for param1 in [50, 70, 90]:
        circles = cv2.HoughCircles(
            edge_blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min_dist,
            param1=param1, param2=param2,
            minRadius=min_radius, maxRadius=max_radius
        )
        if circles is not None:
            for (x_c, y_c, r) in circles[0]:
                x_c, y_c, r = int(x_c), int(y_c), int(r)
                mask = np.zeros(enhanced.shape, dtype=np.uint8)
                cv2.circle(mask, (x_c, y_c), int(r * 0.7), 255, -1)
                mean_center = cv2.mean(enhanced, mask=mask)[0]

                ring_mask = np.zeros(enhanced.shape, dtype=np.uint8)
                cv2.circle(ring_mask, (x_c, y_c), int(r * 1.5), 255, -1)
                cv2.circle(ring_mask, (x_c, y_c), r, 0, -1)
                mean_ring = cv2.mean(enhanced, mask=ring_mask)[0]

                if mean_center < mean_ring * 0.85:
                    pct = contrast_pct(mean_center, mean_ring)
                    validated.append((x_c, y_c, r, pct))
    return validated


def process_image(image_bytes, bullet_type="7.62mm", contrast_factor=2.0):
    np_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img = cv2.imdecode(np_bytes, cv2.IMREAD_COLOR)

    # Find paper
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        cv2.GaussianBlur(gray, (5, 5), 0), 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 1
    )
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 100000:
            pts = approx.reshape(4, 2).astype(np.float32)
            break
    else:
        return {"error": "Paper not found"}

    # Warp
    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s, d = pts.sum(axis=1), np.diff(pts, axis=1)
        rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
        rect[1], rect[3] = pts[np.argmin(d)], pts[np.argmax(d)]
        return rect

    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))

    M = cv2.getPerspectiveTransform(rect, np.array([
        [0, 0], [width - 1, 0],
        [width - 1, height - 1], [0, height - 1]
    ], dtype="float32"))
    warped = cv2.warpPerspective(img, M, (width, height))

    # Split quarters
    h, w = warped.shape[:2]
    quarters = {
        "top_left": warped[0:h//2, 0:w//2],
        "top_right": warped[0:h//2, w//2:w],
        "bottom_left": warped[h//2:h, 0:w//2],
        "bottom_right": warped[h//2:h, w//2:w],
    }

    os.makedirs("debug_output", exist_ok=True)
    output = {}
    for name, sub_img in quarters.items():
        hits = detect_bullets(sub_img, bullet_type, contrast_factor)
        for (x, y, r, pct) in hits:
            cv2.circle(sub_img, (x, y), int(r), (0, 255, 0), 2)
            cv2.circle(sub_img, (x, y), 2, (0, 0, 255), -1)
            cv2.putText(sub_img, f"{pct:.0f}%", (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        file_path = f"debug_output/{name}.jpg"
        cv2.imwrite(file_path, sub_img)
        output[name] = {
            "hits": len(hits),
            "points": [(int(x), int(y), round(pct, 1)) for (x, y, _, pct) in hits],
            "image_file": file_path
        }

    return output
