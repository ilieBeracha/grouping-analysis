import cv2
import numpy as np
from scipy.spatial.distance import pdist
from helpers.image_setup import contrast_pct, find_grid_lines


def process_image(image_bytes, bullet_type: str = "7.62mm", contrast_factor: float = 2.0):
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # 1. Convert to grayscale & threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11,   1)
    print('thresh', thresh)

    # 2. Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    page_rect = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 100000:
            page_rect = approx
            break

    if page_rect is None:
        return {"error": "Paper not found"}

    # 3. Warp the perspective to crop just the paper
    pts = page_rect.reshape(4, 2).astype(np.float32)

    # Sort points in consistent order: top-left, top-right, bottom-right, bottom-left
    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=contrast_factor, tileGridSize=(8,8))
    enhanced = clahe.apply(warped_gray)

    # ------------------------------------------------------------------
    # ⇣ INSERT GRID‑BASED CALIBRATION HERE ⇣
    # ------------------------------------------------------------------
    grid_cols    = np.sort(np.unique(find_grid_lines(enhanced)))
    print('grid_cols', grid_cols)
    if len(grid_cols) > 1:                                     # we found the grid
        grid_size_px = int(np.median(np.diff(grid_cols)))      # pixels per square (≈10 mm)
        px_per_mm    = grid_size_px / 10.0
        diameter_mm  = 7.62 if bullet_type == "7.62mm" else 8.6
        min_radius   = int(px_per_mm * diameter_mm * 0.35)     # 0.7 × r
        max_radius   = int(px_per_mm * diameter_mm * 0.75)     # 1.5 × r
    else:
        # fallback to your previous fixed values
        min_radius, max_radius = (12, 18) if bullet_type == "7.62mm" else (14, 20)

    # Different blur for edge detection vs circle detection
    edge_blur = cv2.GaussianBlur(enhanced, (3, 3), 0)
    circle_blur = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # Adaptive parameters based on bullet type
    if bullet_type == "7.62mm":
        min_radius = 11  # ~30.5 pixels / 2.5
        max_radius = 17  # ~30.5 pixels / 1.7
        min_dist = 25
        param2 = 16  #6Lower for better detection of faint holes
    else:  # .338 inch (8.6mm)
        min_radius = 14  # ~34.4 pixels / 2.5
        max_radius = 20  # ~34.4 pixels / 1.7
        min_dist = 28
    
    # Try multiple param1 values for better detection
    all_circles = []
    for param1 in [50, 70, 90]:
        circles = cv2.HoughCircles(
            edge_blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min_dist,
            param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius
        )
        if circles is not None:
            all_circles.extend(circles[0])
    
    # Remove duplicates by checking proximity
    if all_circles:
        all_circles = np.array(all_circles)
        unique_circles = []
        for circle in all_circles:
            is_duplicate = False
            for unique in unique_circles:
                dist = np.linalg.norm(circle[:2] - unique[:2])
                if dist < min_dist * 0.8:  # 80% of min distance
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_circles.append(circle)
        circles = np.array(unique_circles).reshape(1, -1, 3) if unique_circles else None
    else:
        circles = None
    # Additional validation - check for dark centers (actual holes)
    validated_circles = []
    if circles is not None:
        for (x_c, y_c, r) in circles[0]:
            
            x_c, y_c, r = int(x_c), int(y_c), int(r)
            # Create a mask for the circle area
            mask = np.zeros(enhanced.shape, dtype=np.uint8)
            cv2.circle(mask, (x_c, y_c), int(r * 0.7), 255, -1)
            
            # Check if the center is darker than the surroundings
            mean_center = cv2.mean(enhanced, mask=mask)[0]
            
            
            # Create ring mask for outer area
            ring_mask = np.zeros(enhanced.shape, dtype=np.uint8)
            cv2.circle(ring_mask, (x_c, y_c), int(r * 1.5), 255, -1)
            cv2.circle(ring_mask, (x_c, y_c), r, 0, -1)
            mean_ring = cv2.mean(enhanced, mask=ring_mask)[0]
            
            # Bullet holes should have darker centers
            if mean_center < mean_ring * 0.85:      # <-- keep your test
                pct = contrast_pct(mean_center, mean_ring)
                validated_circles.append([x_c, y_c, r, pct])
    
    print(f'Total circles detected: {len(circles[0]) if circles is not None else 0}')
    print(f'Validated circles: {len(validated_circles)}')
    
    results = []
    if validated_circles:
        for (x_c, y_c, r, pct) in validated_circles:
            results.append((int(x_c), int(y_c), round(pct, 1)))
            cv2.circle(warped, (x_c, y_c), int(r), (0, 255, 0), 3)
            cv2.circle(warped, (x_c, y_c), 2, (0, 0, 255), -1)
            # draw percentage label
            cv2.putText(
                warped,
                f"{pct:.0f}%",                # e.g. “32%”
                (x_c + int(r*0.2), y_c - int(r*0.2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,                          # font scale
                (0, 255, 0), 1, cv2.LINE_AA
            )

    # 5. Distance in cm
    if len(results) >= 1:
        dists = pdist(np.array([pt[:2] for pt in results]))   # use x,y only
        max_px = max(dists)
    else:
        max_px = 0

    pixels_per_cm = 40  # Adjust this based on resolution
    max_cm = round(max_px / pixels_per_cm, 2)

    # Save result and debug images
    cv2.imwrite("annotated_output.jpg", warped)
    cv2.imwrite("enhanced_debug.jpg", enhanced)
    
    # Save threshold debug for analysis
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite("thresh_debug.jpg", binary)

    return {
        "num_hits": len(results),
        "max_distance_cm": max_cm,
        "hit_points": results
    }

