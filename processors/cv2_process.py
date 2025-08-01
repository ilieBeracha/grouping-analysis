from typing import Any, Dict, List, Tuple
import cv2
import numpy as np
import os

from helpers.image_setup import contrast_pct, dedup_circles, find_a4_or_none, order_points

def detect_bullets(
    img: np.ndarray,
    bullet_type: str = "7.62mm",
    contrast_factor: float = 2.0,
    max_hits: int = 6,
    debug_name: str = None,  
) -> list[tuple[int, int, int, float]]:
    """
    Improved bullet detection algorithm that better distinguishes holes from borders
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Step 1: Enhanced preprocessing
    # Use bilateral filter to smooth while preserving edges
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(bilateral)
    
    # Step 2: Edge detection to identify what to exclude
    edges = cv2.Canny(bilateral, 50, 150)
    edges_dilated = cv2.dilate(edges, np.ones((7,7), np.uint8), iterations=1)
    
    # Step 3: Multiple adaptive thresholding attempts for better coverage
    # Try multiple parameter sets and combine results
    adaptive_params = [
        (15, 5),   # More sensitive, smaller features
        (21, 8),   # Medium sensitivity
        (31, 10),  # Current parameters
    ]
    
    adaptive_combined = np.zeros_like(gray)
    
    for blockSize, C in adaptive_params:
        adaptive_temp = cv2.adaptiveThreshold(
            enhanced, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV,
            blockSize=blockSize,
            C=C
        )
        adaptive_combined = cv2.bitwise_or(adaptive_combined, adaptive_temp)
    
    # Use the combined result
    adaptive = adaptive_combined
    
    # Remove edge regions from adaptive threshold
    adaptive_no_edges = cv2.bitwise_and(adaptive, cv2.bitwise_not(edges_dilated))
    
    # Step 4: Morphological operations to clean up noise
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_circular = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    
    # Remove small noise
    cleaned = cv2.morphologyEx(adaptive_no_edges, cv2.MORPH_OPEN, kernel_small)
    
    # Fill small gaps in bullet holes
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_circular)
    
    # Step 5: Smart exclusion - identify printed targets vs bullet holes
    # Key insight: Printed targets are SOLID black, bullet holes have VARIED intensity
    
    # Find potential black areas - only VERY dark printed targets
    _, dark_areas = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    
    # Fill small gaps to connect printed graphics
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    dark_filled = cv2.morphologyEx(dark_areas, cv2.MORPH_CLOSE, kernel_close)
    
    # Find large connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark_filled, connectivity=8)
    
    exclusion_zone = np.zeros_like(gray)
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Check if it's a large dark region
        if area > 1500:  # Lower threshold
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            
            # Extract the region
            roi = gray[y:y+h, x:x+w]
            label_mask = (labels[y:y+h, x:x+w] == i)
            
            # Calculate uniformity - printed targets are very uniform
            masked_values = roi[label_mask]
            if len(masked_values) > 0:
                std_dev = np.std(masked_values)
                mean_val = np.mean(masked_values)
                
                # Printed targets: VERY low std dev (very uniform) and VERY dark
                # Bullet holes have variation due to torn paper
                if std_dev < 10 and mean_val < 50:
                    exclusion_zone[labels == i] = 255
    
    # Very minimal dilation - just catch the edges
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    exclusion_zone = cv2.dilate(exclusion_zone, edge_kernel, iterations=1)
    
    # Save debug images if requested
    if debug_name:
        DEBUG_DIR = "debug_output"
        os.makedirs(DEBUG_DIR, exist_ok=True)
        cv2.imwrite(f"{DEBUG_DIR}/{debug_name}_1_enhanced.jpg", enhanced)
        cv2.imwrite(f"{DEBUG_DIR}/{debug_name}_2_adaptive.jpg", adaptive)
        cv2.imwrite(f"{DEBUG_DIR}/{debug_name}_2b_edges.jpg", edges_dilated)
        cv2.imwrite(f"{DEBUG_DIR}/{debug_name}_2c_adaptive_no_edges.jpg", adaptive_no_edges)
        cv2.imwrite(f"{DEBUG_DIR}/{debug_name}_3_cleaned.jpg", cleaned)
        cv2.imwrite(f"{DEBUG_DIR}/{debug_name}_5_exclusion_zone.jpg", exclusion_zone)
    
    # Step 3: Find contours of potential bullet holes
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Keep track of detection sources
    contour_bullets = []    # Best results come from contours
    other_bullets = []      # Secondary detection methods
    
    if debug_name:
        print(f"\n[{debug_name}] Starting detection...")
        print(f"  Image shape: {img.shape}")
        print(f"  Found {len(contours)} total contours")
    
    # Debug: visualize all contours with ring detection info
    if debug_name:
        contour_vis = img.copy()
        ring_vis = img.copy()
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 50:  # Only show larger contours
                # Calculate properties
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0
                
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                # Determine rejection reason
                (x, y), radius = cv2.minEnclosingCircle(contour)
                r = int(radius)
                
                rect = cv2.minAreaRect(contour)
                width, height = rect[1]
                aspect_ratio = min(width, height) / max(width, height) if max(width, height) > 0 else 0
                
                # Check all criteria (matching the actual validation)
                size_ok = 5 <= r <= 25
                circ_ok = circularity > 0.3
                aspect_ok = aspect_ratio > 0.2
                solid_ok = 0.3 < solidity < 0.95
                
                # Determine color and label
                if not size_ok:
                    color = (255, 0, 0)  # Blue = wrong size
                    label = f"R{r}"
                elif not circ_ok:
                    color = (255, 0, 255)  # Magenta = not circular
                    label = f"C{circularity:.1f}"
                elif not solid_ok:
                    color = (0, 255, 255)  # Yellow = wrong solidity
                    label = f"S{solidity:.1f}"
                else:
                    color = (0, 255, 0)  # Green = good
                    label = "OK"
                
                cv2.drawContours(contour_vis, [contour], -1, color, 2)
                
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(contour_vis, label, (cx-10, cy+3), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    
                    # Show valid contours separately
                    if size_ok and circ_ok and aspect_ok and solid_ok:
                        cv2.drawContours(ring_vis, [contour], -1, (0, 255, 0), 2)
                        cv2.putText(ring_vis, f"{i}", (cx, cy), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        cv2.imwrite(f"{DEBUG_DIR}/{debug_name}_4_all_contours.jpg", contour_vis)
        cv2.imwrite(f"{DEBUG_DIR}/{debug_name}_5_ring_contours.jpg", ring_vis)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter by area (bullet holes have specific size range)
        # Balance between catching real holes and avoiding tiny marks
        if 50 < area < 3000:  # Much more permissive for contour detection
            # Get bounding circle
            (x, y), radius = cv2.minEnclosingCircle(contour)
            x, y, r = int(x), int(y), int(radius)
            
            # Skip if too close to edge
            if x < 20 or y < 20 or x > w - 20 or y > h - 20:
                continue
            
            # CRITICAL: Skip if in exclusion zone (printed targets, text, etc.)
            if exclusion_zone[y, x] > 0:
                continue
            
            # Check circularity (bullet holes are circular)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                # Also check aspect ratio to filter out elongated shapes (borders)
                rect = cv2.minAreaRect(contour)
                width, height = rect[1]
                aspect_ratio = min(width, height) / max(width, height) if max(width, height) > 0 else 0
                
                # Check if contour is closed (True means closed contour)
                is_closed = cv2.isContourConvex(contour) or len(contour) > 4
                
                # Check if it's a ring-like shape (not a solid blob)
                # Real bullet holes have a ring pattern, not solid fills
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0
                
                # STRICT validation for REAL bullet holes
                # Real bullets have very specific characteristics
                
                # Balanced validation - catch real bullets without false positives
                
                # More permissive validation for real bullets
                
                # Size validation - wider range to catch all bullets
                is_valid_size = 5 <= r <= 25  # Wider range
                
                # Shape validation - VERY permissive
                is_very_circular = circularity > 0.3  # Much lower threshold
                is_not_elongated = aspect_ratio > 0.2  # Very permissive
                
                # Ring pattern validation - very permissive
                is_ring_pattern = 0.3 < solidity < 0.95  # Almost any non-solid shape
                
                # Perimeter to area ratio - helps identify borders vs holes
                # Borders have high perimeter relative to area
                perimeter_ratio = perimeter / area if area > 0 else 999
                is_not_border = perimeter_ratio < 1.0  # More permissive
                
                # Debug logging
                if debug_name and r > 3:
                    print(f"  Checking ({x}, {y}): r={r}, size_ok={is_valid_size}, "
                          f"circ={circularity:.2f}>0.3={is_very_circular}, "
                          f"aspect={aspect_ratio:.2f}>0.2={is_not_elongated}, "
                          f"solid={solidity:.2f} in (0.3,0.95)={is_ring_pattern}, "
                          f"closed={is_closed}, border={perimeter_ratio:.2f}<1.0={is_not_border}")
                
                # All criteria must be met
                if (is_valid_size and is_very_circular and is_not_elongated and 
                    is_closed and is_ring_pattern and is_not_border):
                    # Calculate how dark the center is compared to surroundings
                    mask = np.zeros(gray.shape, dtype=np.uint8)
                    cv2.drawContours(mask, [contour], -1, 255, -1)
                    
                    # Get mean intensity inside the hole
                    mean_inside = cv2.mean(gray, mask=mask)[0]
                    
                    # Create ring mask around the hole
                    mask_ring = cv2.dilate(mask, kernel_medium) - mask
                    mean_ring = cv2.mean(gray, mask=mask_ring)[0]
                    
                    # Bullet holes are darker inside
                    if mean_inside < mean_ring * 0.82:  # At least 18% darker
                        contrast = contrast_pct(mean_inside, mean_ring)
                        
                        # Additional validation: more permissive
                        # Accept most potential bullet holes
                        if r >= 5 and contrast > 20:  # Lower requirements
                            contour_bullets.append((x, y, r, contrast, circularity))
    
    # Step 4: Use HoughCircles as a secondary detection method
    # This helps find bullets that might be missed by contour detection
    
    # Enhance for circle detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    
    # Try multiple parameter sets for better coverage
    param_sets = [
        # (dp, minDist, param1, param2, minR, maxR)
        (1.0, 20, 40, 15, 5, 25),    # More sensitive to smaller/fainter
        (1.2, 30, 50, 20, 8, 30),    # Medium sensitivity
        (1.5, 25, 30, 12, 10, 35),   # Larger circles
    ]
    
    for dp, minDist, p1, p2, minR, maxR in param_sets:
        circles = cv2.HoughCircles(
            blurred, 
            cv2.HOUGH_GRADIENT,
            dp=dp,
            minDist=minDist,
            param1=p1,
            param2=p2,
            minRadius=minR,
            maxRadius=maxR
        )
        
        if circles is not None:
            for (x, y, r) in circles[0]:
                x, y, r = int(x), int(y), int(r)
                
                # Skip edge detections
                if x < 20 or y < 20 or x > w - 20 or y > h - 20:
                    continue
                
                # Skip if in exclusion zone
                if exclusion_zone[y, x] > 0:
                    continue
                
                # Check if this circle corresponds to a dark region
                roi = gray[max(0, y-r):min(h, y+r), max(0, x-r):min(w, x+r)]
                if roi.size > 0:
                    # Check if center is dark
                    center_mask = np.zeros(gray.shape, dtype=np.uint8)
                    cv2.circle(center_mask, (x, y), max(1, r//2), 255, -1)
                    mean_center = cv2.mean(gray, mask=center_mask)[0]
                    
                    # Ring around
                    ring_mask = np.zeros(gray.shape, dtype=np.uint8)
                    cv2.circle(ring_mask, (x, y), min(r+5, r*2), 255, -1)
                    cv2.circle(ring_mask, (x, y), r, 0, -1)
                    mean_ring = cv2.mean(gray, mask=ring_mask)[0]
                    
                    if mean_center < mean_ring * 0.85:  # Dark center
                        contrast = contrast_pct(mean_center, mean_ring)
                        if contrast > 20:  # Basic quality check
                            other_bullets.append((x, y, r, contrast, 0.8))  # Assume good circularity
    
    # Step 5: Detect paper deformation around bullet holes
    # Real bullet holes have the paper curving/deforming around them
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
    gradient_norm = cv2.normalize(gradient_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Look for circular gradient patterns (paper curves)
    gradient_circles = cv2.HoughCircles(
        gradient_norm,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=35,
        param1=50,
        param2=30,
        minRadius=10,
        maxRadius=35
    )
    
    if gradient_circles is not None:
        for (x, y, r) in gradient_circles[0]:
            x, y, r = int(x), int(y), int(r)
            
            if 30 < x < w - 30 and 30 < y < h - 30 and exclusion_zone[y, x] == 0:
                # Verify dark center with gradient ring
                mask_center = np.zeros(gray.shape, dtype=np.uint8)
                cv2.circle(mask_center, (x, y), max(1, r//2), 255, -1)
                mean_center = cv2.mean(gray, mask=mask_center)[0]
                
                # Check gradient ring strength
                mask_ring = np.zeros(gradient_norm.shape, dtype=np.uint8)
                cv2.circle(mask_ring, (x, y), r, 255, 3)
                ring_gradient = cv2.mean(gradient_norm, mask=mask_ring)[0]
                
                if mean_center < 140 and ring_gradient > 30:  # Dark with strong gradient
                    contrast = contrast_pct(mean_center, 200)  # Assume white paper
                    other_bullets.append((x, y, r, contrast, 0.85))
    
    # Step 6: Additional detection using local minima (dark spots)
    # This helps find holes that might be missed by contour detection
    blurred_for_minima = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Find local minima using morphological operations
    kernel_minima = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    local_min = cv2.morphologyEx(blurred_for_minima, cv2.MORPH_ERODE, kernel_minima)
    local_max = cv2.morphologyEx(blurred_for_minima, cv2.MORPH_DILATE, kernel_minima)
    
    # Points that are significantly darker than their surroundings
    darkness_diff = local_max - local_min
    is_minimum = (blurred_for_minima - local_min) < 10
    is_significant = darkness_diff > 30
    
    dark_points = np.logical_and(is_minimum, is_significant).astype(np.uint8) * 255
    
    # Find contours of these dark regions
    dark_contours, _ = cv2.findContours(dark_points, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in dark_contours:
        area = cv2.contourArea(contour)
        if 50 < area < 3000:  # More permissive size range
            (x, y), radius = cv2.minEnclosingCircle(contour)
            x, y, r = int(x), int(y), int(radius)
            
            if 5 < r < 35 and 20 < x < w - 20 and 20 < y < h - 20 and exclusion_zone[y, x] == 0:
                # Verify it's actually dark
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.circle(mask, (x, y), r, 255, -1)
                mean_inside = cv2.mean(gray, mask=mask)[0]
                
                # Compare with surrounding
                mask_ring = np.zeros(gray.shape, dtype=np.uint8)
                cv2.circle(mask_ring, (x, y), min(r + 10, r * 2), 255, -1)
                cv2.circle(mask_ring, (x, y), r, 0, -1)
                mean_outside = cv2.mean(gray, mask=mask_ring)[0]
                
                if mean_inside < mean_outside * 0.75:  # At least 25% darker
                    contrast = contrast_pct(mean_inside, mean_outside)
                    if contrast > 30:
                        other_bullets.append((x, y, r, contrast, 0.8))
    
    # Step 7: Combine results - PRIORITIZE CONTOUR DETECTIONS
    # First add all contour-based detections (they're the most reliable)
    final_bullets = []
    
    # Add contour bullets first
    for bullet in contour_bullets:
        x, y, r, contrast, circ = bullet
        final_bullets.append((x, y, r, contrast))
    
    # Only add other detection methods if we need more bullets
    if len(final_bullets) < max_hits:
        for bullet in other_bullets:
            x, y, r, contrast, circ = bullet
            
            # Check if too close to existing detection
            is_duplicate = False
            for fx, fy, fr, fc in final_bullets:
                dist = np.sqrt((x - fx)**2 + (y - fy)**2)
                if dist < max(r, fr) * 1.5:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                final_bullets.append((x, y, r, contrast))
    
    # Sort by contrast (best bullets first)
    final_bullets.sort(key=lambda b: b[3], reverse=True)
    
    # Quality filter - prioritize REAL bullet holes (large with good contrast)
    quality_bullets = []
    
    # First pass: Add all bullets that are clearly real (proper size and contrast)
    for x, y, r, contrast in final_bullets:
        if r >= 12 and contrast > 35:  # Large holes with good contrast
            quality_bullets.append((x, y, r, contrast))
    
    # Second pass: Add medium-sized holes with excellent contrast
    if len(quality_bullets) < max_hits:
        for x, y, r, contrast in final_bullets:
            if (x, y, r, contrast) not in quality_bullets:
                if r >= 10 and contrast > 50:  # Medium but very dark
                    quality_bullets.append((x, y, r, contrast))
    
    # Third pass: Add any remaining holes that meet minimum criteria
    if len(quality_bullets) < max_hits:
        for x, y, r, contrast in final_bullets:
            if (x, y, r, contrast) not in quality_bullets:
                if r >= 8 and contrast > 25:  # Minimum acceptable size
                    quality_bullets.append((x, y, r, contrast))
                    if len(quality_bullets) >= max_hits:
                        break
    
    return quality_bullets[:max_hits]


# --------------------------------------------------------------------------- #
# 2.  Main pipeline                                                           #
# --------------------------------------------------------------------------- #
def process_image(image_bytes: bytes,
                  bullet_type: str = "7.62mm",
                  contrast_factor: float = 2.0) -> Dict[str, Any]:
    """
    Improved image processing pipeline
    """
    img = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Invalid image data"}
    
    # Try to detect and warp A4 sheet
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pts = find_a4_or_none(gray)
    
    if pts is not None:
        # Warp to get rectangular view
        rect = order_points(pts)
        w = int(max(np.linalg.norm(rect[2] - rect[3]),
                    np.linalg.norm(rect[1] - rect[0])))
        h = int(max(np.linalg.norm(rect[1] - rect[2]),  
                    np.linalg.norm(rect[0] - rect[3])))
        
        # Ensure minimum size
        w = max(w, 800)
        h = max(h, 600)
        
        M = cv2.getPerspectiveTransform(
            rect, np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype="float32"))
        sheet = cv2.warpPerspective(img, M, (w, h))
    else:
        sheet = img.copy()
        h, w = sheet.shape[:2]
    
    # Split into quadrants
    mid_x, mid_y = w // 2, h // 2
    quarters = {
        "top_left":     {"bounds": (0, 0, mid_x, mid_y), "hits": []},
        "top_right":    {"bounds": (mid_x, 0, w, mid_y), "hits": []},
        "bottom_left":  {"bounds": (0, mid_y, mid_x, h), "hits": []},
        "bottom_right": {"bounds": (mid_x, mid_y, w, h), "hits": []},
    }
    
    # Process each quadrant separately for better accuracy
    DEBUG_DIR = "debug_output"
    os.makedirs(DEBUG_DIR, exist_ok=True)
    
    output: Dict[str, Any] = {}
    
    for name, data in quarters.items():
        x1, y1, x2, y2 = data["bounds"]
        quadrant_img = sheet[y1:y2, x1:x2]
        
        # Detect bullets in this quadrant
        hits = detect_bullets(quadrant_img, bullet_type, contrast_factor, max_hits=6, debug_name=name)
        
        # Adjust coordinates to full sheet coordinates
        adjusted_hits = []
        for (x, y, r, contrast) in hits[:6]:  # Keep only top 6 per quadrant
            adj_x = x + x1
            adj_y = y + y1
            # Validate that the bullet is within the sheet boundaries
            # Also exclude detections very close to sheet edges (often artifacts)
            edge_margin = 30  # pixels from sheet edge
            if (edge_margin < adj_x < w - edge_margin) and (edge_margin < adj_y < h - edge_margin):
                adjusted_hits.append((adj_x, adj_y, r, contrast))
        
        # Draw visualization
        vis_img = quadrant_img.copy()
        for i, (x, y, r, contrast) in enumerate(hits[:6]):
            # Draw circle
            cv2.circle(vis_img, (x, y), r, (0, 255, 0), 2)
            cv2.circle(vis_img, (x, y), 2, (0, 0, 255), -1)
            
            # Add label with number and contrast
            label = f"{i+1}: {contrast:.0f}%"
            cv2.putText(vis_img, label, (x + r + 5, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Save debug image
        path = f"{DEBUG_DIR}/{name}.jpg"
        cv2.imwrite(path, vis_img)
        
        # Prepare output
        output[name] = {
            "hits": len(adjusted_hits),
            "points": [(int(x), int(y), round(contrast, 1)) 
                      for (x, y, r, contrast) in adjusted_hits],
            "image_file": path,
        }
    
    # Also save full sheet with all detections for debugging
    full_vis = sheet.copy()
    total_hits = 0
    for name, data in output.items():
        for i, (x, y, contrast) in enumerate(data["points"]):
            cv2.circle(full_vis, (x, y), 12, (0, 255, 0), 2)
            cv2.putText(full_vis, name.split('_')[0][0].upper() + 
                       name.split('_')[1][0].upper() + str(i+1), 
                       (x-15, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            total_hits += 1
    
    cv2.imwrite(f"{DEBUG_DIR}/full_sheet.jpg", full_vis)
    
    return output
