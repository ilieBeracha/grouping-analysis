from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from scipy.spatial.distance import pdist
import io

app = FastAPI()

def process_image(image_bytes):
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # 1. Convert to grayscale & threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 2)
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
    warped_blur = cv2.GaussianBlur(warped_gray, (5, 5), 0)
    

    circles = cv2.HoughCircles(
        warped_blur, cv2.HOUGH_GRADIENT, dp=1.1, minDist=30,
        param1=45, param2=25, minRadius=10, maxRadius=22
    )
    print('circles', circles)


    results = []
    if circles is not None:
        circles = np.uint16(np.around(circles[0]))
        for (x_c, y_c, r) in circles:
            results.append((int(x_c), int(y_c)))
            cv2.circle(warped, (x_c, y_c), r, (0, 255, 0), 2)

    # 5. Distance in cm
    if len(results) >= 2:
        dists = pdist(np.array(results))
        max_px = max(dists)
    else:
        max_px = 0

    pixels_per_cm = 40  # Adjust this based on resolution
    max_cm = round(max_px / pixels_per_cm, 2)

    # 6. Save result
    cv2.imwrite("annotated_output.jpg", warped)

    return {
        "num_hits": len(results),
        "max_distance_cm": max_cm,
        "hit_points": results
    }


@app.get("/")
async def root():
    return {"message": "Grouping Analysis API - ScopeStats"}

@app.post("/upload/")
async def upload_image():
    test_image = "test.jpeg"
    with open(test_image, "rb") as f:
        contents = f.read()
        result = process_image(contents)
        print('result', result)
        return JSONResponse(content=result, status_code=200)

# @app.post("/upload/")
# async def upload_image(file: UploadFile = File(...)):
#     contents = await file.read()
#     result = process_image(contents)
#     return JSONResponse(content=result)
