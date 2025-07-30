from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from scipy.spatial.distance import pdist
import io
from typing import Optional, Tuple, List
import json
from processors.cv2_process import process_image, pdist
from processors.sahi_process import process_image_sahi, init_detection_model

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Grouping Analysis API - ScopeStats"}

@app.post("/upload/")
async def upload_image(bullet_type: str = "7.62mm", contrast: float = 2.0):
    test_image = "test.jpeg"
    with open(test_image, "rb") as f:
        contents = f.read()
        result = process_image(contents, bullet_type=bullet_type, contrast_factor=contrast)
        print('result', result)
        return JSONResponse(content=result, status_code=200)

@app.post("/upload/sahi")
async def upload_image_sahi():
    result = process_image_sahi("test.jpeg")
    if result is None:
        return JSONResponse(content={"error": "Failed to process image"}, status_code=500)
    
    # Extract relevant information from SAHI result
    detections = []
    for prediction in result.object_prediction_list:
        detections.append({
            "bbox": prediction.bbox.to_voc_bbox(),  # [xmin, ymin, xmax, ymax]
            "score": prediction.score.value,
            "category": prediction.category.name
        })
    
    return JSONResponse(content={
        "num_detections": len(detections),
        "detections": detections
    }, status_code=200)  