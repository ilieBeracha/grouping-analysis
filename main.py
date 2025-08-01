from fastapi import FastAPI, UploadFile, File   
from fastapi.responses import JSONResponse
import base64
import os
from fastapi.middleware.cors import CORSMiddleware

from processors.cv2_process import process_image

app = FastAPI({
    "title": "Bullet detection API",
    "description": "API for detecting bullet holes in images",
    "version": "0.1.0",
    "contact": {
        "name": "John Doe",
        "email": "john.doe@example.com"
    }

})

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Bullet detection API is running!", "status": "ok"}

def encode_img(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...), bullet_type: str = "7.62mm", contrast: float = 2.0):
    img = await file.read()
    
    result = process_image(img, bullet_type=bullet_type, contrast_factor=contrast)
    print(result)
    for key in result:
        path = result[key]['image_file']
        if os.path.exists(path):
            result[key]['preview_base64'] = encode_img(path)

    return JSONResponse(content=result, status_code=200)
