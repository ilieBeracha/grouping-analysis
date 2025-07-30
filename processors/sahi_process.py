from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.ultralytics import download_yolo11n_model
import cv2
import numpy as np
 
def init_detection_model(img):
    try:
        model_path = "models/yolo11n.pt"
        download_yolo11n_model(model_path)
        detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=model_path,
            confidence_threshold=0.3,
            device="cuda:0",  # or 'cuda:0'
    )
    except Exception as e:
        print(f"Error initializing detection model: {e}")
        return None
    return detection_model, img

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.ultralytics import download_yolo11n_model
import cv2
import numpy as np

def process_image_sahi(image_bytes: bytes):
    # Decode image
    img_np = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image format")

    # Load detection model
    model_path = "models/yolo11n.pt"
    try:
        download_yolo11n_model(model_path)
        detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=model_path,
            confidence_threshold=0.3,
            device="cuda:0",  # or 'cpu'
        )
    except Exception as e:
        print(f"[ERROR] Model load failed: {e}")
        return None

    # Run SAHI sliced prediction
    try:
        result = get_sliced_prediction(
            image=img,
            detection_model=detection_model,
            slice_height=250,
            slice_width=250,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
        )
        result.export_visuals(export_dir="sahi_res")
        print(f"[INFO] Detected {len(result.object_prediction_list)} bullet holes")
        return result
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return None
