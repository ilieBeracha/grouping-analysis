from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.ultralytics import download_yolo11n_model
 
def init_detection_model(image_path):
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
    return detection_model, image_path

def process_image_sahi(image_path):
    try:
        sahi_model, image_path = init_detection_model(image_path)
        print('sahi_model', sahi_model)
        print('image_path', image_path)
        result = get_sliced_prediction(
            image_path,
            sahi_model,
            slice_height=200,
            slice_width=200,
            overlap_height_ratio=0.4,
            overlap_width_ratio=0.4,
        )
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

    return result


