import modal
import numpy as np
from starlette.requests import Request

# 1. Define the Modal App
app = modal.App("weaponscan-inference")

# 2. Define the container environment and mount local best.pt weights
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("ultralytics", "opencv-python-headless", "numpy", "fastapi")
    .add_local_file("models/best.pt", "/root/best.pt")
)

_model = None

def get_modal_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        print("[MODAL CONTAINER] Loading YOLO model from /root/best.pt...", flush=True)
        _model = YOLO("/root/best.pt")
    return _model

@app.function(image=image, cpu=2.0)
@modal.fastapi_endpoint(method="POST")
async def predict(request: Request):
    """
    Serverless Web Endpoint on Modal using native PyTorch best.pt model.
    Receives raw JPEG image bytes in POST body, runs YOLO inference on best.pt,
    and returns detected bounding boxes as JSON with exact trained class labels.
    """
    import cv2

    image_bytes = await request.body()

    # Decode JPEG bytes into OpenCV image array
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"error": "Invalid image payload"}, 400

    # Get cached YOLO model
    model = get_modal_model()

    # Run inference
    results = model.predict(
        frame,
        conf=0.45,  # Confidence threshold
        iou=0.45,   # NMS IoU threshold
        imgsz=640,
        verbose=False,
    )

    detections = []
    boxes = results[0].boxes

    if len(boxes) > 0:
        for box in boxes:
            class_id = int(box.cls[0])
            label = model.names[class_id] # Dynamically gets 'guns' or 'knife'
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "label": label,
                "confidence": confidence,
                "bbox": [float(x1), float(y1), float(x2), float(y2)]
            })

    return {"detections": detections}
