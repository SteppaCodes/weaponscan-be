import modal
import numpy as np

# 1. Define the Modal App
app = modal.App("weaponscan-inference")

# 2. Define the container environment and mount local ONNX weights
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("onnxruntime", "opencv-python-headless", "numpy", "fastapi")
    .add_local_file("models/best.onnx", "/root/best.onnx")
)

from starlette.requests import Request

@app.function(image=image, cpu=2.0)
@modal.fastapi_endpoint(method="POST")
async def predict(request: Request):
    """
    Serverless Web Endpoint on Modal.
    Receives raw JPEG image bytes in POST body, runs ONNX Runtime inference on best.onnx,
    and returns detected bounding boxes as JSON.
    """
    import cv2
    import onnxruntime as ort

    image_bytes = await request.body()

    # Decode JPEG bytes into OpenCV image array
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"error": "Invalid image payload"}, 400

    h, w = frame.shape[:2]

    # Preprocess image for YOLOv8 (640x640 RGB normalized)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (640, 640))
    img_input = img_resized.astype(np.float32) / 255.0
    img_input = np.transpose(img_input, (2, 0, 1)) # HWC -> CHW
    img_input = np.expand_dims(img_input, axis=0)  # Add batch dimension (1, 3, 640, 640)

    # Initialize ONNX Runtime session
    session = ort.InferenceSession("/root/best.onnx")
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Run inference
    outputs = session.run([output_name], {input_name: img_input})
    predictions = outputs[0][0] # Shape: (84, 8400) or similar

    # Post-process predictions
    # Class names for WeaponScan model
    class_names = ["weapon", "handgun", "rifle", "knife"]

    boxes = []
    scores = []
    class_ids = []

    # Format of YOLOv8 output: [cx, cy, w, h, score_class1, score_class2...]
    predictions = np.transpose(predictions) # Shape: (8400, 84)

    for pred in predictions:
        class_scores = pred[4:]
        class_id = np.argmax(class_scores)
        score = class_scores[class_id]

        if score >= 0.45: # Confidence threshold
            cx, cy, bw, bh = pred[0:4]
            # Scale coordinates back to original frame dimensions
            x1 = int((cx - bw / 2) * (w / 640.0))
            y1 = int((cy - bh / 2) * (h / 640.0))
            x2 = int((cx + bw / 2) * (w / 640.0))
            y2 = int((cy + bh / 2) * (h / 640.0))

            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(float(score))
            class_ids.append(int(class_id))

    # Apply Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.45, nms_threshold=0.45)

    detections = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, bw, bh = boxes[i]
            label = class_names[class_ids[i]] if class_ids[i] < len(class_names) else "weapon"
            detections.append({
                "label": label,
                "confidence": float(scores[i]),
                "bbox": [int(x), int(y), int(x + bw), int(y + bh)]
            })

    return {"detections": detections}
