import os
import cv2
import requests
from pathlib import Path

class HuggingFaceInferenceClient:
    """
    Remote Serverless API client for Hugging Face Inference API.
    
    Executes YOLO model predictions remotely on Hugging Face GPU hardware over HTTP.
    Requires ZERO local PyTorch/ONNX runtime memory and loads ZERO model weights into Python!
    """

    def __init__(self, model_id: str = "steppacodes/weaponscan", token: str = None):
        self.model_id = model_id
        self.api_url = f"https://router.huggingface.co/hf-inference/models/{model_id}"
        self.token = token or os.environ.get("HF_TOKEN")
        
        self.headers = {"Content-Type": "image/jpeg"}
        if self.token:
            self.headers["Authorization"] = f"Bearer {self.token}"

    def predict_frame(self, frame_np):
        """
        Send a single OpenCV frame (numpy array) to Hugging Face Inference API over HTTP.
        
        Returns:
            List of detected objects: [{'label': str, 'confidence': float, 'bbox': (x1, y1, x2, y2)}]
        """
        success, encoded_img = cv2.imencode('.jpg', frame_np, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            raise ValueError("Failed to encode frame to JPEG for Hugging Face API payload")

        response = requests.post(
            self.api_url,
            headers=self.headers,
            data=encoded_img.tobytes(),
            timeout=15,
        )

        if response.status_code != 200:
            raise RuntimeError(f"Hugging Face API error ({response.status_code}): {response.text}")

        results = response.json()
        h, w = frame_np.shape[:2]
        formatted_detections = []

        # Hugging Face object detection API returns list of dicts:
        # [{'score': 0.95, 'label': 'pistol', 'box': {'xmin': 10, 'ymin': 20, 'xmax': 100, 'ymax': 200}}]
        if isinstance(results, list):
            for item in results:
                score = item.get('score', 0.0)
                label = item.get('label', 'weapon')
                box = item.get('box', {})

                # Extract bounding box coordinates (normalize if fractional or keep absolute)
                xmin = box.get('xmin', 0)
                ymin = box.get('ymin', 0)
                xmax = box.get('xmax', 0)
                ymax = box.get('ymax', 0)

                # Convert to floats
                if isinstance(xmin, float) and xmin <= 1.0:
                    xmin, xmax = xmin * w, xmax * w
                    ymin, ymax = ymin * h, ymax * h

                formatted_detections.append({
                    'label': label,
                    'confidence': float(score),
                    'bbox': (float(xmin), float(ymin), float(xmax), float(ymax)),
                })

        return formatted_detections
