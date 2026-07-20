import os
import cv2
import requests
from django.conf import settings

class ModalInferenceClient:
    """
    Client for Modal Serverless Web Endpoint.
    Sends image frame bytes over HTTP to Modal and returns detected bounding boxes.
    """

    def __init__(self, endpoint_url: str = None):
        self.endpoint_url = endpoint_url or getattr(
            settings,
            'MODAL_ENDPOINT_URL',
            os.environ.get('MODAL_ENDPOINT_URL', '')
        )
        if not self.endpoint_url:
            raise ValueError("MODAL_ENDPOINT_URL environment variable or setting is missing!")

    def predict_frame(self, frame_np):
        """
        Send OpenCV frame (numpy array) to Modal web endpoint over HTTP.
        
        Returns:
            List of detected objects: [{'label': str, 'confidence': float, 'bbox': (x1, y1, x2, y2)}]
        """
        success, encoded_img = cv2.imencode('.jpg', frame_np, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            raise ValueError("Failed to encode frame to JPEG for Modal payload")

        response = requests.post(
            self.endpoint_url,
            headers={"Content-Type": "image/jpeg"},
            data=encoded_img.tobytes(),
            timeout=15,
        )

        if response.status_code != 200:
            raise RuntimeError(f"Modal Inference API error ({response.status_code}): {response.text}")

        res_data = response.json()
        raw_detections = res_data.get("detections", [])

        formatted_detections = []
        for d in raw_detections:
            x1, y1, x2, y2 = d["bbox"]
            formatted_detections.append({
                "label": d.get("label", "weapon"),
                "confidence": float(d.get("confidence", 0.0)),
                "bbox": (float(x1), float(y1), float(x2), float(y2)),
            })

        return formatted_detections
