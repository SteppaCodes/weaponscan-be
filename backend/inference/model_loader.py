import threading
from pathlib import Path
from ultralytics import YOLO

_model = None
_model_lock = threading.Lock()

def get_model(model_path: str = None) -> YOLO:
    """
    Load and cache the YOLO model as a singleton.
    
    Thread-safe: uses a lock to prevent two workers from
    loading the model simultaneously on the same process.
    
    The model stays in memory for the lifetime of the worker
    process. Celery workers are long-lived, so this is efficient.
    
    Args:
        model_path: Path to .pt weights file.
                    Defaults to settings.MODEL_PATH.
    
    Returns:
        Loaded YOLO model instance.
    """
    global _model

    if _model is not None:
        return _model
    
    with _model_lock:
        if _model is not None:
            return _model
        
        if model_path is None:
            from django.conf import settings
            model_path = getattr(settings, 'MODEL_PATH', 'models/best.onnx')

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model weights not found at {path.absolute()}. ")
        
        print(f"Loading YOLO model from {path}...")
        _model = YOLO(str(path))
        print(f"Model loaded successfully. Classes: {_model.names}")
        
        return _model
    
def unload_model():
    """Release the model from memory. Used in tests."""
    global _model
    _model = None