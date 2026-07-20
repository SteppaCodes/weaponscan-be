import threading
from pathlib import Path
_model = None
_model_lock = threading.Lock()

def get_model(model_path: str = None) -> "YOLO":
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
            model_path = getattr(settings, 'MODEL_PATH', 'models/best.pt')

        path = Path(model_path)
        if not path.exists():
            print(f"Local model file '{path}' not found on disk. Downloading 'best.onnx' from Hugging Face Hub (steppacodes/weaponscan)...", flush=True)
            try:
                from huggingface_hub import hf_hub_download
                downloaded_path = hf_hub_download(
                    repo_id="steppacodes/weaponscan",
                    filename="best.onnx",
                )
                path = Path(downloaded_path)
                print(f"Hugging Face model downloaded successfully to: {path}", flush=True)
            except Exception as e:
                raise FileNotFoundError(
                    f"Model weights not found locally at {Path(model_path).absolute()} and failed to download from Hugging Face: {str(e)}"
                )
        
        from ultralytics import YOLO
        print(f"Loading YOLO model from {path}...", flush=True)
        _model = YOLO(str(path))
        print(f"Model loaded successfully! Classes: {_model.names}", flush=True)
        
        return _model
    
def unload_model():
    """Release the model from memory. Used in tests."""
    global _model
    _model = None