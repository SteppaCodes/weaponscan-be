import os, cv2
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Maximum file size: 500MB (REQ-F01)
MAX_FILE_SIZE = 500 * 1024 * 1024

# Allowed extensions
ALLOWED_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv'}

# Minimum video duration (seconds)
MIN_DURATION = 1.0

# Maximum video duration (seconds) — 30 minutes for MVP
MAX_DURATION = 30 * 60


@dataclass
class ValidationResult:
    """Result of video file validation."""
    valid: bool
    error: Optional[str] = None
    duration: Optional[float] = None
    fps: Optional[float] = None
    total_frames: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None


def validate_video_file(file_path: str) -> ValidationResult:
    """
    Validate an uploaded video file before it enters the pipeline.
    
    Checks:
    1. File exists and is readable
    2. Extension is allowed
    3. File size is within limits (REQ-F01: <= 500MB)
    4. OpenCV can open the file (not corrupt)
    5. Video has valid FPS and frame count
    6. Duration is within acceptable range
    
    Returns:
        ValidationResult with metadata if valid, error message if not.
    """

    path = Path(file_path)

    if not path.exists():
        return ValidationResult(valid=False, error="File not found")

    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        return ValidationResult(valid=False, error=f"Invalid format: {path.suffix}. Allowed: {','.join(ALLOWED_EXTENSIONS)}")
    
    file_size = os.path.getsize(file_path)
    if file_size > MAX_FILE_SIZE:
        size_mb = file_size /(1024 * 1024)
        return ValidationResult(valid=False, error=f"File too large: size {size_mb:0f}MB (max: 500MB)")
    
    if file_size == 0:
        return ValidationResult(valid=False, error="File is empty (0 bytes)")
    
    cap = cv2.VideoCapture(str(file_path))
    if not cap.isOpened():
        return ValidationResult(
            valid=False,
            error="Cannot open video file. File may be corrupt "
                  "or use an unsupported codec."
        )
    
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps <= 0 or fps >= 120:
            return ValidationResult(valid=False, error=f"Invalid FPS: {fps}. Expected 1-120")

        if total_frames <= 0:
            return ValidationResult(valid=False, error="Video has no readable frames")
        
        duration = total_frames / fps

        if duration < MIN_DURATION:
            return ValidationResult(valid=False, error=f"Video too long: {duration/60:.0f}min (max: 30min)")
        
        ret, frame = cap.read()
        if not ret or frame is None:
            return ValidationResult(valid=False, error="Cannot read first frame. File may be corrupt")
        
        return ValidationResult(
            valid=True,
            duration=duration,
            fps=fps,
            total_frames=total_frames,
            width=width,
            height=height,
        )
    
    finally:
        cap.release()
