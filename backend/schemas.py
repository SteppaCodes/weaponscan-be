from ninja import Schema
from uuid import UUID
from datetime import datetime
from typing import List, Optional


class JobCreateResponse(Schema):
    """Returned after a successful video upload."""
    id: UUID
    filename: str
    status: str
    created_at: datetime


class JobStatusResponse(Schema):
    """Returned when polling for job status."""
    id: UUID
    filename: str
    status: str
    progress: int
    duration_seconds: Optional[float] = None
    total_frames: Optional[int] = None
    error_message: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class DetectionResponse(Schema):
    """A single weapon detection."""
    id: int
    timestamp: float
    frame_index: int
    label: str
    confidence: float
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    image_path: str


class JobResultsResponse(Schema):
    """Complete results for a finished job."""
    job: JobStatusResponse
    detections: List[DetectionResponse]
    total_detections: int


class ErrorResponse(Schema):
    """Standard error response."""
    error: str
    detail: Optional[str] = None


    