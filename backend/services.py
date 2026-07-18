import os, shutil
from uuid import UUID
from pathlib import Path
from datetime import datetime

from django.conf import settings
from django.utils import timezone

from .models import VideoJob, Detection, JobStatus
from .validators import validate_video_file, ValidationResult

class JobServiceError(Exception):
    """Base exception for job service errors."""
    pass


class InvalidVideoError(JobServiceError):
    """Raised when video validation fails."""
    pass


class JobNotFoundError(JobServiceError):
    """Raised when job ID doesn't exist."""
    pass


class JobService:
    """
    Central service for all VideoJob lifecycle operations.
    
    Every operation that creates, reads, or modifies a job
    MUST go through this service. Views, CLI commands, and
    admin actions all use this single entry point.
    """

    @classmethod
    def create_job(cls, uploaded_file) -> VideoJob:
        """
        Create a new video scan job.
        
        Steps:
        1. Save the uploaded file to disk
        2. Validate the video (format, size, codec)
        3. Create the database record
        4. Trigger the async Celery task
        
        Args:
            uploaded_file: Django UploadedFile object
            
        Returns:
            The created VideoJob instance
            
        Raises:
            InvalidVideoError: If the file fails validation
        """
        upload_dir = Path(settings.MEDIA_ROOT) / 'uploads'
        upload_dir.mkdir(parents=True, exist_ok=True)
        file_path = upload_dir / uploaded_file.name
        with open(file_path, 'wb') as dest:
            for chunk in uploaded_file.chunks():
                dest.write(chunk)

        result = validate_video_file(str(file_path))
        if not result.valid:
            if os.path.exists(file_path):
                os.remove(file_path)
            raise InvalidVideoError(result.error)
        
        job = VideoJob.objects.create(
            filename=uploaded_file.name,
            file_path=str(file_path),
            status=JobStatus.QUEUED,
            duration_seconds=result.duration,
            total_frames=result.total_frames,
            fps=result.fps,
        )

        from .tasks import process_video_task
        process_video_task.delay(str(job.id))
        return job
    

    @classmethod
    def get_job(cls, job_id: str) -> VideoJob:
        """
        Retrieve a VideoJob by its ID.
        """
        try:
            return VideoJob.objects.get(id=job_id)
        except VideoJob.DoesNotExist:
            raise JobNotFoundError(f"Job {job_id} not found")


    @classmethod
    def update_progress(cls, job_id: str, progress: int, status: str = None):
        """
        Update job progress atomically.
        
        Uses update() instead of save() to avoid race conditions
        when the worker and the API read the same row simultaneously.
        """
        updates = {'progress': min(progress, 100)}
        
        if status:
            updates['status'] = status
        
            if status == JobStatus.PROCESSING:
                updates['started_at'] = timezone.now()
            
            if status in (JobStatus.COMPLETE, JobStatus.ERROR):
                updates['completed_at'] = timezone.now()
        
        VideoJob.objects.filter(id=job_id).update(**updates)

    @classmethod
    def mark_error(cls, job_id: str, error_message: str):
        """Mark a job as failed with an error message."""
        VideoJob.objects.filter(id=job_id).update(
            status=JobStatus.ERROR,
            error_message=error_message,
            completed_at=timezone.now(),
        )

    @classmethod
    def save_detection(cls, job_id: str, timestamp: float,
                       frame_index: int, label: str,
                       confidence: float, bbox: tuple,
                       image_path: str) -> Detection:
        """Save a single detection result."""
        x1, y1, x2, y2 = bbox
        return Detection.objects.create(
            job_id=job_id,
            timestamp=timestamp,
            frame_index=frame_index,
            label=label,
            confidence=confidence,
            bbox_x1=x1,
            bbox_y1=y1,
            bbox_x2=x2,
            bbox_y2=y2,
            image_path=image_path,
        )
    
    @classmethod
    def save_detections_idempotent(cls, job_id: str, detections: list):
        """
        Save detections idempotently.
        
        If called twice with the same data (e.g., after a retry),
        existing detections for the same frame_index are skipped.
        
        Uses get_or_create to prevent duplicates.
        """
        from .models import Detection
        
        created_count = 0
        skipped_count = 0
        
        for d in detections:
            _, created = Detection.objects.get_or_create(
                job_id=job_id,
                frame_index=d['frame_index'],
                label=d['label'],
                defaults={
                    'timestamp': d['timestamp'],
                    'confidence': d['confidence'],
                    'bbox_x1': d['bbox'][0],
                    'bbox_y1': d['bbox'][1],
                    'bbox_x2': d['bbox'][2],
                    'bbox_y2': d['bbox'][3],
                    'image_path': d['image_path'],
                }
            )
            
            if created:
                created_count += 1
            else:
                skipped_count += 1
        
        return created_count, skipped_count
        
    @classmethod
    def get_job_detections(cls, job_id: UUID):
        """Get all detections for a job, ordered by timestamp."""
        return Detection.objects.filter(job_id=job_id).order_by('timestamp')
    
    @classmethod
    def get_results_zip(cls, job_id: str) -> str:
        """
        Generate a ZIP of all annotated frames for a job.
        
        IDEMPOTENT: If the ZIP already exists on disk,
        return it immediately without regenerating.
        """
        import zipfile
        from pathlib import Path
        
        media_root = getattr(settings, 'MEDIA_ROOT', 'media')
        zip_path = Path(media_root) / 'exports' / f'{job_id}.zip'
        
        # Idempotency check — don't regenerate
        if zip_path.exists():
            return str(zip_path)
        
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        
        detections = Detection.objects.filter(job_id=job_id)
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            seen_paths = set()
            for det in detections:
                if det.image_path in seen_paths:
                    continue
                if os.path.exists(det.image_path):
                    zipf.write(det.image_path, os.path.basename(det.image_path))
                    seen_paths.add(det.image_path)
        
        return str(zip_path)
