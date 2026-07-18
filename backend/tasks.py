from celery import shared_task
import os, cv2, traceback
from pathlib import Path
from django.conf import settings
from django.utils import timezone
from datetime import timedelta

from .services import JobService
from .models import VideoJob, JobStatus
from .inference.model_loader import get_model
from .inference.frame_extractor import FrameExtractor



@shared_task(
    bind=True,
    queue='inference',
    acks_late=True,
    reject_on_worker_lost=True,
    max_retries=3,
    autoretry_for=(ConnectionError, TimeoutError),
    retry_backoff=True,        # Exponential backoff
    retry_backoff_max=600,     # Max 10 minutes between retries
    retry_jitter=True,         # Add randomness to prevent thundering herd
)
def process_video_task(self, job_id: str):
    """
    Main inference task. Runs on the 'inference' queue.
    
    Lifecycle:
    1. Load model (singleton — instant after first load)
    2. Extract frames from the video
    3. Run YOLOv8m on each frame
    4. Save detections to database
    5. Save annotated frames to disk
    6. Update progress periodically
    
    Error handling:
    - Corrupt video → mark ERROR, no retry
    - OOM → retry with exponential backoff
    - Unknown error → retry up to max_retries
    """

    try:
        job = JobService.get_job(job_id)
    except Exception:
        return
    
    JobService.update_progress(job_id, 0, status=JobStatus.PROCESSING)
    job_results_dir = Path(settings.MEDIA_ROOT) / 'results' / str(job_id)
    job_results_dir.mkdir(parents=True, exist_ok=True)

    try:
        model = get_model()
        extractor = FrameExtractor(job.file_path, skip_frames=3)

        detection_count = 0
        last_progress = 0
        detection_buffer = []

        for frame_info in extractor.extract():
            # Run inference
            results = model.predict(
                frame_info.frame,
                conf=0.5,          # Confidence threshold
                iou=0.45,          # NMS IoU threshold
                imgsz=640,
                verbose=False,
            )

            boxes = results[0].boxes

            # Process detection
            if len(boxes) > 0:
                annotated = results[0].plot()
                frame_filename = f"frame_{frame_info.index:06d}.jpg"
                frame_path = job_results_dir / frame_filename
                cv2.imwrite(str(frame_path), annotated)

                for box in boxes:
                    detection_buffer.append({
                        'job_id': job_id,
                        'timestamp': frame_info.timestamp,
                        'frame_index': frame_info.index,
                        'label': model.names[int(box.cls[0])],
                        'confidence': float(box.conf[0]),
                        'bbox': tuple(box.xyxy[0].cpu().numpy()),
                        'image_path': str(frame_path),
                    })
                    detection_count += 1

            if len(detection_buffer) >= 50:
                _flush_detections(detection_buffer)
                detection_buffer = []

            # Prevent division by zero if total_frames is 0 or None
            total_frames = extractor.total_frames if extractor.total_frames > 0 else 1
            progress = int((frame_info.index / total_frames) * 100)

            if progress > last_progress:
                JobService.update_progress(job_id, progress)
                last_progress = progress

        if detection_buffer:
            _flush_detections(detection_buffer)

        JobService.update_progress(job_id, 100, status=JobStatus.COMPLETE)
        print(
            f"Job {job_id} complete: "
            f"{detection_count} detections found"
        )
    
    except cv2.error as e:
        # Corrupt video — don't retry, it will always fail
        JobService.mark_error(job_id, f"Video processing error: {str(e)}")
        return
    
    except MemoryError:
        # OOM — retry with backoff, might work if other tasks finish
        JobService.update_progress(
            job_id, last_progress, status=JobStatus.QUEUED
        )
        raise self.retry(exc=MemoryError("OOM during inference"))
    
    except Exception as exc:
        # Unknown error — log full traceback, then retry
        error_msg = traceback.format_exc()
        
        if self.request.retries >= self.max_retries:
            JobService.mark_error(job_id, error_msg[-500:])
            return
        
        raise self.retry(exc=exc)
    
def _flush_detections(buffer):
    """Bulk-insert detections for performance."""
    from .models import Detection

    Detection.objects.bulk_create([
        Detection(
            job_id=d['job_id'],
            timestamp=d['timestamp'],
            frame_index=d['frame_index'],
            label=d['label'],
            confidence=d['confidence'],
            bbox_x1=d['bbox'][0],
            bbox_y1=d['bbox'][1],
            bbox_x2=d['bbox'][2],
            bbox_y2=d['bbox'][3],
            image_path=d['image_path'],
        )
        for d in buffer
    ])


@shared_task(queue='default')
def reap_stuck_jobs():
    """
    Periodic task that finds and cleans up stuck jobs.
    
    A job is "stuck" if it has been in PROCESSING for longer
    than MAX_PROCESSING_TIME without any progress update.
    """
    MAX_PROCESSING_TIME = timedelta(minutes=15)
    MAX_QUEUED_TIME = timedelta(minutes=30)
    
    now = timezone.now()
    
    # Find stuck PROCESSING jobs
    stuck_processing = VideoJob.objects.filter(
        status=JobStatus.PROCESSING,
        started_at__lt=now - MAX_PROCESSING_TIME,
    )
    
    for job in stuck_processing:
        elapsed = now - job.started_at
        print(f"Reaping stuck job {job.id} (stuck for {elapsed})")
        
        JobService.mark_error(
            str(job.id),
            f"Job timed out after {elapsed.total_seconds():.0f}s. "
            f"Worker may have crashed."
        )
    
    # Find stuck QUEUED jobs (Redis may have lost the task)
    stuck_queued = VideoJob.objects.filter(
        status=JobStatus.QUEUED,
        created_at__lt=now - MAX_QUEUED_TIME,
    )
    
    for job in stuck_queued:
        elapsed = now - job.created_at
        print(f"Re-queuing stuck job {job.id} (queued for {elapsed})")
        
        # Re-dispatch to Celery
        process_video_task.delay(str(job.id))
    
    reaped = stuck_processing.count()
    requeued = stuck_queued.count()
    
    if reaped or requeued:
        print(f"Reaper: {reaped} jobs marked ERROR, {requeued} jobs re-queued")
