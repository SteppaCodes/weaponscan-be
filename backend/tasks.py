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
@shared_task(
    bind=True,
    queue='inference',
    acks_late=True,
    reject_on_worker_lost=True,
    max_retries=1,
)
def process_video_task(self, job_id: str):
    """
    Main inference task with comprehensive debug logging and RSS memory tracking.
    """
    import sys, resource, time
    def get_mem():
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    start_time = time.time()
    print(f"\n[JOB START {job_id[:8]}] Starting video processing task. Initial RSS: {get_mem():.1f} MB", flush=True)

    try:
        job = JobService.get_job(job_id)
        print(f"[JOB INFO {job_id[:8]}] Found job record: filename='{job.filename}', path='{job.file_path}'", flush=True)
    except Exception as e:
        err_msg = f"Failed to fetch job record from database: {str(e)}\n{traceback.format_exc()}"
        print(f"[JOB ERROR {job_id[:8]}] {err_msg}", flush=True)
        return
    
    if not os.path.exists(job.file_path):
        err_msg = f"Video file missing on disk at path: {job.file_path}"
        print(f"[JOB ERROR {job_id[:8]}] {err_msg}", flush=True)
        JobService.mark_error(job_id, err_msg)
        return

    file_size_mb = os.path.getsize(job.file_path) / (1024 * 1024)
    print(f"[JOB FILE {job_id[:8]}] Video file verified on disk ({file_size_mb:.2f} MB)", flush=True)

    JobService.update_progress(job_id, 0, status=JobStatus.PROCESSING)
    job_results_dir = Path(settings.MEDIA_ROOT) / 'results' / str(job_id)
    job_results_dir.mkdir(parents=True, exist_ok=True)

    use_hf_api = getattr(settings, 'USE_HF_INFERENCE', False) or os.environ.get('USE_HF_INFERENCE', '').lower() == 'true' or bool(os.environ.get('HF_TOKEN'))

    try:
        if use_hf_api:
            print(f"[JOB INFERENCE {job_id[:8]}] Mode: REMOTE HUGGING FACE INFERENCE API (Zero local model weights loaded)", flush=True)
            from .inference.hf_client import HuggingFaceInferenceClient
            hf_client = HuggingFaceInferenceClient(model_id="steppacodes/weaponscan")
            model = None
        else:
            print(f"[JOB INFERENCE {job_id[:8]}] Mode: LOCAL ONNX MODEL INFERENCE", flush=True)
            import torch
            import gc
            torch.set_num_threads(1)
            model = get_model()
            hf_client = None

        print(f"[JOB EXTRACT {job_id[:8]}] Initializing FrameExtractor...", flush=True)
        temp_extractor = FrameExtractor(job.file_path, skip_frames=1)
        fps = temp_extractor.fps if temp_extractor.fps > 0 else 30
        dynamic_skip = max(1, int(fps // 2))
        extractor = FrameExtractor(job.file_path, skip_frames=dynamic_skip)
        
        print(
            f"[JOB EXTRACT {job_id[:8]}] Extractor ready: "
            f"Resolution={extractor.width}x{extractor.height}, FPS={extractor.fps:.1f}, "
            f"TotalFrames={extractor.total_frames}, DynamicSkip={dynamic_skip}",
            flush=True
        )

        detection_count = 0
        last_progress = 0
        detection_buffer = []
        processed_frames_count = 0

        for frame_info in extractor.extract():
            processed_frames_count += 1
            
            if processed_frames_count == 1 or processed_frames_count % 5 == 0:
                print(
                    f"[JOB INFERENCE {job_id[:8]}] Frame {frame_info.index}/{extractor.total_frames} "
                    f"({processed_frames_count} processed). RSS: {get_mem():.1f} MB",
                    flush=True
                )

            # Perform inference either remotely via HF API or locally via ONNX
            if hf_client:
                frame_detections = hf_client.predict_frame(frame_info.frame)
                annotated = frame_info.frame.copy()
                
                if len(frame_detections) > 0:
                    for d in frame_detections:
                        x1, y1, x2, y2 = map(int, d['bbox'])
                        label_text = f"{d['label']} {d['confidence']:.2f}"
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(annotated, label_text, (x1, max(15, y1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                        detection_buffer.append({
                            'job_id': job_id,
                            'timestamp': frame_info.timestamp,
                            'frame_index': frame_info.index,
                            'label': d['label'],
                            'confidence': d['confidence'],
                            'bbox': (x1, y1, x2, y2),
                            'image_path': '', # set below
                        })
                        detection_count += 1

                    frame_filename = f"frame_{frame_info.index:06d}.jpg"
                    frame_path = job_results_dir / frame_filename
                    cv2.imwrite(str(frame_path), annotated)
                    for d in detection_buffer[-len(frame_detections):]:
                        d['image_path'] = str(frame_path)

            else:
                # Local ONNX inference
                results = model.predict(
                    frame_info.frame,
                    conf=0.5,
                    iou=0.45,
                    imgsz=640,
                    verbose=False,
                )
                boxes = results[0].boxes

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

                del results, boxes

            if len(detection_buffer) >= 50:
                _flush_detections(detection_buffer)
                detection_buffer = []

            total_frames = extractor.total_frames if extractor.total_frames > 0 else 1
            progress = int((frame_info.index / total_frames) * 100)

            if progress > last_progress:
                JobService.update_progress(job_id, progress)
                last_progress = progress

            if processed_frames_count % 5 == 0:
                gc.collect()

        if detection_buffer:
            _flush_detections(detection_buffer)

        elapsed = time.time() - start_time
        JobService.update_progress(job_id, 100, status=JobStatus.COMPLETE)
        print(
            f"[JOB SUCCESS {job_id[:8]}] Task completed in {elapsed:.2f}s! "
            f"Found {detection_count} detections across {processed_frames_count} frames. Final RSS: {get_mem():.1f} MB",
            flush=True
        )
    
    except cv2.error as e:
        err_msg = f"OpenCV Video Processing Error: {str(e)}\n{traceback.format_exc()}"
        print(f"[JOB CRASH {job_id[:8]}] {err_msg}", flush=True)
        JobService.mark_error(job_id, err_msg)
    
    except Exception as exc:
        err_msg = f"Unhandled Task Error ({type(exc).__name__}): {str(exc)}\n{traceback.format_exc()}"
        print(f"[JOB CRASH {job_id[:8]}] {err_msg}", flush=True)
        JobService.mark_error(job_id, err_msg)
    
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
