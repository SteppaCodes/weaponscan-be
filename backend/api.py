from uuid import UUID

from ninja import NinjaAPI, File
from ninja.files import UploadedFile
from django.http import FileResponse

from .schemas import (
    JobCreateResponse,
    JobStatusResponse,
    JobResultsResponse,
    DetectionResponse,
    ErrorResponse,
)
from .services import (
    JobService,
    InvalidVideoError,
    JobNotFoundError,
)


api = NinjaAPI(
    title="WeaponScan API",
    version="1.0.0",
    description="Video weapon detection API powered by YOLOv8m",
)


# ── Upload ──────────────────────────────────────────────

@api.post(
    "/jobs/upload",
    response={201: JobCreateResponse},
    summary="Upload a video for weapon scanning",
)
def upload_video(request, video: UploadedFile = File(...)):
    """
    Upload a video file to start a weapon detection scan.
    
    The video is validated, saved, and queued for async processing.
    Returns immediately with a job ID for polling.
    
    Supported formats: .mp4, .avi, .mov, .mkv
    Maximum file size: 500MB
    """
    try:
        job = JobService.create_job(video)
        return 201, job
    except InvalidVideoError as e:
        return api.create_response(
            request,
            {"error": "Invalid video", "detail": str(e)},
            status=422,
        )
    except Exception as e:
        import traceback
        err_detail = f"{str(e)}\n{traceback.format_exc()}"
        print(f"[API UPLOAD ERROR] {err_detail}", flush=True)
        return api.create_response(
            request,
            {"error": "Upload failed", "detail": str(e)},
            status=500,
        )


# ── Status Polling ──────────────────────────────────────

@api.get(
    "/jobs/{job_id}/status",
    response={200: JobStatusResponse, 404: ErrorResponse},
    summary="Get the current status of a scan job",
)
def get_job_status(request, job_id: UUID):
    """
    Poll this endpoint to track scan progress.
    
    Recommended polling interval: 2 seconds.
    Stop polling when status is COMPLETE or ERROR.
    """
    try:
        job = JobService.get_job(job_id)
        return 200, job
    except JobNotFoundError:
        return 404, {"error": f"Job {job_id} not found"}


# ── Results ─────────────────────────────────────────────

@api.get(
    "/jobs/{job_id}/results",
    response={200: JobResultsResponse, 404: ErrorResponse},
    summary="Get detection results for a completed job",
)
def get_job_results(request, job_id: UUID):
    """
    Returns all detections found during the scan.
    
    Only available when job status is COMPLETE.
    """
    try:
        job = JobService.get_job(job_id)
    except JobNotFoundError:
        return 404, {"error": f"Job {job_id} not found"}
    
    detections = JobService.get_job_detections(job_id)
    
    return 200, {
        "job": job,
        "detections": list(detections),
        "total_detections": detections.count(),
    }


# ── Job List ────────────────────────────────────────────

@api.get(
    "/jobs",
    response=list[JobStatusResponse],
    summary="List all scan jobs",
)
def list_jobs(request, limit: int = 20, offset: int = 0):
    """List recent jobs, newest first."""
    from .models import VideoJob
    return VideoJob.objects.all()[offset:offset + limit]


# ── Health Check ────────────────────────────────────────

@api.get("/health", summary="System health check")
def health_check(request):
    """
    Check that all dependencies are reachable.
    
    Used by Docker HEALTHCHECK and load balancers.
    """
    from django.db import connection
    from django.core.cache import cache
    
    db_ok = False
    redis_ok = False
    
    try:
        connection.ensure_connection()
        db_ok = True
    except Exception:
        pass
    
    try:
        cache.set('health_check', '1', timeout=5)
        redis_ok = cache.get('health_check') == '1'
    except Exception:
        pass
    
    status = "healthy" if (db_ok and redis_ok) else "unhealthy"
    code = 200 if status == "healthy" else 503
    
    return api.create_response(
        request,
        {
            "status": status,
            "services": {"database": db_ok, "redis": redis_ok},
        },
        status=code,
    )


# ── Metrics ─────────────────────────────────────────────

@api.get("/metrics", summary="Application metrics")
def metrics_endpoint(request):
    """
    Expose application metrics for monitoring.
    """
    from .metrics import get_metrics
    return get_metrics()