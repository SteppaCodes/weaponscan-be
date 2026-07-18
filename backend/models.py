import uuid
from django.db import models

class JobStatus(models.TextChoices):
    QUEUED = 'QUEUED', 'Queued'
    PROCESSING = 'PROCESSING', 'Processing'
    COMPLETE = 'COMPLETE', 'Complete'
    ERROR = 'ERROR', 'Error'


class VideoJob(models.Model):
    """
    Aggregate Root: tracks the lifecycle of a single video scan.
    
    State machine: QUEUED → PROCESSING → COMPLETE | ERROR
    
    Invariants:
    - progress must be 0-100
    - completed_at must be set when status is COMPLETE or ERROR
    - file_path must exist on disk while status is PROCESSING
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    filename = models.CharField(max_length=255)
    file_path = models.CharField(max_length=512)
    status = models.CharField(max_length=20, choices=JobStatus, default=JobStatus.QUEUED, db_index=True)
    progress = models.IntegerField(default=0)

    duration_seconds = models.FloatField(null=True, blank=True)
    total_frames = models.IntegerField(null=True)
    fps = models.FloatField(null=True, blank=True)

    error_message = models.TextField(blank=True, default="")

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=['status'], name='idx_job_status'),
            models.Index(fields=['-created_at'], name='idx_job_created'),
        ]

    def __str__(self):
        return f"Job {str(self.id)[:8]}... [{self.status}] {self.filename}"
    

class Detection(models.Model):
    """
    A single weapon detection within a video frame.
    
    Belongs to a VideoJob via CASCADE — deleting the job
    automatically removes all its detections.
    
    Uses BigAutoField (not UUID) because this ID is never
    exposed publicly. Internal-only references use integers
    for smaller index size.
    """

    job = models.ForeignKey(VideoJob, on_delete=models.CASCADE, related_name="detections")
    timestamp = models.FloatField()
    frame_index = models.IntegerField()
    label = models.CharField(max_length=50)
    confidence = models.FloatField()

    bbox_x1 = models.FloatField()
    bbox_y1 = models.FloatField()
    bbox_x2 = models.FloatField()
    bbox_y2 = models.FloatField()

    image_path = models.CharField(max_length=512)
    created_at = models.DateTimeField(auto_now_add=True)


    class Meta:
        ordering = ['timestamp']
        indexes = [
            models.Index(
                fields=['job', 'timestamp'],
                name='idx_det_job_ts'
            ),
        ]
    
    def __str__(self):
        return f"{self.label} ({self.confidence:.0%}) at {self.timestamp:.1f}s"