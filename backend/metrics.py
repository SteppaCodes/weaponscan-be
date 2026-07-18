import time
from functools import wraps
from django.conf import settings

# Simple metrics collection (replace with Prometheus in Stage 2)
_metrics = {
    'jobs_created': 0,
    'jobs_completed': 0,
    'jobs_failed': 0,
    'detections_found': 0,
    'inference_seconds_total': 0,
    'inference_count': 0,
}


def increment(metric, value=1):
    """Thread-safe metric increment."""
    _metrics[metric] = _metrics.get(metric, 0) + value


def get_metrics():
    """Return all current metrics."""
    result = dict(_metrics)
    if result['inference_count'] > 0:
        result['inference_avg_seconds'] = (
            result['inference_seconds_total'] / result['inference_count']
        )
    return result


def track_inference_time(func):
    """Decorator to track inference duration."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        increment('inference_seconds_total', duration)
        increment('inference_count')
        return result
    return wrapper
