import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'weaponscan.settings')

app = Celery('weaponscan')

# Load config from Django settings using the CELERY_ namespace
app.config_from_object('django.conf:settings', namespace='CELERY')

# Auto-discover tasks from all installed apps
app.autodiscover_tasks()

from celery.signals import worker_shutting_down

@worker_shutting_down.connect
def on_worker_shutdown(sig, how, exitcode, **kwargs):
    """
    Called when the worker receives SIGTERM.
    
    Celery will finish the current task before exiting
    because we use acks_late=True.
    """
    print("Worker shutting down gracefully. Finishing current task before exit.")


@app.task(bind=True, ignore_result=True)
def debug_task(self):
    print(f'Request: {self.request!r}')
