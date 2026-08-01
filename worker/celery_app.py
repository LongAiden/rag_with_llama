import os
from celery import Celery
from celery.schedules import crontab


def create_celery_app() -> Celery:
    """Create a Celery application configured from environment variables."""
    broker_url = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
    result_backend = os.getenv("CELERY_RESULT_BACKEND", broker_url)

    app = Celery(
        "rag_tasks",
        broker=broker_url,
        backend=result_backend,
        include=["worker.tasks", "worker.ingestion_tasks"],
    )

    app.conf.update(
        task_default_queue=os.getenv("CELERY_DEFAULT_QUEUE", "rag"),
        task_track_started=True,
        worker_prefetch_multiplier=int(os.getenv("CELERY_PREFETCH_MULTIPLIER", "1")),
        task_acks_late=True,
        result_expires=int(os.getenv("CELERY_RESULT_EXPIRES", "3600")),
        task_routes={
            "worker.ingestion_tasks.*": {"queue": "ingestion"},
            "worker.tasks.*": {"queue": "rag"},
        },
        beat_schedule={
            "register_and_dispatch_weekly": {
                "task": "worker.ingestion_tasks.register_and_dispatch",
                "schedule": crontab(day_of_week="sunday", hour=0, minute=0),
            },
            "sweep_stale_documents_6h": {
                "task": "worker.ingestion_tasks.sweep_stale_documents",
                "schedule": crontab(minute=0, hour="*/6"),
            },
        },
    )

    return app


celery_app = create_celery_app()
