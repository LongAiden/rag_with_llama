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
        include=["worker.ingestion_tasks"],
    )

    ingestion_queue = os.getenv("CELERY_INGESTION_QUEUE", "ingestion")

    app.conf.update(
        # Stage tasks set their queue explicitly per signature (see
        # worker.ingestion_tasks.build_ingestion_chain); this is only the fallback
        # for anything dispatched without one, and must be a queue a worker consumes.
        task_default_queue=os.getenv("CELERY_DEFAULT_QUEUE", ingestion_queue),
        task_track_started=True,
        worker_prefetch_multiplier=int(os.getenv("CELERY_PREFETCH_MULTIPLIER", "1")),
        task_acks_late=True,
        result_expires=int(os.getenv("CELERY_RESULT_EXPIRES", "3600")),
        beat_schedule={
            # Full directory scan: picks up files dropped into INPUT_RAW_DIR
            # outside the upload API.
            "register_and_dispatch_weekly": {
                "task": "worker.ingestion_tasks.register_and_dispatch",
                "schedule": crontab(day_of_week="sunday", hour=0, minute=0),
                "options": {"queue": ingestion_queue},
            },
            # Recovery: release claims from workers that died mid-stage, retry
            # errored documents, and re-queue them. Resetting a stage without
            # re-dispatching would leave the document idle until the weekly scan.
            "recover_and_dispatch_6h": {
                "task": "worker.ingestion_tasks.recover_and_dispatch",
                "schedule": crontab(minute=0, hour="*/6"),
                "options": {"queue": ingestion_queue},
            },
        },
    )

    return app


celery_app = create_celery_app()
