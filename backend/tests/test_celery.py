from __future__ import annotations

from app.core.constants import QUEUE_DEFAULT, QUEUE_INTERACTIVE, QUEUE_PIPELINE
from app.workers.celery_app import celery_app
from app.workers.tasks.system import ping


def test_ping_task_runs_locally() -> None:
    assert ping.apply().get() == "pong"


def test_task_is_registered() -> None:
    assert "system.ping" in celery_app.tasks


def test_queues_are_declared() -> None:
    names = {queue.name for queue in celery_app.conf.task_queues}
    assert names == {QUEUE_DEFAULT, QUEUE_PIPELINE, QUEUE_INTERACTIVE}
    assert celery_app.conf.task_default_queue == QUEUE_DEFAULT


def test_each_queue_has_its_own_exchange_and_routing_key() -> None:
    """Regression: without explicit values Celery binds every queue to exchange/key
    'default', so a task sent to 'pipeline' would also be delivered to the other queues."""
    for name in (QUEUE_DEFAULT, QUEUE_PIPELINE, QUEUE_INTERACTIVE):
        queue = celery_app.amqp.queues[name]
        assert queue.exchange.name == name
        assert queue.routing_key == name


def test_long_running_job_safety_settings() -> None:
    assert celery_app.conf.task_acks_late is True
    assert celery_app.conf.worker_prefetch_multiplier == 1
    assert celery_app.conf.task_serializer == "json"
