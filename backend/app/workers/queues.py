"""Celery queue definitions.

* ``pipeline``    - long-running catalogue processing (batch, GPU-heavy).
* ``interactive`` - short admin-triggered jobs (e.g. click-to-segment) that must not
  wait behind a 200-page catalogue.
* ``default``     - everything else.
"""

from __future__ import annotations

from kombu import Exchange, Queue

from app.core.constants import QUEUE_DEFAULT, QUEUE_INTERACTIVE, QUEUE_PIPELINE


def _queue(name: str) -> Queue:
    """A queue with its OWN exchange and routing key.

    Left unspecified, Celery binds every queue to the shared "default" exchange with routing
    key "default", so a message addressed to one queue is delivered to all of them.
    """
    return Queue(name, Exchange(name, type="direct"), routing_key=name)


QUEUES = (_queue(QUEUE_DEFAULT), _queue(QUEUE_PIPELINE), _queue(QUEUE_INTERACTIVE))
