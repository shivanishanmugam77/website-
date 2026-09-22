"""Errors shared by the machine-learning providers."""

from __future__ import annotations


class ModelUnavailableError(RuntimeError):
    """A model or the software it needs is missing or failed to load (an operator problem, not
    a problem with the picture being examined). The message says what to do about it."""
