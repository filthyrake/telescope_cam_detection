from __future__ import annotations

import logging
import os
import uuid
import contextvars
from typing import Optional

try:
    from pythonjsonlogger import jsonlogger
except Exception:  # pragma: no cover - defensive fallback
    jsonlogger = None

# Thread-local correlation id for request/frame level tracing
correlation_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("correlation_id", default=None)


class CorrelationIdFilter(logging.Filter):
    """Logging filter that injects correlation_id into every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        cid = correlation_id.get()
        # Make sure attribute exists for formatters
        record.correlation_id = cid
        return True


def set_correlation_id(cid: Optional[str] = None) -> str:
    """Set a correlation id for the current context and return it."""
    if cid is None:
        cid = str(uuid.uuid4())
    correlation_id.set(cid)
    return cid


def clear_correlation_id() -> None:
    correlation_id.set(None)


def setup_logging(level: int = logging.INFO, json_output: bool = False) -> None:
    """Configure root logging with optional JSON output.

    Args:
        level: logging level
        json_output: when True emit JSON logs using python-json-logger
    """
    root = logging.getLogger()
    # Remove existing handlers to avoid duplicates
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler()
    if json_output and jsonlogger is not None:
        fmt_fields = (
            '%(asctime)s %(name)s %(levelname)s %(message)s %(correlation_id)s'
        )
        formatter = jsonlogger.JsonFormatter(fmt_fields, rename_fields={
            'asctime': 'timestamp',
            'name': 'logger',
            'levelname': 'level'
        })
    else:
        # Human readable fallback
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    handler.setFormatter(formatter)
    root.addHandler(handler)

    # Add filter for correlation id on root logger so every record has the field
    root.addFilter(CorrelationIdFilter())
    root.setLevel(level)


__all__ = ["setup_logging", "set_correlation_id", "clear_correlation_id", "correlation_id"]
