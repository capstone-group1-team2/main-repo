"""Structured (JSON) logging configuration — the single place logging is
set up for both the FastAPI backend (app/main.py) and the standalone
ingestion script (ingestion/ingest.py). Every other module in this codebase
just calls logging.getLogger(__name__) and writes through whatever handler
this attaches to the root logger, so nothing else needs to change.

Deliberately takes `level` as a parameter rather than reading os.environ
itself — app/config.py and ingestion/config.py remain the only two places
in this codebase allowed to read environment variables directly; each
exposes its own LOG_LEVEL and passes it in here.
"""

from __future__ import annotations

import logging
import sys

from pythonjsonlogger import jsonlogger

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def configure_logging(level: str = "INFO") -> None:
    """Attaches a single JSON-formatted stream handler to the root logger.
    Idempotent — safe to call more than once (e.g. once from app/main.py's
    module scope and once from a test) without duplicating log lines."""
    root = logging.getLogger()
    root.setLevel(level.upper())

    # Drop any handler this function previously added instead of stacking
    # a second one on repeated calls.
    for handler in list(root.handlers):
        if getattr(handler, "_is_app_json_handler", False):
            root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        jsonlogger.JsonFormatter(
            _LOG_FORMAT,
            rename_fields={"asctime": "timestamp", "levelname": "level", "name": "logger"},
        )
    )
    handler._is_app_json_handler = True  
    root.addHandler(handler)