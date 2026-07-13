"""Pytest tests replaying the exact log calls ingestion/ingest.py makes,
without needing a live Weaviate/Neo4j/Redis stack (R6).
 
Run from the repo root:
 
    pytest test/test_logging_ingest_style.py -v
 
Confirms ingestion/config.py's LOG_LEVEL is wired correctly and that
ingest.py's real log call shapes (including %-style formatting args)
come out as valid JSON with the right logger name and level.
 
IMPORTANT: configure_logging() is called from inside each test's own body
(the "call" phase), never from a fixture's setup code (the "setup" phase).
logging.StreamHandler binds whatever sys.stdout object exists at the
moment it's constructed — if that happens during fixture setup, it can
end up pointing at a different stdout capture buffer than the one
capsys.readouterr() drains during the test body, causing readouterr() to
come back empty even though the JSON line was genuinely printed. Calling
configure_logging() in the same phase as the log call and the capture
sidesteps this entirely.
"""
 
from __future__ import annotations
 
import json
import logging
 
import pytest
 
from ingestion.config import LOG_LEVEL
from logging_config import configure_logging
 
 
@pytest.fixture(autouse=True)
def _reset_root_handlers():
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    yield
    root.handlers = original_handlers
    root.setLevel(original_level)
 
 
def _ingest_logger_and_capture(capsys):
    """Configures logging (inside the test's own call phase) and returns
    (logger, capture_fn), mirroring test_logging_smoke.py's proven pattern."""
    configure_logging(LOG_LEVEL)
    logger = logging.getLogger("ingestion.ingest")
    capsys.readouterr()  # clear setup noise
    return logger, lambda: capsys.readouterr().out
 
 
def test_ingestion_config_defines_log_level():
    """ingest.py imports LOG_LEVEL directly — if ingestion/config.py doesn't
    define it, every import of this module (and ingest.py itself) breaks."""
    assert LOG_LEVEL, "ingestion.config.LOG_LEVEL is missing or empty"
    assert LOG_LEVEL.upper() in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
 
 
def test_skip_unchanged_file_message(capsys):
    logger, capture = _ingest_logger_and_capture(capsys)
    logger.info(
        "%s: file content unchanged and already present in Weaviate+Neo4j — "
        "skipped chunking and embedding entirely.",
        "cancel.md",
    )
    parsed = json.loads(capture().strip())
    assert parsed["logger"] == "ingestion.ingest"
    assert parsed["level"] == "INFO"
    assert "cancel.md" in parsed["message"]
    assert "skipped chunking and embedding entirely" in parsed["message"]
 
 
def test_self_heal_warning_message(capsys):
    logger, capture = _ingest_logger_and_capture(capsys)
    logger.warning(
        "%s: file content unchanged but missing downstream (weaviate: %d, neo4j: %d) — "
        "reprocessing to self-heal (Weaviate/Neo4j reset independently of hash_store.json).",
        "refund.md", 3, 0,
    )
    parsed = json.loads(capture().strip())
    assert parsed["level"] == "WARNING"
    assert "refund.md" in parsed["message"]
    assert "weaviate: 3, neo4j: 0" in parsed["message"]
 
 
def test_chunk_diff_summary_message(capsys):
    logger, capture = _ingest_logger_and_capture(capsys)
    logger.info(
        "%s: +%d added, -%d removed, %d unchanged (version %d).",
        "subscription.md", 2, 0, 1, 7,
    )
    parsed = json.loads(capture().strip())
    assert parsed["level"] == "INFO"
    assert parsed["message"] == "subscription.md: +2 added, -0 removed, 1 unchanged (version 7)."
 
 
def test_run_complete_summary_message(capsys):
    logger, capture = _ingest_logger_and_capture(capsys)
    logger.info(
        "Ingestion run complete: %d chunks added, %d removed, version=%d.",
        5, 0, 7,
    )
    parsed = json.loads(capture().strip())
    assert parsed["level"] == "INFO"
    assert parsed["message"] == "Ingestion run complete: 5 chunks added, 0 removed, version=7."
 