"""Pytest tests for logging_config.configure_logging() (R6).
 
Run from the repo root:
 
    pytest test/test_logging_smoke.py -v
 
Unlike the original standalone script, these assert on the actual JSON
output rather than just printing it for manual inspection. No
Weaviate/Neo4j/Redis/Groq needed — this only exercises the logging setup.
"""
 
from __future__ import annotations
 
import json
import logging
 
import pytest
 
from logging_config import configure_logging
 
 
@pytest.fixture(autouse=True)
def _reset_root_handlers():
    """configure_logging() attaches to the root logger, which persists
    across tests unless we clean it up. Removes anything this module's
    handler marker added, before AND after each test, so tests don't leak
    handlers into each other or into the rest of the suite."""
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    yield
    root.handlers = original_handlers
    root.setLevel(original_level)
 
 
def _configure_and_capture(capsys, level: str = "INFO"):
    """Configures logging at the given level. Returns a helper that emits
    a log line and returns the captured stdout for that line."""
    configure_logging(level)
 
    def emit(logger_name: str, log_level: str, message: str) -> str:
        capsys.readouterr()  # clear anything captured so far
        logger = logging.getLogger(logger_name)
        getattr(logger, log_level.lower())(message)
        return capsys.readouterr().out
 
    return emit
 
 
def test_configure_logging_produces_valid_json(capsys):
    emit = _configure_and_capture(capsys)
    output = emit("agent.llm_client", "warning", "test warning message")
 
    line = output.strip()
    assert line, "expected a log line on stdout, got nothing"
    parsed = json.loads(line)  # raises if it's not valid JSON
 
    assert parsed["message"] == "test warning message"
    assert parsed["level"] == "WARNING"
    assert parsed["logger"] == "agent.llm_client"
    assert "timestamp" in parsed
 
 
def test_configure_logging_preserves_logger_name(capsys):
    emit = _configure_and_capture(capsys)
    output = emit("ingestion.ingest", "info", "test info message")
 
    parsed = json.loads(output.strip())
    assert parsed["logger"] == "ingestion.ingest"
    assert parsed["level"] == "INFO"
 
 
def test_configure_logging_is_idempotent(capsys):
    """Calling configure_logging() twice must not attach a second handler —
    otherwise every log line would be duplicated."""
    configure_logging("INFO")
    configure_logging("INFO")  # called again, deliberately
 
    capsys.readouterr()  # clear setup noise
    logging.getLogger("ingestion.ingest").info("duplicate handler check")
    output = capsys.readouterr().out
 
    lines = [line for line in output.strip().splitlines() if line]
    assert len(lines) == 1, (
        f"expected exactly 1 log line after calling configure_logging() twice, "
        f"got {len(lines)}: {lines}"
    )
 
 
def test_log_level_filters_root_logger(capsys):
    """At LOG_LEVEL=WARNING, an INFO call must be suppressed and a WARNING
    call must still come through."""
    configure_logging("WARNING")
    logger = logging.getLogger("agent.escalation")
 
    capsys.readouterr()
    logger.info("this INFO line should NOT appear")
    suppressed_output = capsys.readouterr().out
    assert suppressed_output.strip() == "", (
        f"LOG_LEVEL=WARNING should suppress INFO lines, but got: {suppressed_output!r}"
    )
 
    logger.warning("this WARNING line SHOULD appear")
    warning_output = capsys.readouterr().out
    assert warning_output.strip(), "expected the WARNING line to be emitted"
    parsed = json.loads(warning_output.strip())
    assert parsed["level"] == "WARNING"
 