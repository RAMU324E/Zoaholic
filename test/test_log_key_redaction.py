"""No real credentials or network calls: log output and admin buffer regressions."""
import asyncio
import io
import logging
import logging.handlers
import queue
import sys
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.key_pool as key_pool
import core.log_config as logs

KEY = "synthetic-upstream-secret/with+encoding="
BYOK = "synthetic-byok-secret-not-in-pool"


@pytest.fixture
def configured_key(monkeypatch):
    monkeypatch.setattr(key_pool, "provider_api_circular_list", {"test": SimpleNamespace(items=[KEY])})


def test_redaction_keeps_stable_distinct_fingerprints(configured_key):
    text = f"Error 429 provider glm API key: {KEY}; https://example.invalid/?key={quote(KEY, safe='')}"
    result = logs.redact_log_text(text)
    assert KEY not in result
    assert quote(KEY, safe="") not in result
    assert result.count(logs.key_fingerprint(KEY)) == 2
    assert "Error 429 provider glm" in result
    assert logs.key_fingerprint(KEY) != logs.key_fingerprint(BYOK)
    assert logs.redact_log_text(result) == result


def test_queue_prepared_traceback_is_redacted_in_output_and_buffer(configured_key):
    before = logs.get_backend_log_entries()["max_id"]
    stream = io.StringIO()
    handler = logs.BackendCaptureStreamHandler(stream, "stderr")
    handler.setFormatter(logs.RedactingFormatter(logs.LOG_FORMAT))
    try:
        raise ValueError(f"upstream echoed {KEY}; BYOK echoed {BYOK}")
    except ValueError:
        record = logging.LogRecord("redaction-test", logging.ERROR, __file__, 1,
                                   "provider glm status 429: %s", (KEY,), sys.exc_info())
    record.sensitive_keys = (BYOK,)
    # Production QueueHandler formats exceptions into msg before the listener sees them.
    prepared = logging.handlers.QueueHandler(queue.Queue()).prepare(record)
    handler.handle(prepared)
    output = stream.getvalue()
    entries = logs.get_backend_log_entries(since_id=before, logger_name="redaction-test")["items"]
    assert len(entries) == 1
    for text in [output, entries[0]["message"]]:
        assert KEY not in text and BYOK not in text
        assert logs.key_fingerprint(KEY) in text
        assert logs.key_fingerprint(BYOK) in text
        assert "ValueError" in text and "status 429" in text


def test_stdout_tee_redacts_both_destinations(configured_key):
    before = logs.get_backend_log_entries()["max_id"]
    stream = io.StringIO()
    tee = logs.TeeStream(stream, "stdout")
    text = f"upstream error {KEY}\n"
    assert tee.write(text) == len(text)
    tee.flush()
    assert KEY not in stream.getvalue()
    assert logs.key_fingerprint(KEY) in stream.getvalue()
    entries = logs.get_backend_log_entries(since_id=before)["items"]
    assert entries and KEY not in entries[-1]["message"]


def test_pool_disabled_log_never_emits_key_even_without_registry(monkeypatch):
    messages = []
    monkeypatch.setattr(key_pool, "logger", SimpleNamespace(warning=messages.append))
    monkeypatch.setattr(key_pool, "provider_api_circular_list", {})
    monkeypatch.setattr(key_pool, "_save_all_auto_disabled", lambda: None)
    pool = key_pool.ThreadSafeCircularList([KEY], provider_name="test")
    asyncio.run(pool.set_auto_disabled(KEY, duration=60, reason=f"upstream echoed {KEY}"))
    assert len(messages) == 1
    assert KEY not in messages[0]
    assert logs.key_fingerprint(KEY) in messages[0]
    assert "duration=60s" in messages[0]
    assert pool.auto_disabled_info[KEY]["duration"] == 60
