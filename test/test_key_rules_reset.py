"""Offline checks for opt-in upstream quota reset times."""
import asyncio
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.key_rules as key_rules

NOW = datetime.fromisoformat("2026-11-02T04:00:00+00:00").timestamp()


def quota_rules(duration=172800, enabled=True):
    return key_rules.resolve_key_rules({"key_rules": [
        {"match": {"status": [402], "keyword": ["余额不足"]}, "duration": -1, "retry": True},
        {"match": {"status": [401, 403]}, "duration": -1},
        {"match": {"keyword": ["使用上限"]}, "duration": duration, "quota_reset": enabled, "retry": True},
        {"match": "default", "duration": 3},
    ]})


@pytest.mark.parametrize("text,expected", [
    ("限额将在 2026-11-02 12:54:47 重置", 3347),
    ("限额将在 2026-11-02T05:00:00Z 重置", 3660),
    ("重置时间为：2026-11-02T13:00:00+08:00", 3660),
    ("resets at 2026-11-02T00:00:00-05:00", 3660),
    ("reset_at: 2026-11-02T05:00:00+0000", 3660),
    ("限额将在 2026-11-03 重置", 43260),
])
def test_explicit_reset_time_overrides_fallback_with_one_minute(monkeypatch, text, expected):
    monkeypatch.setattr(key_rules, "time", lambda: NOW)
    result = key_rules.match_key_rules(quota_rules(), 429, "每周/每月使用上限；" + text)
    assert result["duration"] == expected
    assert result["quota_reset"] is True
    assert result["retry"] is True


@pytest.mark.parametrize("text", [
    "没有提供恢复日期",
    "请求时间 2026-11-03 12:00:00，不代表重置时间",
    "限额将在 2026-13-02 12:54:47 重置",
    "限额将在 2026-11-02 12:54:bad 重置",
    "限额将在 2026-11-02 12:54:47+99:99 重置",
    "限额将在 2026-11-02 11:00:00 重置",
    "限额将在 2026-11-02 12:00:00 重置",
])
@pytest.mark.parametrize("fallback", [172800, 604800])
def test_unusable_reset_time_preserves_period_fallback(monkeypatch, text, fallback):
    monkeypatch.setattr(key_rules, "time", lambda: NOW)
    assert key_rules.match_key_rules(quota_rules(fallback), 429, "使用上限；" + text)["duration"] == fallback


@pytest.mark.parametrize("enabled", [False, "true", 1, None])
def test_reset_action_is_explicit_boolean_opt_in(monkeypatch, enabled):
    monkeypatch.setattr(key_rules, "time", lambda: NOW)
    result = key_rules.match_key_rules(quota_rules(enabled=enabled), 429, "使用上限；限额将在 2026-11-03 重置")
    assert result["duration"] == 172800
    assert "quota_reset" not in result


def test_existing_permanent_and_normal_rate_limit_rules_are_unchanged(monkeypatch):
    monkeypatch.setattr(key_rules, "time", lambda: NOW)
    rules = quota_rules()
    error = "使用上限；限额将在 2026-11-03 重置"
    for status, text in [(402, error), (429, "余额不足；" + error), (401, error), (403, error)]:
        result = key_rules.match_key_rules(rules, status, text)
        assert result["duration"] == -1
        assert "quota_reset" not in result
    assert key_rules.match_key_rules(rules, 429, "请求频率过高")["duration"] == 3


def test_committed_stream_uses_actual_key_and_same_dynamic_duration(monkeypatch):
    import core.utils as utils
    from core.stream_errors import record_stream_failure

    monkeypatch.setattr(key_rules, "time", lambda: NOW)
    pool = SimpleNamespace(set_auto_disabled=AsyncMock(), lock=asyncio.Lock(), _sticky_sessions={}, items=["test-request-key"])
    monkeypatch.setitem(utils.provider_api_circular_list, "glm-test", pool)
    info = {"_stream_committed": True, "_used_api_key": "test-request-key",
            "_provider_cfg": {"provider": "glm-test", "preferences": {"key_rules": quota_rules()}}}
    text = "使用上限；限额将在 2026-11-02 12:54:47 重置"
    error = {"status_code": 429, "message": text, "rule_text": text}
    asyncio.run(record_stream_failure(info, error))
    pool.set_auto_disabled.assert_awaited_once_with("test-request-key", duration=3347, reason="keyword=使用上限")
    asyncio.run(record_stream_failure(info, error))
    assert pool.set_auto_disabled.await_count == 1
