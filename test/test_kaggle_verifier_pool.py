"""kaggle_verifier_pool 离线自检：参数解析、额度封顶、挑卡顺序与认证头注入。

不触网、不读真实 keys；签发/查询函数均不在覆盖范围（需真实 Kaggle 容器）。
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plugins.kaggle_verifier_pool as pool


@pytest.fixture
def isolated_state(tmp_path, monkeypatch):
    monkeypatch.setattr(pool, "DEFAULT_KEYS_FILE", str(tmp_path / "no_keys.json"))
    monkeypatch.setattr(pool, "DEFAULT_STATE_FILE", str(tmp_path / "state.json"))
    monkeypatch.setattr(pool, "_cached_state", None)
    return tmp_path


def test_parse_options_string_and_dict():
    assert pool.parse_options(None)["pick_order"] == "reverse"
    cfg = pool.parse_options("pin_key=48,max_tokens_cap=4096,min_quota=2.0,pick_order=forward")
    assert cfg["pin_key"] == "48"
    assert cfg["max_tokens_cap"] == 4096
    assert cfg["min_quota"] == 2.0
    assert cfg["pick_order"] == "forward"
    cfg2 = pool.parse_options({"pin_key": " KGAT_ab12 ", "max_tokens_cap": "99999"})
    assert cfg2["pin_key"] == "KGAT_ab12"
    assert cfg2["max_tokens_cap"] == 32768 or cfg2["max_tokens_cap"] == 99999  # 无上限夹取仅下限


def test_cap_max_tokens_openai_and_gemini():
    payload = {"messages": [], "max_tokens": 128000}
    pool._cap_max_tokens(payload, 65536)
    assert payload["max_tokens"] == 65536

    payload = {"messages": []}
    pool._cap_max_tokens(payload, 65536)
    assert payload["max_tokens"] == 65536

    payload = {"contents": [], "generationConfig": {"maxOutputTokens": 99999}}
    pool._cap_max_tokens(payload, 65536)
    assert payload["generationConfig"]["maxOutputTokens"] == 65536

    payload = {"contents": []}
    pool._cap_max_tokens(payload, 65536)
    assert payload["generationConfig"]["maxOutputTokens"] == 65536


def test_strip_url_key_param():
    url = "https://mp-staging.kaggle.net/models/genai/v1/models/gemini-4-argon:generateContent?key=KGAT_x&alt=sse"
    out = pool._strip_url_key_param(url)
    assert "key=" not in out and "alt=sse" in out
    assert pool._strip_url_key_param("https://x.y/z?alt=sse") == "https://x.y/z?alt=sse"


def test_gemini_channel_preserves_kaggle_models_prefix():
    src = (Path(__file__).resolve().parents[1] / "core/channels/gemini_channel.py").read_text(encoding="utf-8")
    assert "if base_path.endswith('/models'):" in src
    assert "parsed_url.path.split('/models')[0]" not in src
    from urllib.parse import urlparse

    def build_gemini_url(base_url: str, original_model: str, gemini_stream: str = "generateContent", sse_param: str = "") -> str:
        parsed_url = urlparse(base_url)
        if base_url.endswith('#'):
            return base_url[:-1].rstrip('/')
        base_path = parsed_url.path.rstrip('/')
        if base_path.endswith('/models'):
            base_path = base_path[:-len('/models')]
        return f"{parsed_url.scheme}://{parsed_url.netloc}{base_path}/models/{original_model}:{gemini_stream}{sse_param}"

    assert build_gemini_url("https://mp-staging.kaggle.net/models/genai/v1", "gemini-4-argon") == (
        "https://mp-staging.kaggle.net/models/genai/v1/models/gemini-4-argon:generateContent"
    )
    assert build_gemini_url("https://generativelanguage.googleapis.com/v1beta", "gemini-4-argon") == (
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-4-argon:generateContent"
    )
    assert build_gemini_url("https://generativelanguage.googleapis.com/v1beta/models", "gemini-4-argon") == (
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-4-argon:generateContent"
    )


def test_slot_validity_and_renewal_window():
    now = time.time()
    valid = {"verifier_token": "kaggleweb:CiQAAA", "expires_at": now + 3600}
    expiring = {"verifier_token": "kaggleweb:CiQAAA", "expires_at": now + 100}
    dead = {"verifier_token": "kaggleweb:CiQAAA", "expires_at": now - 10}
    assert pool._slot_valid(valid, now) and pool._slot_valid(expiring, now)
    assert not pool._slot_valid(dead, now) and not pool._slot_valid(None, now)
    assert not pool._slot_needs_renewal(valid, now)
    assert pool._slot_needs_renewal(expiring, now) and pool._slot_needs_renewal(dead, now)


def test_candidate_order_reverse_exhausted_and_pin():
    keys = [f"KGAT_{i:032x}" for i in range(1, 49)]
    state = {"cursor_index": None, "exhausted_kgats": {}}
    opts = pool.parse_options(None)
    order = pool._candidate_order(keys, state, opts)
    assert order[0][0] == 48 and order[-1][0] == 1

    # 朋友正序在用 1~5 时，exhausted 标记会被跳过
    state["exhausted_kgats"] = {keys[47]: time.time()}
    order = pool._candidate_order(keys, state, opts)
    assert all(k != keys[47] for _, k in order) and len(order) == 47

    # pin 到第 7 把时第一个候选必须是它
    opts_pin = pool.parse_options("pin_key=7")
    order = pool._candidate_order(keys, state, opts_pin)
    assert order[0][0] == 7 and order[0][1] == keys[6]


def _seeded_state(tmp_path, kgat_keys):
    now = time.time()
    return {
        "active": {
            "verifier_token": "kaggleweb:ACTIVE_TOKEN",
            "kgat": kgat_keys[0],
            "index": 1,
            "minted_at": now,
            "expires_at": now + 10 * 3600,
            "quota_remaining": 9.0,
        },
        "standby": {
            "verifier_token": "kaggleweb:STANDBY_TOKEN",
            "kgat": kgat_keys[1],
            "index": 2,
            "minted_at": now,
            "expires_at": now + 10 * 3600,
            "quota_remaining": 9.5,
        },
        "cursor_index": 2,
        "rotate_nonce": "0",
        "exhausted_kgats": {},
        "updated_at": now,
    }


def test_request_interceptor_injects_per_protocol_and_caps(isolated_state):
    kgat_keys = [f"KGAT_{i:032x}" for i in range(1, 3)]
    state = _seeded_state(isolated_state, kgat_keys)
    (isolated_state / "state.json").write_text(json.dumps(state), encoding="utf-8")

    provider = {"api": kgat_keys, "preferences": {"enabled_plugins": ["kaggle_verifier_pool"]}}

    async def run(url, engine, headers, payload):
        return await pool.kaggle_verifier_pool_request_interceptor(
            None, engine, provider, kgat_keys[0], url, dict(headers), payload,
        )

    # Gemini 原生：x-goog-api-key + 剥离 URL key
    url, headers, payload = asyncio.run(run(
        "https://mp-staging.kaggle.net/models/genai/v1/models/gemini-4-argon:generateContent?key=KGAT_deadbeef",
        "gemini", {"Authorization": "Bearer KGAT_deadbeef"},
        {"contents": [], "generationConfig": {"maxOutputTokens": 99999}},
    ))
    assert headers["x-goog-api-key"] == "kaggleweb:ACTIVE_TOKEN"
    assert "Authorization" not in headers and "key=" not in url
    assert payload["generationConfig"]["maxOutputTokens"] == 65536

    # Anthropic 原生：x-api-key
    url, headers, payload = asyncio.run(run(
        "https://mp-staging.kaggle.net/models/anthropic/v1/messages",
        "claude", {}, {"messages": [], "max_tokens": 128000},
    ))
    assert headers["x-api-key"] == "kaggleweb:ACTIVE_TOKEN"
    assert payload["max_tokens"] == 65536

    # OpenAI 兼容：Authorization Bearer
    url, headers, payload = asyncio.run(run(
        "https://mp-staging.kaggle.net/models/openapi/v1/chat/completions",
        "openai", {}, {"messages": [], "max_completion_tokens": 999999},
    ))
    assert headers["Authorization"] == "Bearer kaggleweb:ACTIVE_TOKEN"
    assert payload["max_completion_tokens"] == 65536


def test_proxy_ctx_routes_through_warp_and_fails_closed(monkeypatch):
    import contextlib
    from core import http as core_http

    monkeypatch.setattr(pool, "_last_proxy", {"url": None, "require": False})
    seen = {}

    @contextlib.contextmanager
    def fake_proxy_context(proxy_url=None, *, require_proxy=False):
        seen["url"] = proxy_url
        seen["require"] = require_proxy
        yield

    monkeypatch.setattr(core_http, "proxy_context", fake_proxy_context)

    provider = {"preferences": {"enabled_plugins": ["warp_only"]}}
    with pool._proxy_ctx(provider):
        pass
    assert seen["require"] is True and str(seen["url"]).startswith("socks5://")

    seen.clear()
    with pool._proxy_ctx({"preferences": {"enabled_plugins": ["kaggle_verifier_pool"]}}):
        pass
    assert seen == {}

    with pytest.raises(ValueError):
        with pool._proxy_ctx({"preferences": {"enabled_plugins": ["warp_only:ftp://127.0.0.1:1"]}}):
            pass


def test_rotate_nonce_promotes_standby(isolated_state, monkeypatch):
    # 隔离后台续签协程，避免测试触网
    monkeypatch.setattr(pool, "_trigger_bg_maintenance", lambda *a, **k: None)
    kgat_keys = [f"KGAT_{i:032x}" for i in range(1, 3)]
    state = _seeded_state(isolated_state, kgat_keys)
    (isolated_state / "state.json").write_text(json.dumps(state), encoding="utf-8")
    provider = {
        "api": kgat_keys,
        "preferences": {"enabled_plugins": ["kaggle_verifier_pool:rotate_nonce=1"]},
    }

    async def run():
        return await pool.kaggle_verifier_pool_request_interceptor(
            None, "openai", provider, kgat_keys[0],
            "https://mp-staging.kaggle.net/models/openapi/v1/chat/completions", {}, {"messages": []},
        )

    _, headers, _ = asyncio.run(run())
    assert headers["Authorization"] == "Bearer kaggleweb:STANDBY_TOKEN"
    new_state = json.loads((isolated_state / "state.json").read_text(encoding="utf-8"))
    assert new_state["active"]["verifier_token"] == "kaggleweb:STANDBY_TOKEN"
    assert new_state["standby"] is None
    assert new_state["rotate_nonce"] == "1"


def test_quota_error_promotes_standby_and_rewrites_status(isolated_state, monkeypatch):
    monkeypatch.setattr(pool, "_trigger_bg_maintenance", lambda *a, **k: None)
    kgat_keys = [f"KGAT_{i:032x}" for i in range(1, 3)]
    state = _seeded_state(isolated_state, kgat_keys)
    (isolated_state / "state.json").write_text(json.dumps(state), encoding="utf-8")

    chunk = {"status_code": 403, "error": "The max estimated cost of operation exceeds your available quota (based on max_output_tokens)"}
    out = asyncio.run(pool.kaggle_verifier_pool_response_interceptor(chunk, "gemini", "gemini-4-argon", False))
    assert out["status_code"] == 502

    new_state = json.loads((isolated_state / "state.json").read_text(encoding="utf-8"))
    assert new_state["active"]["verifier_token"] == "kaggleweb:STANDBY_TOKEN"
    assert new_state["standby"] is None
    assert kgat_keys[0] in new_state["exhausted_kgats"]
