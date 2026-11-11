"""Offline regressions for the seven scoped review-lite ports (fake keys only)."""
import ast
import asyncio
import base64
import json
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import BackgroundTasks, FastAPI, HTTPException
from starlette.websockets import WebSocketDisconnect

import core.auth as auth
import core.handler as handler_module
import core.passthrough as passthrough_module
import core.stats as stats_module
from core.channels.openai_channel import get_gpt_payload
from core.client_manager import ClientManager
from core.dialects import openai_responses_ws as ws
from core.dialects.registry import EndpointDefinition
from core.dialects.router import _create_generic_handler
from core.models import RequestModel
from core.plugins.interceptors import InterceptorRegistry
from core.streaming import LoggingStreamingResponse
from core.utils import is_valid_model_name, scrub_sensitive_error_text
from plugins import file_to_text
from routes import admin, auth as auth_routes, setup, workspace
from test_handler_retry_lifecycle import MODEL, Pool, make_case, provider, request


@pytest.mark.parametrize("entries,expected", [
    ([{"api": "fake-user", "role": "user"}], None),
    ([{"api": "fake-legacy"}], 0),
    ([{"api": "fake-user", "role": "user"}, {"api": "fake-admin", "role": "ADMIN"}], 1),
    ([{"api": "fake-one"}, {"api": "fake-two"}], None),
    ([{"role": "admin"}, {"api": "fake-user", "role": "user"}], None),
    ([], None),
])
def test_admin_selection_uses_shared_rules(entries, expected):
    app = SimpleNamespace(state=SimpleNamespace(api_keys_db=entries))
    key = entries[expected]["api"] if expected is not None else None
    assert auth._resolve_admin_api_index(app) == expected
    assert auth_routes._select_admin_api_key_from_config({"api_keys": entries}) == key
    assert setup._select_admin_api_key_from_config({"api_keys": entries}) == key


@pytest.mark.asyncio
async def test_generator_requires_admin_without_writing_keys(monkeypatch):
    app = FastAPI()
    app.include_router(admin.router)
    entries = [{"api": "fake-user", "role": "user"}, {"api": "fake-admin", "role": "admin"}]
    app.state.admin_initialized = True
    app.state.api_keys_db = entries
    app.state.api_list = [entry["api"] for entry in entries]
    app.state.admin_api_key = auth.compute_admin_api_keys(entries)
    app.state.global_rate_limit = {60: 10000}
    monkeypatch.setattr(auth, "rate_limiter", SimpleNamespace(is_rate_limited=AsyncMock(return_value=False)))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        for token, status in [(None, 403), ("fake-user", 403), ("fake-admin", 200)]:
            response = await client.get("/v1/generate-api-key", headers={"x-api-key": token} if token else {})
            assert response.status_code == status
    assert entries == [{"api": "fake-user", "role": "user"}, {"api": "fake-admin", "role": "admin"}]


@pytest.mark.parametrize("name", ["gpt-4o", "org/model:latest@v2", "prefix/vendor/model", "模型-1", "a.b-c_1"])
def test_legal_model_names(name):
    assert is_valid_model_name(name)
    assert is_valid_model_name(name, check_encoded=True)


@pytest.mark.parametrize("name", ["", "../x", "x..y", "a\\b", "a?key=fake", "a#fragment", "a\x00b", "a\nb", "a\x7fb", "a\x85b"])
@pytest.mark.asyncio
async def test_shared_model_entry_rejects_bad_names(make_case, name):
    case = make_case([provider("p0")], {"p0": Pool()})
    with pytest.raises(HTTPException) as exc:
        await case.handler.request_model(RequestModel(model=name, messages=[]), 0, None)
    assert exc.value.status_code == 400
    case.router.assert_not_awaited()


@pytest.mark.parametrize("name", ["%2e%2e/foo", "%252e%252e/x", "a%5cb", "a%2Fb", "x%3Fy", "x%23y", "x%00"])
@pytest.mark.asyncio
async def test_model_path_rejects_encoded_traversal(name):
    route = _create_generic_handler("gemini", EndpointDefinition(path="/v1beta/models/{model:path}", methods=["POST"]))
    req = SimpleNamespace(json=AsyncMock(return_value={}), headers={}, path_params={"model": name},
                          url=SimpleNamespace(path="/v1beta/models/" + name))
    response = await route(req, BackgroundTasks(), api_index=0)
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_upstream_redirects_disabled_even_when_default_enabled():
    seen = []
    async def upstream(req):
        seen.append(str(req.url))
        return httpx.Response(302, headers={"location": "https://elsewhere.invalid/secret"})
    manager = ClientManager()
    await manager.init({"transport": httpx.MockTransport(upstream), "follow_redirects": True})
    try:
        async with manager.get_client("https://test.invalid") as client:
            response = await client.get("https://test.invalid/a", timeout=900)
            assert client.follow_redirects is False
            assert client.event_hooks["request"]
        assert response.status_code == 302 and len(seen) == 1
    finally:
        await manager.close()
    # Inspect startup defaults without importing main (which owns real startup/config I/O).
    tree = ast.parse(Path(__file__).resolve().parents[1].joinpath("main.py").read_text(encoding="utf-8"))
    values = [v.value for n in ast.walk(tree) if isinstance(n, ast.Dict)
              for k, v in zip(n.keys, n.values) if isinstance(k, ast.Constant) and k.value == "follow_redirects"]
    assert values and values == [False]


def test_scrub_sensitive_error_text():
    text = "fake-current-key https://test.invalid/?key=fake-query&x=1 Bearer fake-bearer-token-123 sk-12345678901234567890"
    clean = scrub_sensitive_error_text(text, ["fake-current-key"])
    assert "fake-current-key" not in clean and "fake-query" not in clean
    assert "fake-bearer-token-123" not in clean and "sk-12345678901234567890" not in clean
    assert "&x=1" in clean
    assert scrub_sensitive_error_text("short fake key", ["fake", "*"]) == "short *** key"


@pytest.mark.asyncio
async def test_retry_scrubs_current_key_but_matches_raw_error(make_case, monkeypatch, caplog):
    keys = {"p0": "fake-current-upstream-A", "p1": "fake-current-upstream-B"}
    force = "fake-force-credential"
    calls = []
    async def process(data, chosen):
        assert case.info.get("_used_api_key") is None
        key = keys[chosen["provider"]]
        case.info["_used_api_key"] = key
        raise HTTPException(503, f"quota failure {key} {force}")
    providers = [provider("p0", None), provider("p1", None)]
    case = make_case(providers, {"p0": Pool(), "p1": Pool()}, process=process)
    import core.key_rules as rules
    original = rules.match_key_rules
    def capture(rule_list, status, text):
        calls.append(text)
        return original(rule_list, status, text)
    monkeypatch.setattr(rules, "match_key_rules", capture)
    response = await case.handler.request_model(request(), 0, None, force_api_key=force)
    output = response.body.decode() + case.info["retry_path"] + caplog.text
    assert case.attempts == ["p0", "p1"]
    assert len(calls) == 2 and all(keys[p] in calls[i] for i, p in enumerate(keys))
    assert all(key not in output for key in [*keys.values(), force])
    assert len(case.stats) == 1


@pytest.mark.asyncio
async def test_byok_error_scrub_uses_real_key(make_case):
    real = "fake-byok-real-credential"
    async def process(data, chosen):
        case.info["_used_api_key"] = "*"
        raise HTTPException(401, f"Invalid credential {real}")
    p = provider("byok", None)
    p["api"] = ["*"]
    case = make_case([p], {}, process=process)
    case.info["_byok_real_key"] = real
    response = await case.handler.request_model(request(), 0, None)
    assert real not in response.body.decode() + case.info["retry_path"]


@pytest.mark.parametrize("path", ["data/.git/config", "plugins/__pycache__/nested/readme.txt", "docs/.env.local/child", "data/secret.db-wal/child"])
def test_workspace_sensitive_parents_denied(tmp_path, monkeypatch, path):
    monkeypatch.setattr(workspace, "PROJECT_ROOT", tmp_path)
    assert workspace._get_permissions(path) == ""
    assert workspace._get_permissions("docs/plain.md") == "rwd"
    assert workspace._get_permissions(".env") == "rw"  # existing explicit whitelist remains


@pytest.mark.parametrize("outside", [True, False])
def test_workspace_symlink_components_denied(tmp_path, monkeypatch, outside):
    root = tmp_path / "project"
    target = tmp_path / "project-other" if outside else root / "real"
    root.mkdir()
    target.mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir()
    monkeypatch.setattr(workspace, "PROJECT_ROOT", root)
    link = root / "data" / "link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink permission unavailable: {exc}")
    assert workspace._get_permissions("data/link/new.txt") == ""
    with pytest.raises(HTTPException) as exc:
        workspace._resolve_path("data/link/new.txt")
    assert exc.value.status_code == 403


def test_workspace_real_path_uses_component_membership(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    escaped = tmp_path / "project-other" / "plain.txt"
    original = Path.resolve
    def resolve(path, *args, **kwargs):
        return escaped if path == root / "data/plain.txt" else original(path, *args, **kwargs)
    monkeypatch.setattr(workspace, "PROJECT_ROOT", root)
    monkeypatch.setattr(Path, "resolve", resolve)
    with pytest.raises(HTTPException):
        workspace._resolve_path("data/plain.txt")
    assert workspace._get_permissions("data/plain.txt") == ""


@pytest.mark.parametrize("kind", ["base64", "data-base64", "data-text"])
def test_inline_file_limits_cover_both_encoded_and_decoded(kind, monkeypatch):
    assert file_to_text.MAX_FILE_BASE64_CHARS == 8 * 1024 * 1024
    assert file_to_text.MAX_DECODED_TEXT_BYTES == 5 * 1024 * 1024
    monkeypatch.setattr(file_to_text, "MAX_FILE_BASE64_CHARS", 80)
    monkeypatch.setattr(file_to_text, "MAX_DECODED_TEXT_BYTES", 30)
    def encode(text):
        if kind == "data-text":
            return "data:text/plain," + text
        encoded = base64.b64encode(text.encode()).decode()
        return ("data:text/plain;base64," if kind == "data-base64" else "") + encoded
    assert file_to_text._decode_content(encode("正常")) == "正常"
    assert file_to_text._decode_content(encode("a" * 30)) == "a" * 30
    for value in [encode("a" * 31), encode("中" * 11), "a" * 81]:
        with pytest.raises(HTTPException) as exc:
            file_to_text._decode_content(value)
        assert exc.value.status_code == 413
    assert file_to_text._decode_content("not valid base64!") is None


@pytest.mark.asyncio
async def test_file_size_rejection_not_swallowed_by_plugin_registry(monkeypatch):
    monkeypatch.setattr(file_to_text, "MAX_FILE_BASE64_CHARS", 8)
    registry = InterceptorRegistry()
    registry.register_request_interceptor("file-port-test", file_to_text._file_to_text_interceptor,
                                          plugin_name="file_to_text")
    payload = {"messages": [{"role": "user", "content": [{"type": "input_file", "filename": "fake.txt", "file_data": "a" * 9}]}]}
    with pytest.raises(HTTPException) as exc:
        await registry.apply_request_interceptors(None, "openai", {}, None, "https://test.invalid", {}, payload,
                                                  enabled_plugins=["file_to_text"])
    assert exc.value.status_code == 413


@pytest.mark.asyncio
async def test_ws_frame_bytes_and_shared_global_rate_limit(monkeypatch):
    allowed = json.dumps({"type": "response.create", "model": "test"})
    frames = iter(["中" * 40, allowed, allowed])
    errors, handled = [], []
    class Socket:
        app = SimpleNamespace(state=SimpleNamespace(global_rate_limit="1/min", admin_initialized=True, api_list=["fake-client"]))
        headers = {"x-forwarded-for": "1.1.1.1, 2.2.2.2"}
        client = SimpleNamespace(host="127.0.0.1")
        async def accept(self): pass
        async def receive_text(self):
            try: return next(frames)
            except StopIteration: raise WebSocketDisconnect()
    socket = Socket()
    monkeypatch.setattr(ws, "MAX_WS_FRAME_BYTES", 80)
    monkeypatch.setattr(ws, "_authenticate", lambda _: (0, "fake-client", None))
    monkeypatch.setattr(ws, "_send_error_frame", AsyncMock(side_effect=lambda *a, **kw: errors.append(a[1])))
    monkeypatch.setattr(ws, "_handle_response_create", AsyncMock(side_effect=lambda *a: handled.append(a)))
    limiter = AsyncMock(side_effect=[True, False])
    monkeypatch.setattr(auth, "rate_limiter", SimpleNamespace(is_rate_limited=limiter))
    await ws.responses_websocket(socket)
    assert errors == [413, 429] and len(handled) == 1
    assert len(limiter.await_args_list) == 2
    assert all(call.args == ("global", "1/min") for call in limiter.await_args_list)
    assert handled[0][-1] == "2.2.2.2"


@pytest.mark.asyncio
@pytest.mark.parametrize("host", ["test.invalid", "api.x.ai"])
@pytest.mark.parametrize("stream", [True, False])
async def test_openai_stream_usage_option_merged(host, stream):
    req = SimpleNamespace(model="test", messages=[], model_dump=lambda **kw: {
        "model": "test", "messages": [], "stream": stream,
        "stream_options": {"include_usage": False, "custom": "keep"},
    })
    p = {"base_url": f"https://{host}/v1", "model": ["test"], "engine": "openai"}
    _, _, payload = await get_gpt_payload(req, "openai", p, "fake-upstream")
    if host == "api.x.ai" and not stream:
        assert "stream_options" not in payload
    else:
        assert payload["stream_options"] == {"include_usage": stream, "custom": "keep"}


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,stream", [(None, True), ("force_stream", False), ("force_non_stream", True)])
async def test_passthrough_openai_stream_options_merged(monkeypatch, mode, stream):
    captured = []
    async def upstream(req):
        payload = json.loads(req.content)
        captured.append(payload)
        if payload.get("stream"):
            return httpx.Response(200, text='data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n',
                                  headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})
    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
        @asynccontextmanager
        async def get_client(*args):
            yield client
        app = FastAPI()
        app.state.config = {}
        app.state.client_manager = SimpleNamespace(get_client=get_client)
        info = {"request_id": "fake-rid", "api_key": "fake-client", "model": "test"}
        p = {"provider": "fake-channel", "api": [], "engine": "openai", "model": ["test"],
             "_model_dict_cache": {"test": "test"}, "base_url": "https://test.invalid/v1"}
        if mode:
            p["preferences"] = {"stream_mode": mode}
        ctx = SimpleNamespace(dialect_id="openai", original_headers={}, modifications={},
                              original_payload={"model": "test", "messages": [], "stream": stream,
                                                "stream_options": {"include_usage": False, "custom": "keep"}})
        monkeypatch.setattr(handler_module, "_fire_and_forget_channel_stats", lambda *a, **kw: None)
        response = await passthrough_module.process_request_passthrough(
            RequestModel(model="test", messages=[], stream=stream), p, None, app, lambda: info,
            AsyncMock(), ctx, keepalive_interval=None)
        await response.close()
    assert len(captured) == 1
    assert captured[0]["stream_options"] == {"include_usage": mode != "force_non_stream", "custom": "keep"}


@pytest.mark.asyncio
async def test_handler_records_text_part_prompt_estimate(make_case):
    case = make_case([provider("p0")], {"p0": Pool()})
    req = RequestModel(model=MODEL, messages=[{"role": "user", "content": [{"type": "text", "text": "abcdefgh"}]}])
    await case.handler.request_model(req, 0, None)
    assert case.info["_estimated_prompt_tokens"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("real_usage,disconnect", [(None, True), (None, False), ({"prompt_tokens": 11, "completion_tokens": 7}, True), ({"prompt_tokens": 0, "completion_tokens": 0}, True)])
async def test_stream_real_usage_wins_and_499_is_estimated_once(monkeypatch, real_usage, disconnect):
    records = []
    channel_records = []
    info = {"start_time": 0, "success": True, "status_code": 200, "_estimated_prompt_tokens": 100,
            "_channel_stats_call": (None, (), {})}
    monkeypatch.setattr(handler_module, "_fire_and_forget_channel_stats", lambda *a, **kw: channel_records.append(kw))
    monkeypatch.setattr("core.streaming.enqueue_stats", lambda data, **kw: records.append(dict(data)))
    async def content():
        usage = {"usage": real_usage} if real_usage is not None else {}
        yield "data: " + json.dumps({"choices": [{"delta": {"content": "hello world"}}], **usage}) + "\n\n"
        yield "data: [DONE]\n\n"
    response = LoggingStreamingResponse(content(), media_type="text/event-stream", current_info=info)
    async def send(event):
        if disconnect and event["type"] == "http.response.body" and event.get("body"):
            raise ConnectionError("fake client disconnect")
    await response({}, AsyncMock(), send)
    assert len(records) == len(channel_records) == 1
    assert channel_records[0]["success"] is (not disconnect)
    result = records[0]
    assert result["success"] is (not disconnect)
    assert result["status_code"] == (499 if disconnect else 200)
    if real_usage is not None:
        assert result.get("prompt_tokens", 0) == real_usage["prompt_tokens"]
        assert result.get("completion_tokens", 0) == real_usage["completion_tokens"]
        assert not result.get("_usage_estimated")
    else:
        assert result["prompt_tokens"] == 100 and result["_usage_estimated"] is True
        assert result["total_tokens"] == result["prompt_tokens"] + result["completion_tokens"]
        assert result["completion_tokens"] == 0 if disconnect else result["completion_tokens"] > 0
        assert result["prompt_price"] == result["completion_price"] == 0


@pytest.mark.asyncio
async def test_499_before_content_estimates_only_prompt(monkeypatch):
    records = []
    info = {"success": True, "status_code": 200, "_estimated_prompt_tokens": 15}
    monkeypatch.setattr("core.streaming.enqueue_stats", lambda data, **kw: records.append(dict(data)))
    async def content():
        yield ": keepalive\n\n"
        raise asyncio.CancelledError()
    response = LoggingStreamingResponse(content(), media_type="text/event-stream", current_info=info)
    with pytest.raises(asyncio.CancelledError):
        await response({}, AsyncMock(), AsyncMock())
    assert len(records) == 1
    assert records[0]["status_code"] == 499 and records[0]["success"] is False
    assert records[0]["prompt_tokens"] == records[0]["total_tokens"] == 15
    assert records[0]["completion_tokens"] == 0


@pytest.mark.asyncio
async def test_adapter_partial_real_usage_is_not_estimated(monkeypatch):
    records = []
    info = {"success": True, "status_code": 200, "_estimated_prompt_tokens": 100,
            "adapter_metrics_managed": True, "prompt_tokens": 11, "completion_tokens": 0, "total_tokens": 11}
    monkeypatch.setattr("core.streaming.enqueue_stats", lambda data, **kw: records.append(dict(data)))
    async def content():
        yield 'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
    async def send(event):
        if event["type"] == "http.response.body":
            raise ConnectionError("fake client disconnect")
    await LoggingStreamingResponse(content(), media_type="text/event-stream", current_info=info)({}, AsyncMock(), send)
    assert len(records) == 1 and records[0]["prompt_tokens"] == 11
    assert records[0]["completion_tokens"] == 0 and not records[0].get("_usage_estimated")


def test_estimated_usage_not_priced_as_exact(monkeypatch):
    monkeypatch.setattr(stats_module, "DISABLE_DATABASE", False)
    monkeypatch.setattr(stats_module, "_stats_buffer", deque())
    monkeypatch.setattr(stats_module, "_ensure_stats_consumer_started", lambda: None)
    price = Mock(return_value=(10, 20, 5))
    info = {"success": True, "model": "test", "prompt_tokens": 100, "completion_tokens": 5,
            "_usage_estimated": True, "prompt_price": 999, "completion_price": 999}
    stats_module.enqueue_stats(info, get_model_prices_func=price)
    price.assert_not_called()
    snapshot = stats_module._stats_buffer[0][0]
    assert snapshot["prompt_price"] == snapshot["completion_price"] == 0
    info.pop("_usage_estimated")
    stats_module.enqueue_stats(info, get_model_prices_func=price)
    assert stats_module._stats_buffer[-1][0]["prompt_price"] == 10
