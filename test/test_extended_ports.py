"""Local adaptation contracts: CF priority, dashboard privacy, compatibility, Windows."""
import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from starlette.datastructures import Headers

import core.auth as auth
from core.file_utils import is_link_or_reparse
from core.http_headers import extract_client_ip
from core.ip_blacklist import extract_client_ip as compatibility_extract_ip
from core.middleware import StatsMiddleware
from routes import auth as login_routes, plugins, setup as setup_routes, stats
from core.dialects import openai_responses_ws as ws


@pytest.mark.parametrize("headers,expected", [
    ({"CF-Connecting-IP": " 203.0.113.7 ", "X-Forwarded-For": "198.51.100.1, 198.51.100.2"}, "198.51.100.2"),
    ({"CF-Connecting-IP": " ", "X-Forwarded-For": "198.51.100.1, 198.51.100.2, "}, "198.51.100.2"),
    ({"X-Forwarded-For": ", ", "X-Real-IP": " 198.51.100.3 "}, "198.51.100.3"),
    ({}, "127.0.0.1"),
])
def test_cf_priority_and_all_shared_callers(headers, expected, monkeypatch):
    monkeypatch.delenv("TRUST_CF_CONNECTING_IP", raising=False)
    values = Headers(headers)
    peer = SimpleNamespace(host="127.0.0.1")
    scope = {"client": (peer.host, 1234)}
    raw = [(key.lower().encode(), value.encode()) for key, value in headers.items()]
    assert compatibility_extract_ip is extract_client_ip
    assert extract_client_ip(values, peer.host) == expected
    assert StatsMiddleware._get_client_ip(scope, raw) == expected
    request = SimpleNamespace(headers=values, client=peer)
    assert login_routes._extract_login_client_ip(request) == expected
    assert ws._get_client_ip(request) == expected


def test_cf_header_trusted_only_when_explicitly_enabled(monkeypatch):
    monkeypatch.setenv("TRUST_CF_CONNECTING_IP", "true")
    values = Headers({"CF-Connecting-IP": " 203.0.113.7 ", "X-Forwarded-For": "198.51.100.2"})
    assert extract_client_ip(values, "127.0.0.1") == "203.0.113.7"


def test_login_dictionary_remains_bounded(monkeypatch):
    monkeypatch.setattr(login_routes, "_login_failures_by_ip", {})
    for index in range(4100):
        login_routes._record_login_failure(f"fake-client-{index}")
    assert len(login_routes._login_failures_by_ip) == 4096


@pytest.mark.asyncio
async def test_setup_login_enforces_shared_login_rate_limit(monkeypatch):
    monkeypatch.setattr(login_routes, "_login_failures_by_ip", {})
    monkeypatch.setattr(setup_routes, "ensure_initialized", AsyncMock())
    monkeypatch.setattr(
        setup_routes,
        "_get_admin_user",
        AsyncMock(return_value=SimpleNamespace(username="admin", password_hash="hashed")),
    )
    monkeypatch.setattr(setup_routes, "verify_password", lambda raw, hashed: False)
    req = SimpleNamespace(
        headers=Headers({"CF-Connecting-IP": "1.1.1.1", "X-Forwarded-For": "198.51.100.99"}),
        client=SimpleNamespace(host="127.0.0.1"),
    )
    body = setup_routes.SetupLoginRequest(username="admin", password="wrongpassword")
    for _ in range(5):
        with pytest.raises(HTTPException) as exc:
            await setup_routes.setup_login(req, body)
        assert exc.value.status_code == 403
        req.headers = Headers({"CF-Connecting-IP": f"2.2.2.{_}", "X-Forwarded-For": "198.51.100.99"})
    with pytest.raises(HTTPException) as exc:
        await setup_routes.setup_login(req, body)
    assert exc.value.status_code == 429


@pytest.mark.asyncio
@pytest.mark.parametrize("path", [
    "/v1/stats/dashboard_overview?key_hash=other-key-hash",
    "/v1/stats/key_analytics/summary",
    "/v1/stats/key_analytics/other-key-hash",
])
async def test_dashboard_and_drilldown_never_grant_regular_key_global_access(monkeypatch, path):
    app = FastAPI()
    app.include_router(stats.router)
    entries = [{"api": "fake-admin-private", "role": "admin"}, {"api": "fake-regular-private", "role": "user"}]
    app.state.admin_initialized = True
    app.state.api_keys_db = entries
    app.state.api_list = [entry["api"] for entry in entries]
    app.state.config = {"api_keys": entries}
    app.state.global_rate_limit = [(10000, 60)]
    monkeypatch.setattr(auth, "rate_limiter", SimpleNamespace(is_rate_limited=AsyncMock(return_value=False)))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://local.test") as client:
        response = await client.get(path, headers={"Authorization": "Bearer fake-regular-private"})
    assert response.status_code == 403
    assert "fake-admin-private" not in response.text


def test_legacy_admin_rule_and_public_aliases_remain_compatible():
    entries = [{"api": "fake-legacy-owner"}]
    assert auth.compute_admin_api_keys(entries) == auth.list_admin_api_keys({"api_keys": entries})
    app = SimpleNamespace(state=SimpleNamespace(api_list=["fake-legacy-owner"], api_keys_db=entries))
    assert auth.authenticate_admin_api_key_token(app, "fake-legacy-owner", None, require_explicit_role=True) == "fake-legacy-owner"
    with pytest.raises(HTTPException):
        auth.authenticate_admin_api_key_token(app, "fake-stranger", None)
    import core.handler as handler
    import core.passthrough as passthrough
    import core.response as response
    import core.utils as utils
    assert handler.resolve_oauth_api_key is handler._resolve_oauth_api_key
    assert handler.fire_and_forget_channel_stats is handler._fire_and_forget_channel_stats
    assert passthrough.fetch_passthrough_stream is passthrough._fetch_passthrough_stream
    assert response.apply_response_path_interceptors is response._apply_response_path_interceptors
    assert utils.prepare_image_for_upload is utils._prepare_image_for_upload


def test_short_key_is_not_exposed_in_analytics_label():
    assert stats._key_analytics_prefix("abc") == "***"


@pytest.mark.skipif(sys.platform != "win32", reason="actual NTFS junction requires Windows")
def test_actual_windows_junction_rejected_without_touching_target(tmp_path, monkeypatch):
    import _winapi
    root = tmp_path / "plugins"
    root.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    marker = external / "fake-marker.txt"
    marker.write_text("untouched", encoding="utf-8")
    junction = root / "linked"
    try:
        _winapi.CreateJunction(str(external), str(junction))
    except OSError as exc:
        pytest.skip(f"NTFS junction creation unavailable: winerror={getattr(exc, 'winerror', None)} {exc}")
    monkeypatch.setattr(plugins, "PLUGINS_DIR", root)
    assert is_link_or_reparse(junction)
    with pytest.raises(HTTPException):
        plugins._plugin_upload_target(root.resolve(), "linked", directory=True)
    plugins._remove_upload_target(junction)
    assert marker.read_text(encoding="utf-8") == "untouched"


@pytest.mark.asyncio
async def test_legacy_stats_settles_ancestors_once_even_with_database_disabled(monkeypatch):
    import core.stats as runtime_stats
    registry = SimpleNamespace(record_usage=Mock())
    app = SimpleNamespace(state=SimpleNamespace(quota_registry=registry))
    info = {"api_key": "fake-child", "api_key_id": "child-id", "key_ancestor_ids": ["root-id"],
            "key_ancestor_keys": ["fake-root"], "model": "fake-model", "success": True,
            "prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10, "_usage_estimated": True}
    monkeypatch.setattr(runtime_stats, "DISABLE_DATABASE", True)
    await runtime_stats.update_stats(info, app)
    registry.record_usage.assert_called_once()
    args = registry.record_usage.call_args.kwargs
    assert args["ancestor_ids"] == ["root-id"]
    assert args["ancestor_keys"] == ["fake-root"]
    assert args["cost"] == 0 and args["tokens"] == 10


@pytest.mark.asyncio
@pytest.mark.parametrize('streaming', [False, True])
async def test_public_gemini_image_helper_does_not_shadow_local_response_flag(streaming):
    from core.channels import gemini_channel as channel
    from core.middleware import request_info
    body = {'candidates': [{'content': {'parts': [{'text': 'fake-answer'}]}, 'finishReason': 'STOP'}]}
    def transport(request):
        if streaming:
            return httpx.Response(200, text='data: ' + json.dumps(body) + '\n\n', headers={'content-type': 'text/event-stream'})
        return httpx.Response(200, json=body)
    token = request_info.set({})
    try:
        async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
            function = channel.fetch_gemini_response_stream if streaming else channel.fetch_gemini_response
            output = [item async for item in function(client, 'https://local.invalid/fake', {}, {}, 'gemini-2.5-flash', 1)]
        assert 'fake-answer' in str(output)
        assert channel._is_image_model is channel.is_image_model
    finally:
        request_info.reset(token)
