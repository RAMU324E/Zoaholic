import asyncio
import sys
from pathlib import Path
from types import ModuleType
import httpx
import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import httpx_socks  # noqa: F401
except ModuleNotFoundError:
    stub = ModuleType("httpx_socks")

    class _DummyAsyncProxyTransport(httpx.AsyncBaseTransport):
        def __init__(self, url: str):
            self.url = url

        @classmethod
        def from_url(cls, url: str, **kwargs):
            return cls(url)

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, request=request)

    stub.AsyncProxyTransport = _DummyAsyncProxyTransport  # type: ignore[attr-defined]
    sys.modules["httpx_socks"] = stub

from core.http import install as install_http_patch, proxy_context
from core.oauth.manager import OAuthManager
from core.utils import is_warp_only_provider, resolve_provider_proxy
from plugins.warp_only import (
    DEFAULT_WARP_PROXY,
    resolve_warp_only_proxy,
    warp_only_request_interceptor,
)


install_http_patch()


def test_resolve_warp_only_default_and_override():
    p_default = {"provider": "vtx", "preferences": {"enabled_plugins": ["warp_only"]}}
    assert is_warp_only_provider(p_default) is True
    assert resolve_warp_only_proxy(p_default) == DEFAULT_WARP_PROXY
    assert resolve_provider_proxy(p_default, {}) == DEFAULT_WARP_PROXY

    p_custom = {
        "provider": "vtx",
        "preferences": {"enabled_plugins": ["warp_only:socks5://127.0.0.1:40000"]},
    }
    assert resolve_provider_proxy(p_custom, {}) == "socks5://127.0.0.1:40000"

    p_pref = {
        "provider": "vtx",
        "preferences": {
            "proxy": "http://127.0.0.1:8080",
            "enabled_plugins": ["warp_only"],
        },
    }
    assert resolve_provider_proxy(p_pref, {}) == "http://127.0.0.1:8080"


def test_warp_only_rejects_invalid_proxy_url():
    p_bad = {
        "provider": "vtx",
        "preferences": {"enabled_plugins": ["warp_only:not-a-valid-proxy"]},
    }
    with pytest.raises(ValueError):
        resolve_provider_proxy(p_bad, {})

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            warp_only_request_interceptor(
                None, "vertex-gemini", p_bad, "k", "https://example.com", {}, {}
            )
        )
    assert exc_info.value.status_code == 502


def test_proxy_context_injects_into_explicit_none_and_blocks_direct_send():
    # Vertex get_access_token passes proxy=None when env vars are unset;
    # proxy_context must still inject the WARP transport.
    with proxy_context("socks5://172.18.1.1:40000", require_proxy=True):
        client = httpx.AsyncClient(proxy=None)
        assert getattr(client, "_zoaholic_proxied", False) is True
        asyncio.run(client.aclose())

    # An unproxied client created outside proxy_context must be blocked from sending inside require_proxy=True
    unproxied = httpx.AsyncClient()
    assert getattr(unproxied, "_zoaholic_proxied", False) is False
    try:
        with proxy_context("socks5://172.18.1.1:40000", require_proxy=True):
            with pytest.raises(httpx.ProxyError, match="warp_only blocked a direct connection attempt"):
                asyncio.run(unproxied.get("https://example.com"))
    finally:
        asyncio.run(unproxied.aclose())


def test_oauth_manager_channel_proxy_context_enforces_warp_only(tmp_path, monkeypatch):
    cfg = {
        "providers": [
            {
                "provider": "vertex-main",
                "engine": "vertex",
                "preferences": {"enabled_plugins": ["warp_only"]},
            }
        ]
    }
    monkeypatch.setattr(OAuthManager, "_install_channel_oauth_registrar", lambda self: None)
    mgr = OAuthManager(state_path=str(tmp_path / "oauth.json"))
    mgr._config_ref = lambda: cfg
    with mgr.channel_proxy_context("vertex-main"):
        client = httpx.AsyncClient(proxy=None)
        assert getattr(client, "_zoaholic_proxied", False) is True
        asyncio.run(client.aclose())
