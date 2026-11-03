"""强制 WARP 代理出口插件（warp_only）

启用后，该渠道的所有出站请求（含模型请求、透传请求、OAuth 授权/刷新、
Vertex 服务账号换取 Token、余额查询、模型列表拉取）均强制走本地 Cloudflare WARP
SOCKS5 代理（或插件参数/渠道显式指定的代理），绝不暴露宿主机真实 IP。

断路保护（Fail-Closed）：
- 绝不允许回退到直连（proxy=None）；
- 若代理地址非法或本地 WARP SOCKS5 服务未就绪/不可达，底层连接将直接抛错阻断，
  不会发出一字节的直连流量。
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from fastapi import HTTPException

from core.plugins import register_request_interceptor, unregister_request_interceptor
from core.plugins.interceptors import parse_enabled_plugins
from core.utils import safe_get


PLUGIN_NAME = "warp_only"
_INTERCEPTOR_ID = "warp_only_request"

DEFAULT_WARP_PROXY = "socks5://172.18.1.1:40000"
_ALLOWED_PROXY_SCHEMES = frozenset({"socks5", "socks5h", "http", "https"})

PLUGIN_INFO = {
    "name": PLUGIN_NAME,
    "version": "1.0.0",
    "description": "强制走 Cloudflare WARP 代理出口 - 防漏真实 IP，代理异常时直接阻断绝不回退直连",
    "author": "Zoaholic Team",
    "dependencies": [],
    "metadata": {
        "category": "interceptors",
        "tags": ["proxy", "warp", "privacy", "security"],
        "params_hint": (
            "启用后该渠道所有出站流量（模型请求、透传、Vertex/OAuth 换 Token、余额查询）强制走 WARP 代理，绝不走宿主机真实 IP。\n"
            "参数格式：\n"
            "  (空)                        = 默认使用本地 WARP (socks5://172.18.1.1:40000) 或渠道已填 proxy\n"
            "  socks5://127.0.0.1:40000    = 覆盖指定代理地址"
        ),
    },
}

EXTENSIONS = [f"interceptors:{_INTERCEPTOR_ID}"]


def is_warp_only_enabled(provider: Optional[Dict[str, Any]]) -> bool:
    """判断 provider 是否启用了 warp_only 插件。"""
    if not isinstance(provider, dict):
        return False
    enabled_plugins = safe_get(provider, "preferences", "enabled_plugins", default=None)
    if not enabled_plugins:
        return False
    parsed = parse_enabled_plugins(enabled_plugins)
    return PLUGIN_NAME in parsed


def validate_warp_proxy_url(proxy_url: Any) -> str:
    """校验并返回合法的代理 URL；不合法则立即抛错（Fail-Closed）。"""
    text = str(proxy_url or "").strip()
    if not text:
        raise ValueError("warp_only requires a non-empty proxy URL (direct connection is forbidden)")
    parsed = urlparse(text)
    scheme = (parsed.scheme or "").lower()
    if scheme not in _ALLOWED_PROXY_SCHEMES or not parsed.hostname:
        raise ValueError(
            f"warp_only rejected invalid proxy URL {text!r}; "
            f"allowed schemes: {', '.join(sorted(_ALLOWED_PROXY_SCHEMES))}"
        )
    return text


def resolve_warp_only_proxy(provider: Optional[Dict[str, Any]]) -> Optional[str]:
    """若渠道启用了 warp_only，返回强制代理 URL；未启用则返回 None。

    优先级：
    1. 插件参数（如 warp_only:socks5://127.0.0.1:40000 或 {proxy: ...}）
    2. 渠道显式配置的 preferences.proxy / provider.proxy
    3. 环境变量 ZOAHOLIC_WARP_PROXY 或默认 socks5://172.18.1.1:40000
    """
    if not is_warp_only_enabled(provider):
        return None

    enabled_plugins = safe_get(provider, "preferences", "enabled_plugins", default=None)
    options = parse_enabled_plugins(enabled_plugins).get(PLUGIN_NAME)

    candidate: Any = None
    if isinstance(options, str) and options.strip():
        candidate = options.strip()
    elif isinstance(options, dict):
        candidate = options.get("proxy") or options.get("url")

    if not candidate and isinstance(provider, dict):
        candidate = safe_get(provider, "preferences", "proxy", default=None) or provider.get("proxy")

    if not candidate:
        candidate = os.environ.get("ZOAHOLIC_WARP_PROXY") or DEFAULT_WARP_PROXY

    return validate_warp_proxy_url(candidate)


async def warp_only_request_interceptor(
    request: Any,
    engine: str,
    provider: Dict[str, Any],
    api_key: Optional[str],
    url: str,
    headers: Dict[str, Any],
    payload: Dict[str, Any],
):
    """出站请求拦截器：二次确认 warp_only 代理配置合法且当前上下文未退回直连。"""
    if not is_warp_only_enabled(provider):
        return url, headers, payload

    try:
        proxy_url = resolve_warp_only_proxy(provider)
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if not proxy_url:
        raise HTTPException(
            status_code=502,
            detail="warp_only blocked request: no proxy configured",
        )

    return url, headers, payload


def setup(manager=None):
    register_request_interceptor(
        _INTERCEPTOR_ID,
        warp_only_request_interceptor,
        priority=10,
        plugin_name=PLUGIN_NAME,
        metadata={"description": "强制走 WARP 代理出口并阻断直连回退"},
    )


def teardown(manager=None):
    unregister_request_interceptor(_INTERCEPTOR_ID)
