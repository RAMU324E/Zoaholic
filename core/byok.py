"""BYOK（Bring Your Own Key）透传工具模块。

修改原因：BYOK 身份必须复用现有 api_keys/provider 配置，不能增加新的配置字段。
修改方式：把通配符 API Key 前缀匹配、BYOK provider 判断和请求级真实上游 key 暂存集中到本模块。
目的：让鉴权、请求转发、统计脱敏和模型列表动态拉取使用同一套规则，避免真实用户 key 进入日志或 key pool。
"""

from __future__ import annotations

import contextvars
import logging
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)

# 修改原因：标准 FastAPI 依赖可以写 request.state，但 handler/process_request 并不接收原始 Request。
# 修改方式：使用 ContextVar 在同一个请求上下文中临时传递 BYOK 真实上游 key 和模板身份。
# 目的：不改全部路由签名，也不把真实 key 写入 request_info 的公开统计字段。
_byok_real_key_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "byok_real_key",
    default=None,
)
_byok_template_key_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "byok_template_key",
    default=None,
)


def build_byok_prefixes(api_keys_config: list) -> list:
    """从 api_keys 配置构建 BYOK 前缀表。

    返回 [(prefix, api_index), ...]，按前缀长度降序排列，最长优先匹配。
    只有 api 字段以 * 结尾的条目才算 BYOK。
    """
    prefixes = []
    for idx, entry in enumerate(api_keys_config or []):
        if not isinstance(entry, dict):
            continue
        api = entry.get("api", "")
        if isinstance(api, str) and api.endswith("*") and len(api) > 1:
            prefix = api[:-1]
            prefixes.append((prefix, idx))
    prefixes.sort(key=lambda x: len(x[0]), reverse=True)
    return prefixes


def resolve_byok_token(
    token: Optional[str],
    byok_prefixes: list,
) -> Optional[Tuple[int, str, str]]:
    """尝试将 token 匹配到 BYOK 前缀。

    返回 (api_index, template_key, real_upstream_key) 或 None。
    real_upstream_key 为空时拒绝，避免把纯模板前缀当作可用上游 key。
    """
    if not isinstance(token, str) or not token:
        return None
    for prefix, api_index in byok_prefixes or []:
        if token.startswith(prefix):
            real_key = token[len(prefix):]
            # 修改原因：配置中的 byok-xxx-* 只是模板身份，客户端不能把结尾的 * 当成真实上游 key 使用。
            # 修改方式：除空字符串外，额外拒绝 real_key == "*" 的精确模板 token。
            # 目的：防止 BYOK 模板 key 被当作普通可用 API Key，绕过“必须自带上游 key”的约束。
            if not real_key or real_key == "*":
                return None
            template_key = prefix + "*"
            return (api_index, template_key, real_key)
    return None


def is_byok_provider(provider: dict) -> bool:
    """判断 provider 是否是 BYOK（api 列表显式包含 "*"）。"""
    api = (provider or {}).get("api")
    # 修改原因：空 api 与“管理员忘记填写 key”无法区分，不能再被自动判定为 BYOK。
    # 修改方式：只接受列表形式且包含字符串 "*" 的 provider.api 作为 BYOK 标记。
    # 目的：让 BYOK 模式必须由管理员显式配置，同时保留 api_keys 的通配符模板鉴权逻辑。
    return isinstance(api, list) and "*" in api


def is_byok_api_key(api_keys_config: list, api_index: int) -> bool:
    """判断 api_index 对应的条目是否是 BYOK 通配符条目。"""
    if api_index < 0 or api_index >= len(api_keys_config or []):
        return False
    entry = api_keys_config[api_index]
    if not isinstance(entry, dict):
        return False
    api = entry.get("api", "")
    return isinstance(api, str) and api.endswith("*")


def get_byok_prefixes(app: Any) -> list:
    """安全读取 app.state.byok_prefixes，缺失时返回空列表。"""
    try:
        return getattr(app.state, "byok_prefixes", None) or []
    except Exception:
        return []


def set_byok_context(real_key: Optional[str], template_key: Optional[str] = None) -> tuple:
    """设置请求级 BYOK 上下文，并返回可用于 reset 的 token 元组。"""
    real_token = _byok_real_key_var.set(real_key)
    template_token = _byok_template_key_var.set(template_key)
    return real_token, template_token


def reset_byok_context(tokens: Optional[tuple]) -> None:
    """恢复请求级 BYOK 上下文，避免同一执行上下文中残留真实 key。"""
    if not tokens:
        return
    real_token, template_token = tokens
    try:
        _byok_real_key_var.reset(real_token)
    except Exception:
        logger.debug("Failed to reset BYOK real key context", exc_info=True)
    try:
        _byok_template_key_var.reset(template_token)
    except Exception:
        logger.debug("Failed to reset BYOK template key context", exc_info=True)


def get_byok_real_key(default: Optional[str] = None) -> Optional[str]:
    """读取当前请求的真实上游 BYOK key。"""
    return _byok_real_key_var.get() or default


def get_byok_template_key(default: Optional[str] = None) -> Optional[str]:
    """读取当前请求的 BYOK 模板身份，如 byok-gemini-*。"""
    return _byok_template_key_var.get() or default


def store_byok_request_state(
    request: Any,
    *,
    byok_real_key: Optional[str],
    template_key: Optional[str],
    token_for_stats: str,
) -> None:
    """把鉴权结果写入 request.state 和 ContextVar。

    修改原因：不同入口会从 FastAPI Request 或 ASGI scope 读取状态，handler 又只能读请求上下文。
    修改方式：同时写 request.state 与 ContextVar；统计身份统一使用 token_for_stats。
    目的：让真实 key 只用于本次上游请求，日志和计费仍以模板 key 归集。
    """
    set_byok_context(byok_real_key, template_key)
    try:
        state = request.state
        state.byok_real_key = byok_real_key
        state.byok_template_key = template_key
        state.authenticated_token = token_for_stats
    except Exception:
        pass


def store_byok_scope_state(
    scope: dict,
    *,
    byok_real_key: Optional[str],
    template_key: Optional[str],
    token_for_stats: str,
) -> None:
    """把鉴权结果写入 ASGI scope.state 和 ContextVar。"""
    set_byok_context(byok_real_key, template_key)
    try:
        state = scope.setdefault("state", {})
        state["byok_real_key"] = byok_real_key
        state["byok_template_key"] = template_key
        state["authenticated_token"] = token_for_stats
    except Exception:
        pass


def update_request_info_auth(
    app: Any,
    api_index: Optional[int],
    token_for_stats: str,
    byok_real_key: Optional[str] = None,
    template_key: Optional[str] = None,
) -> None:
    """更新 request_info 中的鉴权展示字段，不写入真实 BYOK key。"""
    try:
        from core.middleware import request_info
        from utils import safe_get

        info = request_info.get()
        if not info:
            return
        config = getattr(app.state, "config", None) or {}
        info["api_key"] = token_for_stats
        info["api_key_name"] = safe_get(config, "api_keys", api_index, "name", default=None)
        info["api_key_group"] = safe_get(config, "api_keys", api_index, "group", default=None)
        # 修改原因：handler 需要知道当前请求是否为 BYOK，但公开统计字段不能出现真实 key。
        # 修改方式：只写内部下划线字段；数据库写入只读取白名单列，不会持久化这些运行时字段。
        # 目的：在保持统计脱敏的同时，把真实 key 限制在请求内存中使用。
        if byok_real_key:
            info["_byok_real_key"] = byok_real_key
            info["_byok_template_key"] = template_key or token_for_stats
        else:
            info.pop("_byok_real_key", None)
            info.pop("_byok_template_key", None)
    except Exception:
        pass
