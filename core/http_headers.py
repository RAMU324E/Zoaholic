"""HTTP header 合并与客户端 IP 提取。"""

from typing import Any, Optional


# 迁移说明：
# 修改原因：该模块承载业务逻辑，不应继续放在 utils_pkg 这种通用工具包中。
# 修改方式：按照 Scout 的归位方案迁移到 core 对应业务模块，并只调整必要的内部导入路径。
# 目的：让业务代码按领域归属维护，同时保留根 utils.py 和 utils_pkg shim 的旧导入兼容性。
def has_header_case_insensitive(headers: dict, key: str) -> bool:
    """大小写无关地检查请求头是否存在。"""
    if not isinstance(headers, dict):
        return False

    key_lower = str(key).lower()
    return any(str(existing_key).lower() == key_lower for existing_key in headers.keys())


def _set_header_case_insensitive(headers: Optional[dict], key: str, value) -> None:
    """大小写无关地写入请求头，避免 Content-Type/content-type 一类重复键。"""
    if not isinstance(headers, dict):
        return

    key_str = str(key)
    key_lower = key_str.lower()
    target_key = None

    for existing_key in headers.keys():
        if str(existing_key).lower() == key_lower:
            target_key = existing_key
            break

    normalized_value = ",".join(str(i) for i in value) if isinstance(value, list) else str(value)

    if target_key is None:
        headers[key_str] = normalized_value
    else:
        headers[target_key] = normalized_value


def apply_custom_headers(headers: dict, custom_headers: Optional[dict]) -> None:
    """将渠道自定义 headers 合并到请求头中。

    custom_headers 的值支持两种格式：
    - str: 直接设置
    - list[str]: 用逗号拼接后设置（符合 RFC 7230 §3.2.2）

    示例::
        {"anthropic-beta": ["val1", "val2"]}  →  "anthropic-beta": "val1,val2"
        {"X-Custom": "abc"}                   →  "X-Custom": "abc"
    """
    if not isinstance(custom_headers, dict):
        return
    for k, v in custom_headers.items():
        if v is None:
            continue
        # 值为 "null" 字符串时删除该 header（用于屏蔽渠道硬编码的头）
        if isinstance(v, str) and v.strip().lower() == "null":
            key_lower = str(k).lower()
            for existing_key in list(headers.keys()):
                if str(existing_key).lower() == key_lower:
                    del headers[existing_key]
                    break
            continue
        _set_header_case_insensitive(headers, k, v)


def extract_client_ip(headers: Any, fallback_host: str | None = None) -> str:
    """统一从请求头提取真实客户端 IP（CF-Connecting-IP > X-Forwarded-For 最右跳 > X-Real-IP > fallback）。"""
    if headers is not None:
        cf_ip = (headers.get("cf-connecting-ip") or "").strip()
        if cf_ip:
            return cf_ip
        forwarded_for = headers.get("x-forwarded-for")
        if forwarded_for:
            hops = [part.strip() for part in str(forwarded_for).split(",") if part.strip()]
            if hops:
                return hops[-1]
        real_ip = (headers.get("x-real-ip") or "").strip()
        if real_ip:
            return real_ip
    return fallback_host or "unknown"
