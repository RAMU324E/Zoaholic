"""Gemini model-turn prefill plugin (gemini_model_prefill).

Appends one ``role: model`` turn to the end of a Gemini-native ``contents``
array, so clients that cannot send a trailing assistant message (e.g. Clonoth)
still get a prefill. Enable it only on channels whose upstream accepts a
model-ending request (AI Studio style; set includeThoughts=true there).

Only user-message rounds are prefilled. Skipped when the last turn is already
a model turn (client prefill wins) or when any turn after the last model turn
carries a functionResponse: tool rounds may append extra user turns (tool
images, per-round context) after the tool result and must stay free to call tools.
The upstream returns only the continuation; the prefill text is not echoed back.
"""

from __future__ import annotations

from typing import Any, Dict

from core.log_config import logger
from core.plugins import register_request_interceptor, unregister_request_interceptor
from core.plugins.interceptors import get_plugin_options

PLUGIN_NAME = "gemini_model_prefill"
_INTERCEPTOR_ID = "gemini_model_prefill_request"
DEFAULT_PREFILL = "好的，master，我这就开始"

PARAMS_SCHEMA = [
    {
        "key": "text",
        "label": "预填充文本",
        "type": "textarea",
        "placeholder": DEFAULT_PREFILL,
        "default": DEFAULT_PREFILL,
        "hint": "追加到请求末尾的 model 消息内容；留空使用默认文本",
        "serialize": "key_value",
    },
]

PLUGIN_INFO = {
    "name": PLUGIN_NAME,
    "version": "1.1.0",
    "description": "Gemini 末尾 model 预填充 — 在 contents 末尾追加一条 role:model 消息（跳过工具结果轮与已有 model 结尾）",
    "author": "Zoaholic Team",
    "dependencies": [],
    "metadata": {
        "category": "interceptors",
        "tags": ["gemini", "prefill"],
        "params_hint": "text=预填充文本（默认：好的，master，我这就开始）",
        "params_schema": PARAMS_SCHEMA,
    },
}

EXTENSIONS = [f"interceptors:{_INTERCEPTOR_ID}"]


def resolve_prefill_text(options: Any) -> str:
    """Read the prefill text from plugin options (dict, "text=..." or bare string)."""
    value: Any = None
    if isinstance(options, dict):
        value = options.get("text")
    elif isinstance(options, str):
        raw = options.strip()
        # Take everything after "text=" verbatim so commas in the text survive.
        value = raw[len("text="):] if raw.startswith("text=") else raw
    text = str(value).strip() if value is not None else ""
    return text or DEFAULT_PREFILL


def apply_model_prefill(payload: Dict[str, Any], text: str) -> bool:
    """Append a model turn to payload["contents"]; return True when added."""
    contents = payload.get("contents") if isinstance(payload, dict) else None
    if not isinstance(contents, list) or not contents or not isinstance(contents[-1], dict):
        return False
    if contents[-1].get("role") == "model":
        return False
    for turn in reversed(contents):
        if not isinstance(turn, dict) or turn.get("role") == "model":
            break
        parts = turn.get("parts") if isinstance(turn.get("parts"), list) else []
        if any(isinstance(p, dict) and ("functionResponse" in p or "function_response" in p) for p in parts):
            return False
    contents.append({"role": "model", "parts": [{"text": text}]})
    return True


async def gemini_model_prefill_request_interceptor(request, engine, provider, api_key, url, headers, payload):
    try:
        text = resolve_prefill_text(get_plugin_options(PLUGIN_NAME, provider))
        if apply_model_prefill(payload, text):
            logger.debug(f"[{PLUGIN_NAME}] appended model prefill ({len(text)} chars)")
    except Exception as exc:  # fail-open: never break the request
        logger.warning(f"[{PLUGIN_NAME}] skipped: {exc}")
    return url, headers, payload


def setup(manager):
    register_request_interceptor(
        interceptor_id=_INTERCEPTOR_ID,
        callback=gemini_model_prefill_request_interceptor,
        # After format-shaping plugins (linux_toolcall=60), so the model turn stays last.
        priority=90,
        plugin_name=PLUGIN_NAME,
        metadata={
            "description": PLUGIN_INFO["description"],
            "params_hint": PLUGIN_INFO["metadata"]["params_hint"],
            "params_schema": PARAMS_SCHEMA,
        },
        overwrite=True,
    )


def teardown(manager=None):
    unregister_request_interceptor(_INTERCEPTOR_ID)


def unload():
    teardown()
