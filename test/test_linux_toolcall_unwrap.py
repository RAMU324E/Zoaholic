"""linux_toolcall 1.2.1 修复回归测试。

覆盖四类场景：
1. 流式 + 破限工具 -> 参数流被解包成正文，finish_reason 改为 stop；
2. 流式 + 真实工具（save_memory 等）-> 原样透传，finish_reason 保持 tool_calls；
3. 非流式 + 破限工具 -> 解包成 message.content；
4. 非流式 + 真实工具 -> 原样透传，tool_calls 保留。
"""

from __future__ import annotations

import asyncio
import contextvars
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

PLUGIN_PATH = Path(__file__).resolve().parents[1] / "plugins" / "linux_toolcall.py"

TOOL = "dispatch_stream_posix_vfs"


def _install_stub_modules() -> None:
    if "core" not in sys.modules:
        core_pkg = types.ModuleType("core")
        core_pkg.__path__ = []  # type: ignore[attr-defined]
        sys.modules["core"] = core_pkg

    if "core.log_config" not in sys.modules:
        log_config = types.ModuleType("core.log_config")

        class _Logger:
            def _noop(self, *args: Any, **kwargs: Any) -> None:
                return None

            info = debug = warning = error = exception = _noop

        log_config.logger = _Logger()
        sys.modules["core.log_config"] = log_config

    if "core.middleware" not in sys.modules:
        middleware = types.ModuleType("core.middleware")
        middleware.request_info = contextvars.ContextVar("request_info", default=None)
        sys.modules["core.middleware"] = middleware

    if "core.plugins" not in sys.modules:
        plugins = types.ModuleType("core.plugins")
        plugins.register_request_interceptor = lambda **kwargs: None
        plugins.unregister_request_interceptor = lambda *args, **kwargs: None
        plugins.register_response_interceptor = lambda **kwargs: None
        plugins.unregister_response_interceptor = lambda *args, **kwargs: None
        sys.modules["core.plugins"] = plugins


def _load_plugin() -> Any:
    _install_stub_modules()
    spec = importlib.util.spec_from_file_location("linux_toolcall_under_test", PLUGIN_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plugin = _load_plugin()


def _sse(payload: dict[str, Any] | str) -> str:
    body = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return f"data: {body}\n\n"


def _linux_name_chunk() -> str:
    return _sse({
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "model": "gemini-test",
        "choices": [{
            "index": 0,
            "delta": {"tool_calls": [{
                "index": 0,
                "id": "call_linux",
                "type": "function",
                "function": {"name": TOOL, "arguments": ""},
            }]},
            "finish_reason": None,
        }],
    })


def _args_chunk(arguments: str, *, call_id: str = "call_linux", name: str | None = None) -> str:
    fn: dict[str, Any] = {"arguments": arguments}
    if name:
        fn["name"] = name
    call: dict[str, Any] = {"index": 0, "function": fn}
    if name:
        call["id"] = call_id
        call["type"] = "function"
    return _sse({
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "model": "gemini-test",
        "choices": [{"index": 0, "delta": {"tool_calls": [call]}, "finish_reason": None}],
    })


def _finish_chunk(reason: str = "tool_calls") -> str:
    return _sse({
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "model": "gemini-test",
        "choices": [{"index": 0, "delta": {}, "finish_reason": reason}],
    })


def _run_stream(module: Any, chunks: list[str], *, tool_name: str = TOOL) -> list[dict[str, Any]]:
    async def _main() -> list[dict[str, Any]]:
        decoder = module.StreamStringArgDecoder("display_stream")
        info = {
            "_linux_toolcall_active": True,
            "_linux_toolcall_name": tool_name,
            "_linux_toolcall_param": "display_stream",
            "_linux_toolcall_decoder": decoder,
        }
        module._ACTIVE_STREAM_DECODERS[tool_name] = decoder
        token = module.request_info.set(info)
        outputs: list[dict[str, Any]] = []
        try:
            for chunk in chunks:
                out = await module.linux_toolcall_response_interceptor(
                    chunk, "openai", "gemini-test", True,
                )
                for line in str(out).split("\n"):
                    if not line.startswith("data: "):
                        continue
                    data = line[6:].strip()
                    if data == "[DONE]":
                        continue
                    outputs.append(json.loads(data))
        finally:
            module.request_info.reset(token)
            module._ACTIVE_STREAM_DECODERS.pop(tool_name, None)
        return outputs

    return asyncio.run(_main())


def _joined_content(chunks: list[dict[str, Any]]) -> str:
    text = ""
    for chunk in chunks:
        for choice in chunk.get("choices", []):
            text += (choice.get("delta") or {}).get("content") or ""
    return text


def test_stream_linux_tool_is_unwrapped_to_content() -> None:
    chunks = _run_stream(plugin, [
        _linux_name_chunk(),
        _args_chunk('{"display_stream": "你好'),
        _args_chunk('，世界"}'),
        _finish_chunk("tool_calls"),
        _sse("[DONE]"),
    ])
    assert _joined_content(chunks) == "你好，世界"
    assert all(
        not (choice.get("delta") or {}).get("tool_calls")
        for chunk in chunks
        for choice in chunk.get("choices", [])
    )
    reasons = [
        choice.get("finish_reason")
        for chunk in chunks
        for choice in chunk.get("choices", [])
        if choice.get("finish_reason")
    ]
    assert reasons == ["stop"]


def test_stream_foreign_tool_is_passed_through_untouched() -> None:
    raw = [
        _args_chunk('{"content": "用', call_id="call_mem", name="save_memory"),
        _args_chunk('户偏好"}'),
        _finish_chunk("tool_calls"),
        _sse("[DONE]"),
    ]
    chunks = _run_stream(plugin, raw)
    assert _joined_content(chunks) == ""

    calls = [
        tc
        for chunk in chunks
        for choice in chunk.get("choices", [])
        for tc in ((choice.get("delta") or {}).get("tool_calls") or [])
    ]
    assert calls, "真实工具调用必须保留 tool_calls"
    names = {tc.get("function", {}).get("name") for tc in calls if tc.get("function", {}).get("name")}
    assert names == {"save_memory"}
    args = "".join(tc.get("function", {}).get("arguments") or "" for tc in calls)
    assert args == '{"content": "用户偏好"}'
    reasons = [
        choice.get("finish_reason")
        for chunk in chunks
        for choice in chunk.get("choices", [])
        if choice.get("finish_reason")
    ]
    assert reasons == ["tool_calls"], "真实工具的 finish_reason 不能被改成 stop"


def test_stream_mixed_foreign_call_is_not_swallowed() -> None:
    raw = [
        _linux_name_chunk(),
        _args_chunk('{"display_stream": "正文'),
        _args_chunk('{"path": "a.txt"}', call_id="call_read", name="read_file"),
        _finish_chunk("tool_calls"),
    ]
    chunks = _run_stream(plugin, raw)
    calls = [
        tc
        for chunk in chunks
        for choice in chunk.get("choices", [])
        for tc in ((choice.get("delta") or {}).get("tool_calls") or [])
    ]
    assert any(tc.get("function", {}).get("name") == "read_file" for tc in calls)


def test_non_stream_linux_tool_is_unwrapped() -> None:
    payload = {
        "id": "chatcmpl-2",
        "object": "chat.completion",
        "model": "gemini-test",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_linux",
                    "type": "function",
                    "function": {
                        "name": TOOL,
                        "arguments": json.dumps({"display_stream": "非流式正文"}, ensure_ascii=False),
                    },
                }],
            },
            "finish_reason": "tool_calls",
        }],
    }

    async def _main() -> Any:
        decoder = plugin.StreamStringArgDecoder("display_stream")
        info = {
            "_linux_toolcall_active": True,
            "_linux_toolcall_name": TOOL,
            "_linux_toolcall_param": "display_stream",
            "_linux_toolcall_decoder": decoder,
        }
        plugin._ACTIVE_STREAM_DECODERS[TOOL] = decoder
        token = plugin.request_info.set(info)
        try:
            return await plugin.linux_toolcall_response_interceptor(
                json.dumps(payload, ensure_ascii=False), "openai", "gemini-test", False,
            )
        finally:
            plugin.request_info.reset(token)
            plugin._ACTIVE_STREAM_DECODERS.pop(TOOL, None)

    out = json.loads(asyncio.run(_main()))
    choice = out["choices"][0]
    assert choice["message"]["content"] == "非流式正文"
    assert "tool_calls" not in choice["message"]
    assert choice["finish_reason"] == "stop"
    assert out["id"] == "chatcmpl-2"


def test_non_stream_foreign_tool_is_passed_through_untouched() -> None:
    payload = {
        "id": "chatcmpl-3",
        "object": "chat.completion",
        "model": "gemini-test",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_mem",
                    "type": "function",
                    "function": {
                        "name": "save_memory",
                        "arguments": json.dumps({"content": "记住这件事"}, ensure_ascii=False),
                    },
                }],
            },
            "finish_reason": "tool_calls",
        }],
    }
    original = json.dumps(payload, ensure_ascii=False)

    async def _main() -> Any:
        decoder = plugin.StreamStringArgDecoder("display_stream")
        info = {
            "_linux_toolcall_active": True,
            "_linux_toolcall_name": TOOL,
            "_linux_toolcall_param": "display_stream",
            "_linux_toolcall_decoder": decoder,
        }
        plugin._ACTIVE_STREAM_DECODERS[TOOL] = decoder
        token = plugin.request_info.set(info)
        try:
            return await plugin.linux_toolcall_response_interceptor(
                original, "openai", "gemini-test", False,
            )
        finally:
            plugin.request_info.reset(token)
            plugin._ACTIVE_STREAM_DECODERS.pop(TOOL, None)

    out = json.loads(asyncio.run(_main()))
    choice = out["choices"][0]
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "save_memory"
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None


def test_gemini_native_payload_request_interceptor_rewrites_contents_and_tools() -> None:
    provider = {
        "provider": "moefb",
        "engine": "gemini",
        "preferences": {
            "enabled_plugins": ["linux_toolcall(10.18):tool_name=asuka_vfs_stream"],
        },
    }
    gemini_payload = {
        "systemInstruction": {"parts": [{"text": "你是明日香。"}]},
        "contents": [
            {"role": "user", "parts": [{"text": "先看下文件"}]},
            {"role": "model", "parts": [{"functionCall": {"name": "read_file", "args": {"path": "a.txt"}}}]},
            {"role": "user", "parts": [{"functionResponse": {"name": "read_file", "response": {"result": "hello"}}}]},
            {"role": "user", "parts": [{"text": "抱抱我"}]},
        ],
        "tools": [{
            "function_declarations": [
                {"name": "save_memory", "description": "save memory", "parameters": {"type": "object"}},
            ]
        }],
        "tool_config": {"function_calling_config": {"mode": "AUTO"}},
    }

    async def _main() -> dict[str, Any]:
        info: dict[str, Any] = {}
        token = plugin.request_info.set(info)
        try:
            _u, _h, out_payload = await plugin.linux_toolcall_request_interceptor(
                None,
                "gemini",
                provider,
                "key",
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-4-argon:streamGenerateContent?alt=sse",
                {},
                gemini_payload,
            )
            assert info.get("_linux_toolcall_active") is True
            assert info.get("_linux_toolcall_name") == "asuka_vfs_stream"
            return out_payload
        finally:
            plugin.request_info.reset(token)
            plugin._ACTIVE_STREAM_DECODERS.pop("asuka_vfs_stream", None)

    out = asyncio.run(_main())
    assert "messages" not in out
    assert len(out["contents"]) == 9
    assert out["contents"][0]["role"] == "user"
    assert out["contents"][1]["role"] == "model"
    assert out["contents"][2]["role"] == "user"
    assert "asuka_vfs_stream" in out["systemInstruction"]["parts"][0]["text"]

    fn_decls = out["tools"][0]["function_declarations"]
    names = [d["name"] for d in fn_decls]
    assert names == ["save_memory", "asuka_vfs_stream"]
    desc = fn_decls[1]["description"]
    assert "你是明日香。" in desc
    contents_text = "\n".join(
        p.get("text", "")
        for c in out["contents"]
        for p in c.get("parts", [])
    )
    assert "read_file" in contents_text
    assert "抱抱我" in contents_text


def test_stream_embedded_pseudo_tool_call_inside_display_stream_is_rescued() -> None:
    leaked_stream = (
        '[IMAGE_EVAL]\nLooks good.\n[/IMAGE_EVAL]\n'
        '[tool_call: read_file({"files": [{"path": "data/temp/test.png"}]})]'
    )
    chunks = _run_stream(plugin, [
        _linux_name_chunk(),
        _args_chunk(json.dumps({"display_stream": leaked_stream}, ensure_ascii=False)),
        _finish_chunk("tool_calls"),
        _sse("[DONE]"),
    ])
    calls = [
        tc
        for chunk in chunks
        for choice in chunk.get("choices", [])
        for tc in ((choice.get("delta") or {}).get("tool_calls") or [])
    ]
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "read_file"
    assert json.loads(calls[0]["function"]["arguments"]) == {"files": [{"path": "data/temp/test.png"}]}
    reasons = [
        choice.get("finish_reason")
        for chunk in chunks
        for choice in chunk.get("choices", [])
        if choice.get("finish_reason")
    ]
    assert reasons[-1] == "tool_calls"


def test_ephemeral_dynamic_context_is_moved_to_tail_and_gemini_key_order_is_cache_friendly() -> None:
    plugin_src = Path(plugin.__file__).read_text(encoding="utf-8")
    assert "asuka_vfs_stream" not in plugin_src

    turn1_payload = {
        "contents": [
            {"role": "user", "parts": [{"text": "以下是本轮动态上下文信息，每轮可能变化。\n当前时间：15:00"}]},
            {"role": "user", "parts": [{"text": "第一轮真实用户消息"}]},
        ],
        "systemInstruction": {"parts": [{"text": "系统设定固定不变"}]},
        "tools": [{"function_declarations": [{"name": "read_file", "description": "Read", "parameters": {"type": "object"}}]}],
        "generationConfig": {"temperature": 1.0},
    }
    turn2_payload = {
        "contents": [
            {"role": "user", "parts": [{"text": "第一轮真实用户消息"}]},
            {"role": "model", "parts": [{"text": "第一轮助手回复"}]},
            {"role": "user", "parts": [{"text": "以下是本轮动态上下文信息，每轮可能变化。\n当前时间：15:05"}]},
            {"role": "user", "parts": [{"text": "第二轮真实用户消息"}]},
        ],
        "systemInstruction": {"parts": [{"text": "系统设定固定不变"}]},
        "tools": [{"function_declarations": [{"name": "read_file", "description": "Read", "parameters": {"type": "object"}}]}],
        "generationConfig": {"temperature": 1.0},
    }

    _, _, out1 = asyncio.run(
        plugin.linux_toolcall_request_interceptor(
            request=None,
            engine="gemini",
            provider={"preferences": {"enabled_plugins": ["linux_toolcall"]}},
            api_key=None,
            url="https://generativelanguage.googleapis.com/v1beta/models/gemini-4-argon:streamGenerateContent",
            headers={},
            payload=turn1_payload,
        )
    )
    _, _, out2 = asyncio.run(
        plugin.linux_toolcall_request_interceptor(
            request=None,
            engine="gemini",
            provider={"preferences": {"enabled_plugins": ["linux_toolcall"]}},
            api_key=None,
            url="https://generativelanguage.googleapis.com/v1beta/models/gemini-4-argon:streamGenerateContent",
            headers={},
            payload=turn2_payload,
        )
    )

    # 1. Gemini dict key order: systemInstruction -> tools -> tool_config -> generationConfig -> contents
    assert list(out1.keys()) == ["systemInstruction", "tools", "tool_config", "generationConfig", "contents"]

    # 2. Tool description is 100% static (byte-for-byte identical) between turn 1 and turn 2
    desc1 = out1["tools"][0]["function_declarations"][-1]["description"]
    desc2 = out2["tools"][0]["function_declarations"][-1]["description"]
    assert desc1 == desc2
    assert "以下是本轮动态上下文信息" not in desc1

    # 3. Historical message objects in contents before the final user message are 100% identical (object-boundary cache friendly)
    # Turn 1 contents: [router, kernel, user1, ack, final_directive1]
    # Turn 2 contents: [router, kernel, user1, model1, user2, ack, final_directive2]
    assert out1["contents"][:3] == out2["contents"][:3]

    # 4. Ephemeral dynamic context is placed in the final user message before tail obfuscation
    last_text2 = out2["contents"][-1]["parts"][0]["text"]
    assert "<runtime_ephemeral_env>" in last_text2
    assert "当前时间：15:05" in last_text2
    assert last_text2.index("<runtime_ephemeral_env>") < last_text2.index("<posix_kernel_tail_verification>")



