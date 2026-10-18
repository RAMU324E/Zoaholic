import asyncio
import sys
from contextvars import ContextVar
from types import ModuleType, SimpleNamespace

if "core.middleware" not in sys.modules:
    _mw = ModuleType("core.middleware")
    _mw.request_info = ContextVar("request_info", default={})
    sys.modules["core.middleware"] = _mw

from plugins.factory_compat import (
    DEFAULT_BUFFER_PROMPT,
    DEFAULT_CLI_VERSION,
    DEFAULT_COMPLIANCE_PROMPT,
    factory_compat_post_request_interceptor,
    factory_compat_request_interceptor,
)
from plugins.linux_toolcall import linux_toolcall_request_interceptor


def test_gemini_engine_gemini_4_argon_native_factory_route():
    async def _run():
        req = SimpleNamespace(model="fac_gem/gemini-4-argon", tool_choice=None)
        provider = {
            "provider": "factory_gem",
            "engine": "gemini",
            "base_url": "https://api.factory.ai/v1beta",
            "model": [{"fac_gem/gemini-4-argon": "gemini-4-argon"}],
            "preferences": {
                "enabled_plugins": ["factory_compat", "header_scrubber"],
            },
        }
        url = "https://api.factory.ai/v1beta/models/gemini-4-argon:streamGenerateContent?alt=sse"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": "fk-test-secret",
        }
        payload = {
            "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
            "systemInstruction": {"parts": [{"text": "You are a helpful writer."}]},
            "tools": [{"function_declarations": [{"name": "read_file", "description": "Read a file"}]}],
            "generationConfig": {
                "temperature": 0.9,
                "topP": 0.95,
                "topK": 64,
                "maxOutputTokens": 32768,
            },
        }

        url, headers, payload = await factory_compat_request_interceptor(
            req, "gemini", provider, "fk-test-secret", url, headers, payload
        )
        url, headers, payload = await factory_compat_post_request_interceptor(
            req, "gemini", provider, "fk-test-secret", url, headers, payload
        )

        assert url == "https://api.factory.ai/api/llm/g/v1/generate"
        assert headers.get("Authorization") == "Bearer fk-test-secret"
        assert "x-goog-api-key" not in headers
        assert headers.get("x-api-provider") == "google"
        assert headers.get("User-Agent") == f"factory-cli/{DEFAULT_CLI_VERSION}"
        assert headers.get("X-Factory-Client") == "cli"
        assert headers.get("X-Client-Version") == DEFAULT_CLI_VERSION
        assert headers.get("x-session-id")
        assert headers.get("x-assistant-message-id")
        assert headers.get("x-provider-routing-source") == "registry_default"
        assert headers.get("x-message-call-type") == "main turn"

        assert list(payload.keys()) == [
            "model",
            "contents",
            "systemInstruction",
            "tools",
            "generationConfig",
        ]
        assert payload["model"] == "gemini-4-argon"

        # omitSamplingParameters=True for gemini-4-argon
        gen_cfg = payload["generationConfig"]
        assert "temperature" not in gen_cfg
        assert "topP" not in gen_cfg
        assert "topK" not in gen_cfg
        assert gen_cfg["maxOutputTokens"] == 65536
        assert gen_cfg["thinkingConfig"] == {"includeThoughts": True, "thinkingLevel": "MEDIUM"}

        # tools normalized to camelCase functionDeclarations
        assert "functionDeclarations" in payload["tools"][0]
        assert "function_declarations" not in payload["tools"][0]

        # compliance prompt injected into systemInstruction
        sys_text = payload["systemInstruction"]["parts"][0]["text"]
        assert sys_text.startswith(DEFAULT_COMPLIANCE_PROMPT)
        assert DEFAULT_BUFFER_PROMPT in sys_text
        assert "You are a helpful writer." in sys_text

    asyncio.run(_run())


def test_gemini_engine_chained_with_linux_toolcall_preserves_compliance_and_schema():
    async def _run():
        req = SimpleNamespace(model="gemini-4-argon-thinking-high", tool_choice=None, headers={})
        provider = {
            "provider": "factory_gem",
            "engine": "gemini",
            "base_url": "https://api.factory.ai/api/llm/g/v1/generate",
            "preferences": {
                "enabled_plugins": [
                    "factory_compat",
                    "linux_toolcall:tool_name=asuka_vfs_stream,panic=true",
                ],
            },
        }
        url = "https://api.factory.ai/api/llm/g/v1/generate/models/gemini-4-argon-thinking-high:streamGenerateContent?alt=sse"
        headers = {"Content-Type": "application/json", "x-goog-api-key": "fk-key"}
        payload = {
            "systemInstruction": {"parts": [{"text": "你是明日香。"}]},
            "contents": [{"role": "user", "parts": [{"text": "你好"}]}],
            "tools": [{"function_declarations": [{"name": "save_memory", "description": "Save"}]}],
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 32768},
        }

        # Priority 30 -> 60 -> 65
        url, headers, payload = await factory_compat_request_interceptor(
            req, "gemini", provider, "fk-key", url, headers, payload
        )
        url, headers, payload = await linux_toolcall_request_interceptor(
            req, "gemini", provider, "fk-key", url, headers, payload
        )
        url, headers, payload = await factory_compat_post_request_interceptor(
            req, "gemini", provider, "fk-key", url, headers, payload
        )

        assert url == "https://api.factory.ai/api/llm/g/v1/generate"
        assert list(payload.keys()) == [
            "model",
            "contents",
            "systemInstruction",
            "tools",
            "generationConfig",
        ]
        assert "tool_config" not in payload
        assert "safetySettings" not in payload
        assert payload["model"] == "gemini-4-argon"
        assert payload["generationConfig"]["thinkingConfig"] == {
            "includeThoughts": True,
            "thinkingLevel": "HIGH",
        }
        assert "temperature" not in payload["generationConfig"]

        sys_text = payload["systemInstruction"]["parts"][0]["text"]
        assert sys_text.startswith(DEFAULT_COMPLIANCE_PROMPT)
        assert DEFAULT_BUFFER_PROMPT in sys_text
        assert "你是明日香。" in sys_text

        decls = payload["tools"][0]["functionDeclarations"]
        assert [d["name"] for d in decls] == ["save_memory", "asuka_vfs_stream"]

    asyncio.run(_run())


def test_claude_and_openai_fast_models_add_and_aliases_preserved():
    async def _run():
        # 1. Claude with fast_models_add and claude-fable-5-1 canonical alias
        provider_claude = {
            "provider": "factory_A",
            "engine": "claude",
            "preferences": {
                "factory_compat": {
                    "fast_models_add": ["claude-opus-4-8", "claude-opus-5"],
                }
            },
        }
        headers_c = {"x-api-key": "fk-claude"}
        payload_c = {
            "model": "factoryclaude/claude-opus-5",
            "max_tokens": 4096,
            "temperature": 0.8,
            "thinking": {"type": "enabled", "budget_tokens": 8192},
            "system": "Hello",
            "messages": [{"role": "user", "content": "hi"}],
        }
        _, headers_c, payload_c = await factory_compat_request_interceptor(
            None, "claude", provider_claude, "fk-claude", "https://api.factory.ai/api/llm/a/v1/messages", headers_c, payload_c
        )
        assert payload_c["model"] == "claude-opus-5"
        assert payload_c["speed"] == "fast"
        assert "temperature" not in payload_c
        assert payload_c["max_tokens"] == 12288  # bumped > budget_tokens (8192 + 4096)
        assert "fast-mode-2026-02-01" in headers_c["anthropic-beta"]
        assert headers_c["Authorization"] == "Bearer fk-claude"
        assert headers_c["x-api-provider"] == "anthropic"

        # Canonical alias claude-fable-5-1-ver -> claude-fable-5.1 @ vertex_anthropic
        headers_f = {"x-api-key": "fk-claude"}
        payload_f = {"model": "factoryclaude/claude-fable-5-1-ver", "messages": [{"role": "user", "content": "hi"}]}
        _, headers_f, payload_f = await factory_compat_request_interceptor(
            None, "claude", provider_claude, "fk-claude", "https://api.factory.ai/api/llm/a/v1/messages", headers_f, payload_f
        )
        assert payload_f["model"] == "claude-fable-5.1"
        assert headers_f["x-api-provider"] == "vertex_anthropic"
        assert "speed" not in payload_f

        # 2. OpenAI Responses with fast_models_add
        provider_oai = {
            "provider": "factory_oai",
            "engine": "openai-responses",
            "preferences": {
                "factory_compat": {
                    "fast_models_add": ["gpt-5.6-sol"],
                }
            },
        }
        headers_o = {"Authorization": "Bearer fk-oai"}
        payload_o = {
            "model": "factorygpt/gpt-5.6-sol",
            "temperature": 0.7,
            "instructions": "Base instructions",
            "input": "hello",
        }
        _, headers_o, payload_o = await factory_compat_request_interceptor(
            None, "openai-responses", provider_oai, "fk-oai", "https://api.factory.ai/api/llm/o/v1/responses", headers_o, payload_o
        )
        assert payload_o["model"] == "gpt-5.6-sol"
        assert payload_o["service_tier"] == "fast"
        assert payload_o["store"] is False
        assert "temperature" not in payload_o
        assert headers_o["x-api-provider"] == "openai"
        assert headers_o["OpenAI-Platform"] == "org-bHuLtG1fGmYk5YaOihAAXFBw"
        assert DEFAULT_COMPLIANCE_PROMPT in payload_o["instructions"]

        # 3. OpenAI Chat Completions with Gemini 4 Argon & Droid Core Open Model
        provider_open = {"provider": "factory_open", "engine": "openai", "preferences": {}}
        headers_g = {"Authorization": "Bearer fk-open"}
        payload_g = {
            "model": "gemini-4-argon",
            "temperature": 0.5,
            "messages": [{"role": "user", "content": "hi"}],
        }
        _, headers_g, payload_g = await factory_compat_request_interceptor(
            None, "openai", provider_open, "fk-open", "https://api.factory.ai/api/llm/o/v1/chat/completions", headers_g, payload_g
        )
        assert headers_g["x-api-provider"] == "google"
        assert "temperature" not in payload_g
        assert "service_tier" not in payload_g

    asyncio.run(_run())
