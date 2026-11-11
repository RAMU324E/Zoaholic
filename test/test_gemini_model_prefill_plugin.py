import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plugins.gemini_model_prefill import (
    AIS_TEXT_CASTER_SIG,
    AIS_TEXT_CASTER_THOUGHT_TEXT,
    DEFAULT_PREFILL,
    apply_model_prefill,
    gemini_model_prefill_request_interceptor,
    resolve_prefill_text,
)


def _user(text="hi"):
    return {"role": "user", "parts": [{"text": text}]}


def test_appends_model_turn_after_user_turn():
    payload = {"contents": [_user()]}
    assert apply_model_prefill(payload, "好的") is True
    assert payload["contents"][-1] == {"role": "model", "parts": [{"text": "好的"}]}


def test_skips_tool_result_round_and_existing_model_tail():
    tool_round = {"contents": [_user(), {"role": "user", "parts": [{"functionResponse": {"name": "f", "response": {}}}]}]}
    assert apply_model_prefill(tool_round, "x") is False
    assert len(tool_round["contents"]) == 2

    model_tail = {"contents": [_user(), {"role": "model", "parts": [{"text": "client prefill"}]}]}
    assert apply_model_prefill(model_tail, "x") is False
    assert model_tail["contents"][-1]["parts"][0]["text"] == "client prefill"

    # Tool round followed by extra user turns (tool images, per-round context).
    tool_then_user = {"contents": [
        _user(),
        {"role": "model", "parts": [{"functionCall": {"name": "f", "args": {}}}]},
        {"role": "user", "parts": [{"functionResponse": {"name": "f", "response": {}}}]},
        _user("tool produced images"),
        _user("dynamic context"),
    ]}
    assert apply_model_prefill(tool_then_user, "x") is False

    # A new user message after an earlier, finished tool round is prefilled.
    new_user_round = {"contents": [
        {"role": "user", "parts": [{"functionResponse": {"name": "f", "response": {}}}]},
        {"role": "model", "parts": [{"text": "done"}]},
        _user("next request"),
    ]}
    assert apply_model_prefill(new_user_round, "x") is True

    for payload in ({}, {"contents": []}, {"messages": [{"role": "user", "content": "hi"}]}):
        assert apply_model_prefill(payload, "x") is False


def test_option_parsing_keeps_commas_and_falls_back_to_default():
    assert resolve_prefill_text(None) == DEFAULT_PREFILL
    assert resolve_prefill_text("") == DEFAULT_PREFILL
    assert resolve_prefill_text({"text": "  "}) == DEFAULT_PREFILL
    assert resolve_prefill_text({"text": "好的，master"}) == "好的，master"
    assert resolve_prefill_text("text=a,b,c") == "a,b,c"
    assert resolve_prefill_text("plain text") == "plain text"
    assert resolve_prefill_text("text=<think>\nxxx") == "<think>\nxxx"
    assert resolve_prefill_text({"text": "<think>\nxxx\n"}) == "<think>\nxxx"


def test_interceptor_reads_channel_options():
    provider = {"preferences": {"enabled_plugins": [{"name": "gemini_model_prefill", "params": {"text": "开始", "ais_cot": False}}]}}
    payload = {"contents": [_user()]}
    url, headers, out = asyncio.run(
        gemini_model_prefill_request_interceptor(None, "gemini", provider, None, "u", {}, payload)
    )
    assert out["contents"][-1] == {"role": "model", "parts": [{"text": "开始"}]}

    default_provider = {"preferences": {"enabled_plugins": ["gemini_model_prefill"]}}
    payload2 = {"contents": [_user()]}
    asyncio.run(gemini_model_prefill_request_interceptor(None, "gemini", default_provider, None, "u", {}, payload2))
    assert payload2["contents"][-2] == {
        "role": "model",
        "parts": [{"text": AIS_TEXT_CASTER_THOUGHT_TEXT, "thought": True, "thoughtSignature": AIS_TEXT_CASTER_SIG}],
    }
    assert payload2["contents"][-1] == {"role": "user", "parts": [{"text": ".", "thought": True}]}
