import asyncio

import pytest

from core.channels.gemini_channel import get_gemini_payload
from core.channels.vertex_channel import get_vertex_gemini_payload
from core.models import RequestModel


@pytest.mark.parametrize(
    "payload_builder, engine, base_url",
    [
        (get_gemini_payload, "gemini", "https://generativelanguage.googleapis.com/v1beta"),
        (get_vertex_gemini_payload, "vertex-gemini", "https://aiplatform.googleapis.com/v1beta"),
    ],
    ids=["gemini", "vertex-gemini"],
)
def test_nullable_tool_type_arrays_in_payload(payload_builder, engine, base_url):
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "skip": {"type": ["number", "null"], "description": "Rows to skip"},
            "reverse": {"type": ["null", "number"]},
            "label": {"type": ["string", "null"], "description": "Label", "enum": ["a", "b"]},
            "nested": {
                "type": "object",
                "properties": {"flag": {"type": ["boolean", "null"]}},
            },
            "values": {"type": "array", "items": {"type": ["null", "string"]}},
            "legacy": {"anyOf": [{"type": "number"}, {"type": "null"}]},
            "union": {"type": ["string", "number", "null"]},
        },
        "required": ["pattern"],
    }
    request = RequestModel(
        model="gemini-2.0-flash",
        messages=[{"role": "user", "content": "OK"}],
        stream=False,
        tools=[{
            "type": "function",
            "function": {"name": "grep", "parameters": parameters},
        }],
    )
    original_request = request.model_dump(exclude_unset=True)
    provider = {
        "base_url": base_url,
        "model": [request.model],
        "project_id": "test-project",
    }

    _, _, payload = asyncio.run(payload_builder(request, engine, provider))
    function = payload["tools"][0]["function_declarations"][0]

    assert function["name"] == "grep"
    assert function["parameters"] == {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "skip": {"type": "number", "nullable": True, "description": "Rows to skip"},
            "reverse": {"type": "number", "nullable": True},
            "label": {"type": "string", "nullable": True, "description": "Label", "enum": ["a", "b"]},
            "nested": {
                "type": "object",
                "properties": {"flag": {"type": "boolean", "nullable": True}},
            },
            "values": {"type": "array", "items": {"type": "string", "nullable": True}},
            "legacy": {"type": "number", "nullable": True},
            "union": {"type": ["string", "number", "null"]},
        },
        "required": ["pattern"],
    }
    assert request.model_dump(exclude_unset=True) == original_request
