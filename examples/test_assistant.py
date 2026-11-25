"""The example works end to end: a run started through the service's run routes calls the
harness's tools and the example's own, finishes, and its frames can be read again from the
start, as a viewer that reconnects would. A scripted model stands in for a real one, so
nothing here calls a provider.

A process holds one wire namespace, so this runs in its own: ``pytest examples``, or
through ``tests/test_examples.py``, which starts it.
"""

import json
import pathlib
import sys
from collections.abc import AsyncGenerator

import pytest
from fastapi.testclient import TestClient
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.genai import types

EXAMPLES = pathlib.Path(__file__).resolve().parent
BASE = "/runs/apps/assistant/users/u1/sessions/s1"
PLAN = [
    {"content": "Count the words", "status": "in_progress"},
    {"content": "Report the count", "status": "pending"},
]


class ScriptedLlm(BaseLlm):
    """Plans and counts on the first call, then answers with what the count said."""

    model: str = "scripted"

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        results = {
            part.function_response.name: part.function_response.response
            for content in llm_request.contents
            for part in content.parts or []
            if part.function_response
        }
        if "word_count" not in results:
            parts = [
                types.Part(function_call=types.FunctionCall(id="p1", name="todo_list", args={"todos": PLAN})),
                types.Part(function_call=types.FunctionCall(id="w1", name="word_count", args={"text": "one two three"})),
            ]
        else:
            parts = [types.Part(text=f"It has {results['word_count']['words']} words.")]
        yield LlmResponse(content=types.Content(role="model", parts=parts))


@pytest.fixture
def client(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLES))
    import assistant
    import serve

    monkeypatch.setattr(assistant.agent.app.root_agent, "model", ScriptedLlm())
    with TestClient(serve.build()) as client:
        yield client
    for name in [n for n in sys.modules if n == "assistant" or n.startswith("assistant.") or n == "serve"]:
        del sys.modules[name]


def _frames(client: TestClient, after: int = 0) -> list[dict]:
    with client.stream("GET", f"{BASE}/events", params={"after": after}) as response:
        assert response.status_code == 200
        return [json.loads(line[6:]) for line in response.iter_lines() if line.startswith("data: ")]


def test_a_run_uses_the_tools_finishes_and_can_be_read_again(client):
    assert client.post("/apps/assistant/users/u1/sessions/s1").status_code == 200
    reserved = client.post(BASE)
    assert reserved.status_code == 201
    body = {
        "appName": "assistant",
        "userId": "u1",
        "sessionId": "s1",
        "newMessage": {"role": "user", "parts": [{"text": "How many words in 'one two three'?"}]},
        "streaming": False,
    }
    started = client.post(f"{BASE}/start", params={"runId": reserved.json()["runId"]}, json=body)
    assert started.status_code == 202, started.text

    first = _frames(client)
    calls = [p["functionCall"]["name"] for f in first for p in f.get("content", {}).get("parts", []) if "functionCall" in p]
    answers = {p["functionResponse"]["name"]: p["functionResponse"]["response"]
               for f in first for p in f.get("content", {}).get("parts", []) if "functionResponse" in p}
    texts = [p["text"] for f in first for p in f.get("content", {}).get("parts", []) if p.get("text")]
    assert calls == ["todo_list", "word_count"]
    assert answers["word_count"] == {"words": 3, "characters": 13}
    assert texts[-1] == "It has 3 words."
    assert first[-1] == {"assistantRun": {"status": "finished"}}
    assert client.get(BASE).json()["status"] == "finished"

    # A viewer that comes back reads the same run from the start.
    assert _frames(client) == first
