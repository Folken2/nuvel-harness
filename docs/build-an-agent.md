# Build your own agent

This walkthrough builds a small product on the harness: one ADK agent with a tool of its own,
the harness's plan and question tools, a few of its plugins, and the HTTP service that runs it.
It ends with a test that drives a whole run through the service with a scripted model, so you
can check every step without a model key.

It is the same agent as [`examples/assistant`](../examples/assistant), renamed `helper` and
laid out as a project of its own, outside this repository. Every file and command below was
run as written.

## 1. Install

Python 3.13 or later.

```bash
python3.13 -m venv .venv
.venv/bin/pip install "nuvel-harness @ git+https://github.com/Folken2/nuvel-harness" pytest httpx
```

`pytest` and `httpx` are only for the test in step 5 (FastAPI's `TestClient` needs `httpx`).
A product pins the harness by commit SHA (`...nuvel-harness@<sha>`) rather than tracking the
default branch.

The project ends up like this:

```text
myproduct/
  agents/
    helper/
      __init__.py   # names the product and its file layout
      agent.py      # the agent and its plugins
  serve.py          # the HTTP service
  test_helper.py    # a whole run, no model key
```

ADK loads every folder under `agents/` as an app, by its folder name.

## 2. Name the product

`agents/helper/__init__.py`:

```python
from harness import wire_names
from harness.files import FileLayout, set_layout

wire_names.use_namespace("helper")
set_layout(
    FileLayout(
        original="original/",
        output="output/",
        reading="markdown/",
        tool_results="tool-results/",
        pages="app-pages/",
    )
)

from . import agent as agent  # noqa: E402  after the namespace and layout
```

A product says two things before anything else runs:

- **Its wire namespace.** Every key the harness puts on an event, and every frame a run sends
  beside ADK's events, is spelled under it. With `helper`, a run's last frame is
  `{"helperRun": {"status": "finished"}}` and a model call's cost is under the metadata key
  `helper_cost_usd`. Clients and stored events depend on these names, so the namespace is
  said once per process: a second, different one raises `RuntimeError`. See
  [service.md](service.md).
- **Where a conversation's files live**, as name prefixes in ADK's artifact store: the kept
  original of an upload, what a job saves to share, text the agent reads a chunk at a time,
  the whole of a tool result the model was shown cut, and pages a server attached to a tool
  call. See [files.md](files.md).

The agent module is imported last, once both are said.

## 3. Write the agent

`agents/helper/agent.py`:

```python
from __future__ import annotations

import os
from typing import Union

from google.adk.agents import LlmAgent
from google.adk.apps import App, ResumabilityConfig
from google.adk.models import BaseLlm
from google.adk.models.lite_llm import LiteLlm

from harness.plugins import BadRepliesPlugin, LoopGuardPlugin, PlanFieldPlugin, TurnLogPlugin
from harness.runs import LoopSettings, loop_guard
from harness.tools import ask_tools, plan_tools

INSTRUCTION = """You are a careful assistant.

For any task with more than one step, keep a plan with todo_list and update it as you go.
When a choice is the person's to make, ask with ask_person and wait for the answer.
Use word_count when asked about the length of a text."""


def word_count(text: str) -> dict[str, int]:
    """Counts the words and characters of a text.

    Args:
        text: The text to count.
    """
    return {"words": len(text.split()), "characters": len(text)}


def build_app(model: Union[str, BaseLlm]) -> App:
    agent = LlmAgent(
        name="helper",
        model=model,
        instruction=INSTRUCTION,
        tools=[word_count, *plan_tools(), *ask_tools()],
    )
    return App(
        name="helper",
        root_agent=agent,
        plugins=[
            PlanFieldPlugin(),
            BadRepliesPlugin(notes={loop_guard.KIND: loop_guard.NOTE}),
            LoopGuardPlugin(LoopSettings(note_at=3, stop_at=5), ask_tool="ask_person"),
            TurnLogPlugin(),
        ],
        resumability_config=ResumabilityConfig(is_resumable=True),
    )


def _model() -> Union[str, BaseLlm]:
    name = os.environ.get("HELPER_MODEL", "gemini-2.5-flash")
    return LiteLlm(model=name) if "/" in name else name


app = build_app(_model())
```

What each piece does:

- **`LlmAgent` and `App` are ADK's own.** The harness adds tools and plugins to them; it does
  not wrap or replace them. `word_count` is a plain function, which ADK turns into a tool from
  its signature and docstring.
- **`plan_tools()`** gives the agent `todo_list`, a plan it keeps and updates as it works.
  **`ask_tools()`** gives it `ask_person`, a question that pauses the run until the person
  answers. See [tools.md](tools.md).
- **The plugins**, in this order (see [plugins.md](plugins.md)):
  - `PlanFieldPlugin` takes the plan field off a tool call.
  - `BadRepliesPlugin` repairs replies a model got slightly wrong: arguments sent as the
    wrong JSON type, an empty reply, a reply cut off at the output limit. See
    [replies.md](replies.md).
  - `LoopGuardPlugin` counts identical calls: at the third it adds a note to the model, at
    the fifth it holds the call and has the model ask the person through `ask_person`. See
    [runs.md](runs.md). The note travels as a call `BadRepliesPlugin` answers, so
    `BadRepliesPlugin` is given the loop guard's note text by its kind
    (`notes={loop_guard.KIND: loop_guard.NOTE}`); without it, the model would get the
    empty-reply note instead.
  - `TurnLogPlugin` logs tool names, durations and error types, never content.

  Order matters only where the harness says so: the plan field comes off a call before the
  reply is repaired, and a reply is repaired before the loop guard counts its calls.
- **`ResumabilityConfig(is_resumable=True)`** is what lets the answer to `ask_person` resume
  the same run instead of starting a new one.
- **The model** is a Gemini id by default (it needs `GOOGLE_API_KEY`), or any LiteLLM id such
  as `openai/gpt-4.1-mini` with that provider's key. The test in step 5 replaces it.

## 4. Serve it

`serve.py`:

```python
import pathlib

from fastapi import FastAPI

from harness.service import ServiceOptions, Stores, create_service

AGENTS = pathlib.Path(__file__).parent / "agents"


def build() -> FastAPI:
    service = create_service(
        ServiceOptions(agents_dir=str(AGENTS), safe_to_repeat=("word_count",)),
        Stores(session_uri=None, connect_args=lambda uri: {}),
        tasks=lambda runs: None,
    )
    return service.app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(build(), host="127.0.0.1", port=8000)
```

`create_service` builds ADK's API server over the stores it is given, then adds `/health` and
the harness's run routes. A run started through those routes belongs to the service, not to
the HTTP request: close the connection and the run keeps going; open the event stream again
and it is read from the start.

- `Stores(session_uri=None, ...)` keeps sessions in memory, so nothing survives a restart.
  Give it a Postgres URI and every completed event is saved, which is what lets an
  interrupted run resume after a restart. `connect_args` returns the database driver's
  connection options for that URI (see [service.md](service.md)).
- `safe_to_repeat` names the tools that only read. When a run resumes after a restart, an
  interrupted call to one of these is answered with a request to make the call again; any
  other interrupted call is answered as having an unknown outcome, and the model is told to
  check. The harness never runs an interrupted call again on its own.
- `tasks=lambda runs: None` runs no background tasks. See [runs.md](runs.md) to turn them on.

Start it:

```bash
.venv/bin/python serve.py
```

and, from another shell, check it and reserve a run (session `s1` for user `u1`):

```bash
B=http://127.0.0.1:8000
curl $B/health
curl -X POST $B/apps/helper/users/u1/sessions/s1
curl -X POST $B/runs/apps/helper/users/u1/sessions/s1
```

The last call answers `201` with `{"runId": "...", "status": "reserved", ...}`. A run's
routes, all under `/runs/apps/{app}/users/{user}/sessions/{session}`:

| Route | What it does |
| --- | --- |
| `POST` | Reserves the session's run. `409` if one is already going. |
| `POST /start?runId=...` | Starts the reserved run. The body is what ADK's `/run_sse` takes. Answers `202` once ADK has accepted it. |
| `GET /events?after=N` | The run's frames as server-sent events, from frame `N` (default 0). |
| `GET` | The run's status: `reserved`, `running`, `finished`, `stopped` or `failed`. |
| `POST /stop` | Stops the run and its background tasks. |

The rest (`/resume`, `/replay`, `/steer`) are in [runs.md](runs.md). A real run needs a model
key; to try it without one, use the test below.

To use ADK's own dev UI instead of the service, run `.venv/bin/adk web agents` (it serves on
port 8000 and lists `helper`). The dev UI talks to ADK's routes, not the run routes.

## 5. Test it with a scripted model

`test_helper.py`:

```python
import json
import pathlib
import sys
from collections.abc import AsyncGenerator

import pytest
from fastapi.testclient import TestClient
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.genai import types

HERE = pathlib.Path(__file__).parent
BASE = "/runs/apps/helper/users/u1/sessions/s1"


class ScriptedLlm(BaseLlm):
    """Calls word_count first, then answers with what it returned."""

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
            call = types.FunctionCall(id="w1", name="word_count", args={"text": "one two three"})
            parts = [types.Part(function_call=call)]
        else:
            parts = [types.Part(text=f"It has {results['word_count']['words']} words.")]
        yield LlmResponse(content=types.Content(role="model", parts=parts))


@pytest.fixture
def client(monkeypatch):
    monkeypatch.syspath_prepend(str(HERE / "agents"))
    monkeypatch.syspath_prepend(str(HERE))
    import helper
    import serve

    monkeypatch.setattr(helper.agent.app.root_agent, "model", ScriptedLlm())
    with TestClient(serve.build()) as client:
        yield client
    for name in [n for n in sys.modules if n == "helper" or n.startswith("helper.")]:
        del sys.modules[name]


def frames(client: TestClient) -> list[dict]:
    with client.stream("GET", f"{BASE}/events") as response:
        return [json.loads(line[6:]) for line in response.iter_lines() if line.startswith("data: ")]


def test_a_run_counts_words_and_finishes(client):
    assert client.post("/apps/helper/users/u1/sessions/s1").status_code == 200
    run_id = client.post(BASE).json()["runId"]
    body = {
        "appName": "helper",
        "userId": "u1",
        "sessionId": "s1",
        "newMessage": {"role": "user", "parts": [{"text": "How many words in 'one two three'?"}]},
    }
    assert client.post(f"{BASE}/start", params={"runId": run_id}, json=body).status_code == 202

    seen = frames(client)
    texts = [p["text"] for f in seen for p in f.get("content", {}).get("parts", []) if p.get("text")]
    assert texts[-1] == "It has 3 words."
    assert seen[-1] == {"helperRun": {"status": "finished"}}
    assert frames(client) == seen  # a viewer that comes back reads the same run
```

Run it from `myproduct/`:

```bash
.venv/bin/pytest -q test_helper.py
```

```text
.                                                                        [100%]
=============================== warnings summary ===============================
test_helper.py::test_a_run_counts_words_and_finishes
  .../fastapi/routing.py:381: DeprecationWarning: Please use create_session instead. This will be removed in future releases.
    return await dependant.call(**values)

1 passed, 1 warning
```

The warning is expected. It comes from ADK, which marks the route that creates a session under
an id you choose (`POST /apps/{app}/users/{user}/sessions/{id}`) as deprecated. The test and
the `curl` calls in step 4 use that route to name the session `s1`. If `pytest-asyncio` is installed too (the
harness's own test extra includes it), it adds a second warning that
`asyncio_default_fixture_loop_scope` is unset. Set that option in your pytest configuration
to silence it.

How the test works:

- **A scripted model is a `BaseLlm` subclass.** ADK calls its `generate_content_async` with
  the whole request, so it can answer from what the tools returned. It is swapped onto the
  agent before the service is built, so nothing calls a provider.
- **The run goes through the real service**: reserve, start, then read the frames. The last
  frame is the harness's own, `helperRun` under the product's namespace.
- **Reading the events twice gives the same frames**, which is the guarantee a viewer that
  reconnects relies on.
- A process holds one wire namespace, so a test suite that also uses another one (as the
  harness's own does) runs this file in its own process. See `tests/test_examples.py`.

The harness's own version of this test, which also checks the `todo_list` call, is
[`examples/test_assistant.py`](../examples/test_assistant.py).

## Where to go next

- Keep sessions in Postgres and resume runs after a restart: [service.md](service.md),
  [runs.md](runs.md).
- Route each turn to a model, retry well, and fall back: [routing.md](routing.md).
- Keep long conversations inside the context window: [context.md](context.md).
- Price and record every model and tool call: [ledger.md](ledger.md).
- Ask the person before a tool with effects runs: [approval.md](approval.md).
- Large toolsets, MCP servers, skills and background tasks: [tools.md](tools.md).
- Every page: [the docs index](README.md).
