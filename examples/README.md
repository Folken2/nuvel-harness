# Example: an assistant on nuvel-harness

A small product built from the harness's public names only:

- `assistant/__init__.py` names the product (its wire namespace) and its file layout, the two
  things a product says before anything runs.
- `assistant/agent.py` is one ADK `LlmAgent` with a tool of its own (`word_count`) and the
  harness's `todo_list` (a plan the agent keeps) and `ask_person` (a question that pauses the
  run until the person answers). Around it, the harness's plugins repair bad replies and stop
  a loop of identical calls.
- `serve.py` serves it with ADK's API server plus the harness's run routes: a run belongs to
  the service, so it keeps going when the connection drops, and a viewer that comes back
  reads it from the start.

## Run it

From the repository root, with the harness installed (`pip install -e '.[test]'`) and a
model key in the environment:

```bash
export GOOGLE_API_KEY=...                  # or pick any LiteLLM model:
# export ASSISTANT_MODEL=openai/gpt-4.1-mini OPENAI_API_KEY=...
adk web examples                           # ADK's dev UI, http://localhost:8000
```

Or as the service, with the run routes:

```bash
pip install uvicorn
python examples/serve.py
```

A run through the routes, in four calls (session `s1` for user `u1`):

```bash
B=http://127.0.0.1:8000
curl -X POST $B/apps/assistant/users/u1/sessions/s1
RUN=$(curl -s -X POST $B/runs/apps/assistant/users/u1/sessions/s1 | python -c 'import json,sys; print(json.load(sys.stdin)["runId"])')
curl -X POST "$B/runs/apps/assistant/users/u1/sessions/s1/start?runId=$RUN" \
  -H 'content-type: application/json' \
  -d '{"appName":"assistant","userId":"u1","sessionId":"s1","newMessage":{"role":"user","parts":[{"text":"Plan a three-step study session, then count the words in your plan."}]}}'
curl -N $B/runs/apps/assistant/users/u1/sessions/s1/events     # stop it any time, run it again
```

## Test it

`test_assistant.py` drives a whole run through the service with a scripted model, so it
needs no key and calls no provider:

```bash
pytest examples
```
