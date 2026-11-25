# Replies

A model reply can finish and still be unusable: a tool call whose arguments are text where
the schema wants a number, a reply cut at the output limit halfway through a call, or a reply
with nothing in it. In plain ADK the first fails the tool, the second can run a call with
half-written arguments, and the third ends the turn with nothing to show. This package holds
the logic for all three; `BadRepliesPlugin` applies it (see [plugins](plugins.md)). The one
name a product imports is `from harness.replies import agent_note`, for `agent_note.add_note`,
which lets its own plugin send the model a note the same way.

How this sits beside ADK's reflect-and-retry plugins is in [adk](adk.md).

## What it guarantees

All of this holds once `BadRepliesPlugin` is in the app's plugins.

### Argument repair

- **Only where the schema leaves one reading.** A tool call's arguments are checked against the
  tool's declared JSON Schema (`parameters_json_schema`, or the genai `Schema` turned into
  one). These are changed:
  - a number or integer sent as text (`"5"`), when it parses to a finite number (an integer
    must be whole and no larger than `2**53 - 1` in size);
  - `"true"` or `"false"` where a boolean is declared;
  - an object or list sent as JSON text;
  - `{}` for an optional property whose one plain type isn't `object`: it is left out.
- **Everything else is left for the tool to judge:** a value that fits two types, a string
  where a string is allowed, a property the schema doesn't name, a `null` (for some apps it
  means "clear this field"). A type list or `anyOf`/`oneOf` is repaired only toward its one
  non-null branch. `$ref` is followed into `$defs` or `definitions`. Nesting stops at depth 8.
- **Logged without values.** Each repair is logged as its path and change
  (`items[2]: text to integer`), never the value, which can be the person's data.

### A reply cut at the output limit

- **None of its tool calls runs.** Every call in a reply that stopped at the output limit is
  answered with an error result ("Not run: your reply reached the output limit while it was
  writing this ... call"). With `BadRepliesPlugin` ahead of the approval and tool-ledger
  plugins, as [plugins](plugins.md) orders them, none runs, none is put to the person for
  approval, and none is recorded as run. The reply's text stands,
  and the limit error is cleared so the turn goes on.
- **A dropped call gets a note.** When ADK's LiteLLM client dropped the cut call because its
  arguments didn't parse, the reply gets a note telling the model to send it again, split
  into smaller calls. Once per run.
- **A streamed reply's text stands.** A stream that ends with its text, then the dropped call,
  keeps that text as part of the turn.
- **A cut reply with no call is left as the provider gave it:** its text is the answer so far.

### An empty reply

- A reply with no text and no tool call (whitespace or reasoning alone counts as empty) gets
  a note asking the model to go on: answer, or make the next tool call. Once per run. A
  second empty reply in the same run ends the turn as it would without the harness.

### Notes (`agent_note`)

- **A note keeps the loop going.** ADK ends its loop on a reply with no tool call, and a
  callback can't ask the model again. So `agent_note.add_note(response, kind)` appends one
  call named `agent_note` with `{"kind": kind}` to the reply, in place, and
  `BadRepliesPlugin` answers it with the note text for that kind. No tool needs to be
  registered under that name.
- **The provider never sees that call.** Before each model call, the request is rewritten:
  the note call goes, and its answer becomes plain user text. A model turn that held only
  the note call and reasoning is dropped. The session keeps the call, since ADK's loop needs it there.
- **The loop guard ignores it.** `LoopGuardPlugin` never counts the note call as a repeat.

## What it costs

- **Repair:** no model call; a walk over each call's arguments, on the final reply only (not
  on partial stream chunks).
- **A cut reply:** no extra call by itself. The model's next step, which sends the calls
  again, is an ordinary model call.
- **An empty reply or a dropped call:** one extra model call each, at most once per run per
  kind.
- **Storage:** the note call and its answer are saved as events in the session, like any
  tool call. A few run-state keys hold which calls are held and which notes were sent.

## Configure it

There are no settings objects. `BadRepliesPlugin(notes=None)` takes extra note kinds, as a
mapping from kind to the text the model gets. A product's own plugin adds the note call;
`BadRepliesPlugin` answers it.

```python
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin

from harness.plugins import BadRepliesPlugin
from harness.replies import agent_note

NO_SOURCES = "no_sources"


class SourcesPlugin(BasePlugin):
    """Asks the model once more when its answer names no source."""

    def __init__(self) -> None:
        super().__init__(name="sources")

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> Optional[LlmResponse]:
        parts = (llm_response.content.parts if llm_response.content else None) or []
        text = "".join(part.text or "" for part in parts)
        if not llm_response.partial and text and "http" not in text:
            agent_note.add_note(llm_response, NO_SOURCES)
        # In place, returning nothing, so the plugins after this one still see the reply.
        return None


plugins = [
    BadRepliesPlugin(notes={NO_SOURCES: "Add the link to each source you used."}),
    SourcesPlugin(),
]
```

A note kind that `BadRepliesPlugin` doesn't know is answered with the empty-reply note.
Plugin order rules are in [plugins](plugins.md).

## Test code that uses it

- **A scripted model.** A `BaseLlm` subclass that yields chosen `LlmResponse`s, as in
  `examples/test_assistant.py`, run with ADK's `InMemoryRunner`. To test a cut reply, yield a
  response with `finish_reason=types.FinishReason.MAX_TOKENS` and a function call; to test an
  empty one, yield a response with no parts. Then read the session's events, or the
  requests the scripted model was sent, to see what the model got next.
- **Through the example.** `examples/test_assistant.py` runs `BadRepliesPlugin` in a whole app
  with a scripted model and no provider.

## Reference

| Name | What it is |
| --- | --- |
| `agent_note` | Module; its public name is `add_note(response, kind)`, which appends the note call for `kind` to a reply in place. |
