# Routing

A product that lets a person bring their own model key, or that sends calls through a router,
needs each model call to go out on the right key, to fall back when that key or model is
refused, and to retry the failures that lift by themselves without retrying the ones that
don't. This package does that inside ADK's LiteLLM model, as a client it hands every request
to. A product imports
`from harness.routing import ClientSettings, KeyedLiteLLMClient, retry_policy, routed, set_route, TurnRoute`;
from `retry_policy` the public name is `Settings`.

How this sits beside ADK's own retry plugins is in [adk](adk.md).

## What it guarantees

### The route (`TurnRoute`, `set_route`, `routed`)

- **Per call, never shared between turns.** The route lives in a `contextvars.ContextVar`, so
  concurrent turns in one process each see their own. `set_route(route)` sets it for the
  current context; `None` clears it.
- **A scoped route comes back.** `with routed(route):` sends the calls inside on `route`, then
  restores whatever route was set before, even when the block raises. Use it for a call made
  from inside a tool that goes to another model than the turn's.
- **No route, no key.** With no route set, a call goes out with the arguments it came with and
  has no fallback.

### The client (`KeyedLiteLLMClient`)

- **The key is added per request.** With a route set, the first request carries
  `route.api_key`. If `route.fallback_model` is set, a second target is the fallback model,
  with `route.fallback_key` when given and with the turn's key removed.
- **A step for each failure.** Each failed request is sorted into a kind, and the kind decides
  the next step (`tests/test_retry_policy.py`):

  | Kind | Examples | Next step |
  | --- | --- | --- |
  | rate limited | a 429, a per-minute quota | retry after a wait, then fall back |
  | transient | 408, 409, 500, 502, 503, 504, 529, a dropped connection | retry after a wait, then fall back |
  | quota | 402, "insufficient credits", a per-day quota | fall back; never sent again on the same route |
  | refused | an authentication or permission error, an unknown model | fall back |
  | overflow | the request is too long for the model | shrink once and send again |
  | timed out | the client's own time limit (`request_timeout_seconds`) passed with no byte from the API | fall back; never sent again on the same route, which would wait the limit twice |
  | other | anything else | give up |

- **A 429 is never overflow.** The status is checked before the wording, so a 429 that says
  "too many tokens" is a rate limit.
- **The server's wait wins.** A delay in the error body (a `retry_after_seconds` field, or a
  `retryDelay` such as `"17s"`) is used first, then the `retry-after` or `retry-after-ms`
  header (seconds or an HTTP date). Otherwise the wait doubles from `first_wait`, capped at
  `max_wait`, and is spread by a jitter of 0.8 to 1.2 times.
- **A wait never outlasts the budget.** A retry happens only while the wait fits within
  `max_wait`, within what the call has left of `retry_budget_seconds`, and, inside a run with
  `RunLimits`, within the run's remaining retry wait and its time ceiling (see
  [plugins](plugins.md)). Otherwise the call falls back or gives up.
- **Overflow is shrunk once.** The messages are sent again with the tool results of all but the
  newest `overflow_kept_steps` steps cut to a 200-character preview and a note; the results of
  one step stay together. The note tells the model not to call the tool again for the rest. If
  nothing would get smaller, or the shrunk request is refused again, the call gives up. An
  overflow never falls back (`tests/test_overflow.py`).
- **No hidden retries.** Each request is sent with `max_retries: 0`, so the SDK under LiteLLM
  doesn't retry on its own, and every request the client makes is one ledger attempt. A
  fallback is the same call's next attempt (see [ledger](ledger.md)).
- **A started stream is never retried.** Only a request that raised is sent again.
- **A preference never fails a call.** When LiteLLM refuses one of the `preferences` arguments
  for a model (`UnsupportedParamsError`, before anything is sent), the request goes again
  without them.
- **No provider text in the logs.** Retry and fallback log lines name the target model, the
  kind and the error's type only.

## What it costs

- **Latency:** each retry sleeps for its wait (at most `max_wait`, and never past the budget).
  A fallback adds one request. An overflow adds one request with a smaller prompt.
- **Model calls:** one request per attempt; a retried or fallen-back call is several
  attempts, each recorded in the ledger when one is set.
- **Storage and memory:** nothing kept by this package beyond the context variable.
- A call with no route and no failure costs nothing extra.

## Configure it

`ClientSettings` (only `request_timeout_seconds` has a default):

| Field | Meaning |
| --- | --- |
| `retry` | `retry_policy.Settings(retries, first_wait, max_wait)`: retries per target, the first wait in seconds, the longest single wait. |
| `retry_budget_seconds` | The most one call may spend waiting between its retries, across targets. |
| `overflow_kept_steps` | How many newest steps keep their tool results whole when a too-long request is shrunk. |
| `request_timeout_seconds` | How long a request may go without a byte from the API, before the answer starts and between a stream's chunks, before it fails as timed out. A stalled stream then ends the call instead of holding the run; a long answer that keeps streaming is never cut. A call that sets its own `timeout` keeps it. Default 0: none of ours, LiteLLM's applies. |

`TurnRoute(api_key, fallback_model=None, fallback_key=None)`.

`KeyedLiteLLMClient(settings, *, call_transforms=(), request_transforms=(), preferences=(), on_whole_answer=None)`:

- `settings` is a function returning `ClientSettings`, asked at each call.
- `call_transforms` take `(model, kwargs)` and return kwargs, once per call, with the model
  the call asked for.
- `request_transforms` do the same per request, with the model that request goes to (after a
  fallback, the fallback's).
- `preferences` take a model and return arguments to add when the model accepts them.
- `on_whole_answer` gets each answer that came back whole, not streamed (see
  [ledger](ledger.md)).

```python
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

from harness.routing import ClientSettings, KeyedLiteLLMClient, TurnRoute, retry_policy, routed, set_route

SETTINGS = ClientSettings(
    retry=retry_policy.Settings(retries=3, first_wait=1.0, max_wait=20.0),
    retry_budget_seconds=60.0,
    overflow_kept_steps=2,
)

client = KeyedLiteLLMClient(lambda: SETTINGS)
agent = LlmAgent(name="assistant", model=LiteLlm(model="provider/model-name", llm_client=client))


def use_the_persons_key(api_key: str, router_model: str) -> None:
    # Before the turn's model call: its key, and where it goes if that key is refused.
    set_route(TurnRoute(api_key=api_key, fallback_model=router_model))


async def from_a_tool(advisor_key: str) -> None:
    with routed(TurnRoute(api_key=advisor_key)):
        ...  # a model call made here goes on the advisor's key
```

## Test code that uses it

- **No network, through the real client.** A `request_transforms` entry can add LiteLLM's
  `mock_response` per target. LiteLLM then answers without sending anything, and a
  `mock_response` that is an exception is raised as that error. That drives the real
  retry and fallback path:

  ```python
  import litellm
  import pytest
  from harness.routing import ClientSettings, KeyedLiteLLMClient, TurnRoute, retry_policy, routed

  # Model ids need a provider prefix LiteLLM knows; with mock_response nothing is sent.
  FIRST, FALLBACK = "openai/first", "openai/fallback"

  def scripted(model, kwargs):
      if model == FIRST:
          refused = litellm.AuthenticationError(message="bad key", llm_provider="openai", model=model)
          return {**kwargs, "mock_response": refused}
      return {**kwargs, "mock_response": "from the fallback"}

  settings = ClientSettings(retry_policy.Settings(retries=0, first_wait=0.0, max_wait=0.0), 0.0, 2)
  client = KeyedLiteLLMClient(lambda: settings, request_transforms=[scripted])

  @pytest.mark.asyncio
  async def test_a_refused_key_falls_back():
      with routed(TurnRoute(api_key="test-key", fallback_model=FALLBACK)):
          answer = await client.acompletion(FIRST, [{"role": "user", "content": "hi"}], None)
      assert answer.choices[0].message.content == "from the fallback"
  ```

- **Settings are passed in**, so a test sets `first_wait=0.0` and never sleeps.
- **The routing rules themselves** are tested as a pure table in `tests/test_retry_policy.py`
  (which kind an error is, the server's delay, the next step) and the shrink in
  `tests/test_overflow.py`.
- **For the agent above the client**, replace the model with a scripted `BaseLlm` as in
  `examples/test_assistant.py`; the client is then not used at all.

## Reference

| Name | What it is |
| --- | --- |
| `ClientSettings` | Retry settings, the call's retry wait budget and the steps kept whole on overflow. |
| `KeyedLiteLLMClient` | ADK's LiteLLM client with the turn's key, retries, fallback and overflow shrink; every request a ledger attempt. |
| `retry_policy` | Module; its public name is `Settings(retries, first_wait, max_wait)`. |
| `routed` | Context manager: calls inside go on a given route, the previous one comes back after. |
| `set_route` | Sets the route for the current context. |
| `TurnRoute` | A call's key, and the fallback model and key to use if it is refused. |
