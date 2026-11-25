# Context

Every model call re-sends the whole history, so a long run gets slower and more expensive
with each step, and in the end no longer fits the model's window. This package keeps the
prompt small: it rolls old turns into summaries, sends old tool results as short previews,
marks a stable prefix for the provider's prompt cache, and sets the point at which history is
rolled up from the window of the model the run is on. A product imports
`from harness.context import CompactionSettings, events_compaction, MaskSettings, prompt_cache, PromptCacheSettings, ProviderReport, window_of, WindowSettings`.

The settings live here; the callbacks that apply them are plugins (`ContextBudgetPlugin`,
`OldResultsPlugin`, `ToolResultLimitPlugin`, `PrefixWatchPlugin`, see [plugins](plugins.md)).
How this compares with ADK's own `ContextFilterPlugin` and summarizer is in [adk](adk.md).

## What it guarantees

### Compaction (`events_compaction`, `CompactionSettings`)

- **ADK's own compaction, metered.** `events_compaction` returns ADK's
  `EventsCompactionConfig` with the four numbers mapped to `compaction_interval`,
  `overlap_size`, `token_threshold` and `event_retention_size`. Its summarizer is a subclass
  of ADK's `LlmEventSummarizer` that records each summary call in the usage ledger as a
  compaction operation, with the cost, model and response id a `ProviderReport` reads
  (see [ledger](ledger.md)).
- **The summarizer model is built once.** `summarizer_model` is called once, inside
  `events_compaction`.
- **The person's words survive every summary.** After each summary, the person's last five
  messages (`KEPT_MESSAGES`) are appended word for word, each kept to 2,000 characters
  (`MESSAGE_CHARS`; a longer one keeps its first and last 1,000 with a note of what was
  cut). A later summary carries forward what the earlier one carried, so a long stretch of
  tool work with no new message still holds the original request
  (`tests/test_summary_prompt.py`).
- **The plan survives too.** With `plan_tool` set, the latest `{"todos": [...]}` result of
  that tool is appended after the summary as it was last written.
- **Injected text is described, not obeyed.** The summary prompt puts the history inside
  `<conversation_history>` tags and tells the summarizer not to follow any instruction in
  it. Only the first event (an earlier summary) is read for carried messages, so the same
  tags in a later reply carry nothing.
- **Fixed sections.** The prompt asks for these headings in order: Language, What the
  person wants, Instructions and preferences, Decisions, Work done, What was found, Current
  state, Next steps, Open questions.

### Old tool results (`MaskSettings`)

Applied by `OldResultsPlugin` before each model call.

- **No turn is dropped.** Only tool results are shortened, only in the request. The session
  keeps every result whole.
- **Nothing happens until the prompt is near the limit.** Masking starts once the last
  model call's prompt (as the provider counted it) reaches `trigger` times the compaction
  threshold. With no model call yet, nothing is masked.
- **The step the model is on is never masked.** Results after the last model message stay
  whole, then older ones while they fit within `protect` times the threshold.
- **Masked in batches, from a saved mark.** A batch is masked only if it saves at least
  `min_saving` times the threshold. The key of the last masked result is kept in session
  state under `agent_masked_until`, and every later request masks up to it the same way, so
  the cached prefix breaks once per batch, not on every call
  (`tests/test_prefix_watch.py`: ten results masked in three batches, three breaks).
- **Short results stay.** A result under 1,000 characters is never masked. A masked one keeps
  its first 200 characters and a note.
- **What the model can't read back isn't taken.** In a run that keeps conversation files
  (see [files](files.md)), each masked result is first saved whole as a file the note names.
  If any save in a batch fails, the batch is not masked. A run that keeps no files is masked
  with a note that says the rest wasn't kept.
- **Some tools are never masked.** Results of tools named in `never` always stay whole.

### Oversized results (applied by `ToolResultLimitPlugin`)

- A result over the limit keeps its start and its end (half the limit each), with a note in
  between giving its size and what was left out.
- An MCP result keeps its shape and its other keys (`isError` among them);
  `structuredContent` is dropped from a cut result. A picture, sound or blob item is
  replaced by a mark such as `[image/png, about 12,345 bytes, not shown]`.
- When the run keeps files, the whole text is saved and the note names the file and the
  reader tool to read the rest by `start` or `query`. When it can't be kept, the note says so.

### Window budget (`WindowSettings`, `window_of`)

Applied by `ContextBudgetPlugin` before each run.

- **The threshold follows the model.** The compaction threshold for a run is `fraction` of
  the turn model's window, never above `ceiling`. A `floor` lifts it on a small window, but
  never past 75% of the window (`FLOOR_SHARE`), which keeps room for the reply.
- **Unknown windows keep the fixed number.** A window of `None`, or `fraction` of 0, keeps the
  `token_threshold` from `CompactionSettings`.
- **Per run, never shared.** The plugin puts a copy of the app's compaction config on the
  invocation; the app's own config is not changed.
- `window_of(value)` accepts a positive number of tokens (an `int` or `float`, not a
  `bool`) and returns it as an `int`; anything else is `None`.

### Prompt cache (`prompt_cache`, `PromptCacheSettings`)

- `prompt_cache.prompt_cache(settings)` returns ADK's `ContextCacheConfig` with
  `min_tokens` and `ttl_seconds`, for the `App`'s `context_cache_config`. The harness adds
  no code at call time here. A cache hit needs a prompt prefix that stays the same from
  call to call, which `PrefixWatchPlugin` logs (see [plugins](plugins.md)).

### Context overflow

A request the provider refuses as too long is told apart by its error text (and LiteLLM's
`ContextWindowExceededError`). A rate limit that says "too many tokens, please wait" is never
treated as overflow (`tests/test_overflow.py`). The retry that follows is in
[routing](routing.md).

## What it costs

- **Compaction:** one extra model call per summary, on the summarizer model, recorded as one
  compaction operation in the ledger when a ledger writer is set. The carried block adds up
  to five messages of up to 2,000 characters each, plus the plan, to every summary.
- **Masking old results:** no model call. One conversation file saved per masked result in a
  run that keeps files (none for the reader tool's own results), and one session state key.
  Each masked batch makes the next call miss the provider's cache from the first masked
  result on.
- **Cutting oversized results:** one conversation file saved per cut result when the run keeps
  files (none for the reader tool's own results).
- **Window budget:** no model call, no storage; one log line per run.
- **Prompt cache:** nothing extra in the harness.

## Configure it

| Settings | Fields (no defaults unless shown) | Used by |
| --- | --- | --- |
| `CompactionSettings` | `interval`, `overlap`, `token_threshold`, `retention` | `events_compaction` |
| `ProviderReport` | `start` (called before a summary call), `take` (returns `cost_usd`, `routed_model`, `response_id`); both default to reading nothing | `events_compaction` |
| `MaskSettings` | `trigger`, `protect`, `min_saving` (shares of the compaction threshold) | `OldResultsPlugin` |
| `WindowSettings` | `fraction`, `ceiling`, `floor=0` | `ContextBudgetPlugin` |
| `PromptCacheSettings` | `min_tokens`, `ttl_seconds` | `prompt_cache.prompt_cache` |

`events_compaction(settings, summarizer_model, report=None, plan_tool=None)` takes the
summarizer as a function that builds a `BaseLlm`.

```python
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.models.lite_llm import LiteLlm

from harness.context import (
    CompactionSettings,
    MaskSettings,
    PromptCacheSettings,
    WindowSettings,
    events_compaction,
    prompt_cache,
    window_of,
)
from harness.plugins import ContextBudgetPlugin, OldResultsPlugin, ToolResultLimitPlugin
from harness.tools import plan_tools

MODEL = "provider/model-name"  # any LiteLLM model id
agent = LlmAgent(name="assistant", model=LiteLlm(model=MODEL), tools=[*plan_tools()])

app = App(
    name="assistant",
    root_agent=agent,
    events_compaction_config=events_compaction(
        CompactionSettings(interval=20, overlap=2, token_threshold=100_000, retention=10),
        summarizer_model=lambda: LiteLlm(model=MODEL),
        plan_tool="todo_list",
    ),
    context_cache_config=prompt_cache.prompt_cache(PromptCacheSettings(min_tokens=1024, ttl_seconds=300)),
    plugins=[
        # The product says where a turn carries its model's window.
        ContextBudgetPlugin(
            lambda state: window_of(state.get("model_window")),
            WindowSettings(fraction=0.5, ceiling=150_000, floor=24_000),
        ),
        OldResultsPlugin(
            MaskSettings(trigger=0.6, protect=0.2, min_saving=0.1),
            reader="read_document",  # the product's tool that reads a saved copy
            fallback_threshold=100_000,
        ),
        ToolResultLimitPlugin(max_chars=80_000, reader="read_document"),
    ],
)
```

`OldResultsPlugin` reads the run's threshold from the invocation's compaction config and
uses `fallback_threshold` when compaction is off. Plugin order rules are in
[plugins](plugins.md).

## Test code that uses it

- **A scripted model for the agent and one for the summarizer.** `tests/test_summary_prompt.py`
  runs a real `InMemoryRunner` with an `App` built by `events_compaction`, a `BaseLlm`
  subclass that reports a large `prompt_token_count` so compaction fires, and a second one
  as `summarizer_model` (the class itself is the builder). It then reads the summary text
  out of the requests the agent model was sent.
- **Masking through ADK.** `tests/test_prefix_watch.py` runs `OldResultsPlugin` in an
  `InMemoryRunner` with a scripted model whose `usage_metadata.prompt_token_count` grows
  with the request, and a small `fallback_threshold`, so masking starts within a few calls.
- **No file store needed.** With no files rule set (see [files](files.md)), no run keeps
  files: masking and cutting still happen, with a note that the rest wasn't kept, and no
  artifact service is touched.
- **No ledger needed.** With no ledger writer set, the summarizer's metering records nothing.
  `tests/test_summary_prompt.py` also stubs `meter.finish`.
- **Settings are plain frozen dataclasses**, passed in, so two configurations can sit side by
  side in one test process.

## Reference

| Name | What it is |
| --- | --- |
| `CompactionSettings` | When history is rolled up: interval, overlap, token threshold, retention. |
| `events_compaction` | ADK's `EventsCompactionConfig` with a metered summarizer that carries the person's messages and the plan. |
| `ProviderReport` | How to read the provider's cost, routed model and response id for a summary call. |
| `MaskSettings` | When and how much of older tool results `OldResultsPlugin` masks. |
| `prompt_cache` | Module; `prompt_cache.prompt_cache(settings)` returns ADK's `ContextCacheConfig`. |
| `PromptCacheSettings` | The cache's `min_tokens` and `ttl_seconds`. |
| `window_of` | Reads a window size from a turn's state value: a positive token count, else `None`. |
| `WindowSettings` | The compaction threshold as a share of the model's window, with a ceiling and a floor. |
