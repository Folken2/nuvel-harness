# Usage ledger and billing

A product that runs an ADK agent pays for every request the agent makes: model calls, the
retries and fallbacks inside them, compaction summaries, tool calls and hosted web searches.
The ledger writes one row per request, before it goes out and again when it ends, so a crash,
a cut-off stream or a refused key still leaves a priced record. The billing package holds the
pricing rules the ledger uses, with no I/O.

A product imports `from harness.ledger import connect, sweeping, LedgerContext, RunStart, ...`
and `from harness.billing import Rates, FallbackRates, Units, totals, UNKNOWN`. The ADK side is
two plugins, `LedgerPlugin` and `ToolLedgerPlugin`, described in [plugins](plugins.md). The
model client that opens each attempt is `MeteredLiteLLMClient`; the routing client in
[routing](routing.md) builds on it.

## Where the rows go

The harness owns no database and defines no table. `connect(store, catalogue)` hands it two
objects the product writes:

- a **store**, with `start`, `settle`, `stale_operations`, `close_stale_interactions`,
  `open_background`, `close_background` and `units_used` (the `Store` protocol in
  `src/harness/ledger/stores.py`);
- a **catalogue** of prices, with `rates`, `fallback_rates`, `web_search_rate` and
  `highest_web_search_rate` (the `Prices` protocol there).

Each write is a plain `dict` whose keys are meant to be the product's column names. Rows are
operations under an interaction: the interaction is the turn (or a background task), opened by
whoever starts it, and named by `LedgerContext.interaction_id`.

**A started operation** (`store.start(values)`) carries:

| Keys | Value |
| --- | --- |
| `level` | `"operation"` |
| `root_id`, `parent_id` | the interaction id |
| `run_id` | the ADK invocation id |
| `idempotency_key` | `"<run_id>:<kind>:<index>:<attempt>"` (see below) |
| `status` | `"started"` |
| `event_type`, `operation_kind`, `feature`, `use_case` | from the kind the product registered |
| `is_ai` | `True` for model calls; for a tool, whether it was registered as calling a model |
| `quota_scope`, `user_id`, `actor_name`, `actor_email`, `team_id`, `team_name`, `conversation_id`, `message_id`, `requested_model_id`, `requested_model_name` | from `LedgerContext` |
| `provider`, `model_name` | the model asked for, normalised by `ModelIds` |
| `details` | `provider_contacted: true`, the token `estimate` used if the call is cut off, `teams` when several |

A tool's started row adds `tool_call_id`, `tool_name`, `tool_source` (`internal`, `mcp`,
`connected_app` or `provider_native`), `tool_namespace`, `toolkit_slug`, `tool_display_name`,
`billing_sku` and `billing_subject` (`"user:<id>"`, or `"platform"` when the quota scope is not
the user). It never stores the tool's arguments or result.

**A settlement** (`store.settle(key, values)`) carries `status` (`completed`, `failed` or
`cancelled`), `provider` and `model_name` (the model that served it, when the provider said),
`tokens_in`, `tokens_out`, `reasoning_tokens`, `cost_usd`, `cost_source`, `duration_ms` and
`details`. A tool's settlement carries only `status`, `cost_usd`, `cost_source`,
`duration_ms` and `details`. `details` holds, as they apply: `cost_accounting` (cost quality,
source, the four-way split, the per-bucket token cost and a snapshot of the rates used),
`usage` (every token bucket), `provider_usage` (the provider's raw counters, filtered),
`provider_response_id`, `router`, `model_not_reported`, `billing_units`, `output_count`,
`list_cost_usd`, `error_code` and `error_message`.

`store.settle` must settle a row at most once and return its stored cost, or `None` when there
was nothing to settle. The stale sweep relies on this: when a run and the sweep settle the same
row, whichever lands first wins and the other changes nothing.

## What it guarantees

- **Nothing connected, nothing recorded.** `connect(None)` (or never calling it) turns every
  meter into a no-op. `LedgerPlugin` then doesn't read the turn's state at all, and a turn whose
  state names no interaction or no run (`context_from_state` returns `None`) is not recorded.
- **One row per provider request.** `MeteredLiteLLMClient.acompletion` writes the started row
  right before each request and settles it as failed when the request raises. A retry or a
  fallback to another model is the same call's next attempt (`attempt` goes up by one), so a
  call that took three requests has three rows.
- **A streamed request that fails on its first chunk is still that attempt.** The client reads
  the first chunk before returning (`first_chunk.primed`), so a refusal that only shows when a
  lazy stream is first read settles the row and reaches the retry loop. Only the first chunk is
  read early.
- **Replays write the same key.** The key is `<run_id>:<kind>:<index>:<attempt>`. On a resume,
  `LedgerPlugin` starts the index after the calls the session already holds for that run (a
  finished, non-partial event with usage). Calls keyed by a tool call id (`meter.call_key`,
  used by `direct_call`) get a `~2`, `~3` suffix when the same id comes back in one process. The
  key is reused; making a repeated `start` a no-op is the store's job.
- **Recording never stops a run.** Each write runs in its own task. A write that raises is
  tried again after 1, 2 and 4 seconds (four tries in all), then logged with the run id and
  dropped. A failure opening a background task's interaction (`open_task_state`) is logged and
  the task runs unmetered.
- **A settlement lands after its start.** It waits for its own start task, and is skipped when
  `start` returned `False` (the interaction had closed) or never landed.
- **Final totals at the end of the run.** `LedgerPlugin.after_run_callback` waits for the run's
  pending writes, at most 15 seconds (`DRAIN_TIMEOUT_SECONDS`), and logs any still pending. It
  never raises. Compaction after the run waits for its own writes.
- **How a failed attempt is priced** (`meter.attempt_failed`):
  - a request LiteLLM refused before sending (`UnsupportedParamsError`): cost 0, source
    `free`, `provider_contacted: false`;
  - an error with an HTTP status from 400 to 599: cost 0, source `free`;
  - anything else (a dropped stream, a timeout with no status): charged its estimate, source
    `estimated_on_abort`. The estimate is the last prompt size the run reported, else 16,000
    input tokens, plus 4,096 output tokens;
  - a cancelled request settles as `cancelled`, any other as `failed`.
  - `error_code` is `RATE_LIMIT` (429), `AUTH` (401, 403), `TIMEOUT`, or the exception's type
    name in upper case. `error_message` is the exception's type name, never its text.
- **A tool result that says it failed is a failed row.** MCP's `isError: true`,
  `status: "error"`, or a non-empty `error` string settle the tool row as `failed` with
  `TOOL_ERROR`. A tool that wasn't found settles with `UNKNOWN_TOOL`.
- **Hosted web searches get their own row.** When a model call's response metadata names a
  count of searches, `meter.finish` writes one tool row for them, keyed by that call. If the
  call's cost came from the provider, the search row costs 0 and records how much was included.
  Otherwise it is priced per search at the catalogue's rate for that model, or, with no rate,
  at the catalogue's highest search rate (else $0.05) and flagged as missing a price.
- **Abandoned work gets closed.** `sweeping(ceiling_seconds)` runs `sweep_forever` beside the
  service. Every 600 seconds it reads up to 500 operations still `started` from before
  `now - (ceiling_seconds + 3600 s)`, settles each as `failed` with error code `ABANDONED`, and
  then closes interactions left open from before that cutoff. Model-kind rows
  (`INVOCATION_KINDS`: model, image, embedding, compaction, title) are charged the estimate
  stored when they started, at catalogue rates. Tool rows get cost 0, source `unknown` and
  `pricing_missing`, so they read as unresolved, never free. One bad row is logged and the rest
  go on; a failed pass is logged and the next one tries again.
- **No secrets or content in the row.** `provider_usage` keeps only numeric counters under
  usage-like key names and a few labels (`TEXT`, `USD` and the like), at most 500 values, 6
  levels deep, 40 items per list and 80-character keys (`billing/evidence.py`).
- **A missing price is never free** (`billing/pricing.py`):
  - a cost the provider reported wins, 0 included, and nothing is added on top;
  - a missing input or output rate is replaced by the fallback rate (by default $15 and $60 per
    million tokens, or the highest the catalogue lists, `FallbackRates.from_catalogue`), and the
    source becomes `fallback_pricing`;
  - a missing cache or modality rate falls back to the input or output rate and is flagged in
    `missing_pricing`;
  - rates not in USD, a non-standard service tier the provider reported, a 12-hour cache write
    and a unit price with nothing measured are all flagged;
  - only when every rate the call needed is known and exactly 0 is the source `free`.
- **Exact amounts.** Every amount is a `Decimal` rounded half up to 12 places; negatives store
  as 0. The rates used are copied onto the row, so a later price change can't rewrite it.
- **One answer for an interaction's total.** `totals(entries)` sums settled operations into a
  cost, a source (`mixed` when more than one paid source took part, `fallback_pricing` or
  `unknown` when one is unresolved), a split into provider reported, calculated, estimated and
  unresolved parts that adds up to the cost, and a count of unresolved entries.
  `settled_entry(row)` reads a stored row into what `totals` takes.

## What it costs

- **No extra model calls.** The ledger only records the calls the run makes.
- **Store writes:** one `start` and one `settle` per provider attempt, per tool call that runs,
  and per model call that ran hosted searches. One `open_background` per background task. A
  failing write can be retried three times, with 7 seconds of waiting in all.
- **Catalogue reads at settlement:** `fallback_rates()` for every model, compaction or image
  settlement, and `rates(model)` unless it is a turn model call whose rates came with the turn
  (`RunStart.rates`). A hosted search settlement reads `web_search_rate`, and
  `highest_web_search_rate` when there is no rate. `units_used` is read only for a tool price
  rule whose cost depends on earlier usage; the harness attaches no rule to the agent's own tools.
- **Latency:** the started row is built in the request path (a dict, no I/O). Writes run in
  background tasks. The run's end waits for pending writes, up to 15 seconds.
- **Memory:** one `RunLedger` per running run, holding the call under way, open tool calls,
  the keys used, and its write tasks until the run drains.
- **The sweep:** one pass every 600 seconds: one `stale_operations` read (up to 500 rows), one
  `fallback_rates`, one `rates` and one `settle` per stale row, and one
  `close_stale_interactions`.

## Configure it

At start-up, before any run:

1. Register a `Kind` for every operation kind the product records: `MODEL` always, `COMPACTION`
   with compaction, `IMAGE` for image calls, `TOOL` with `ToolLedgerPlugin`, and any kind passed
   to `direct_call`. An unregistered kind raises `KeyError` when its row is built, which is in
   the request path: the model call or tool callback fails.
2. Optionally say how model ids read with `REGISTRY.set_model_ids(ModelIds(...))`. Its four
   callables (`router_of`, `without_router`, `maker_of`, `through_router`) default to "no
   router, the prefix is the maker".
3. Call `connect(store, catalogue)`, and run `sweeping` beside the service.
4. Add `LedgerPlugin(read_start)` and `ToolLedgerPlugin()` to the app, in the order
   [plugins](plugins.md) gives.

`LedgerContext` fields: `interaction_id`, `run_id` (both required for a row), and `user_id`,
`actor_name`, `actor_email`, `team_id`, `team_name`, `conversation_id`, `message_id`,
`requested_model_id`, `requested_model_name` (default `None`), `teams` (default `()`),
`quota_scope` (default `"user"`), `model_feature` (default `None`: the kind's own feature).

`RunStart` fields: `context`, `rates` (the turn model's rates; `None` reads the catalogue),
`requested_model` (the model id asked for), `list_rates` (the model's own rates when someone
other than the platform pays: kept as `list_cost_usd`, never added to the cost).

`Rates.from_mapping` reads camel-case snapshot names, per million (`inputPer1M`) or per token
(`inputPerToken`); per million wins. Rate names: `input`, `output`, `cacheRead`, `cacheWrite`,
`cacheWrite1h`, `inputText`, `inputImage`, `inputAudio`, `outputText`, `outputImage`,
`outputAudio`, plus `currency` (default `USD`), `unit` (`image`, `megapixel`, `second` or
`request`) and `unitPrice`.

```python
from decimal import Decimal

from harness import ledger
from harness.billing import FallbackRates, Rates
from harness.ledger import COMPACTION, MODEL, REGISTRY, TOOL, Kind, LedgerContext, RunStart
from harness.plugins import LedgerPlugin, ToolLedgerPlugin

for name in (MODEL, COMPACTION, TOOL):
    REGISTRY.add_kind(name, Kind(event_type=f"agent.{name}", operation_kind=name,
                                 feature=name, use_case="assistant_response"))


def read_start(state):
    """The turn's state, in this product's shape, or None when the turn isn't recorded."""
    turn = state.get("usage")
    if not turn:
        return None
    return RunStart(
        context=LedgerContext(interaction_id=turn["interaction"], run_id=turn["run"],
                              user_id=turn.get("user")),
        rates=Rates.from_mapping(turn["rates"]),
        requested_model=turn["model"],
    )


async def lifespan(app):
    ledger.connect(MyStore(pool), MyPrices(pool))  # the product's Store and Prices
    async with ledger.sweeping(ceiling_seconds=3600):
        yield

plugins = [LedgerPlugin(read_start), ToolLedgerPlugin()]
```

**A tool that calls a provider itself** (an endpoint the model client can't reach) opens its
own operation:

```python
from google.adk.models import LlmResponse
from google.genai import types

from harness.ledger import MODEL, meter
from harness.ledger.direct_call import direct_call

async with direct_call(MODEL, call_id=tool_context.function_call_id, model="provider/model"):
    answer = await my_client.generate(prompt)  # the tool's own provider call
    meter.finish(LlmResponse(usage_metadata=types.GenerateContentResponseUsageMetadata(
        prompt_token_count=answer.input_tokens, candidates_token_count=answer.output_tokens)))
```

Whatever escapes the block settles the row as failed (a cancel as cancelled). Two such tools in
one response take turns (`one_call_at_a_time`), because a run holds one open call. The harness
itself has no caller of `direct_call` and no test of it.

**Background tasks**: `open_task_state(parent_state, task_id, parser=..., as_task=...)` opens
the task's interaction through `store.open_background` and returns the task's ledger state, or
`None` when the parent's turn isn't recorded. The interaction's key is `"<task_id>:agent.task"`,
so a resumed task finds it. Closing it is the product's.

**Overage**: `overage.overage(before, cost, limit)` is the part of one call's cost above a month
limit: 0 with no limit or while under it, all of it once the month is already past it. It is a
pure function; the harness does not call it, and it never stops or charges anything.

## Test code that uses it

- **An in-memory store.** The protocols are small: a class with the seven `Store` methods
  keeping rows in a dict, keyed by `idempotency_key`, and settling only rows still `started`,
  is enough. Pair it with a catalogue whose `rates` returns `None` and whose `fallback_rates`
  returns `FallbackRates()`, or fixed `Rates`. Call `ledger.connect(store, prices)` in the test
  and `ledger.connect(None)` after it: the writer is process-wide.
- **The writer alone.** `tests/test_usage_frames.py` builds a `LedgerWriter` over a fake store
  with only `settle`, a `RunLedger` by hand, and sets `writer.RETRY_DELAYS` to zeros with
  `monkeypatch` so retry tests don't sleep.
- **Pricing with no ledger.** `tests/test_ledger_cost.py` tests every pricing rule as pure
  functions (`price_call`, `price_aborted`, `normalise`, `totals`) with exact `Decimal`s. A
  product's own pricing tests can do the same.
- **Overage**: `tests/test_overage.py`.
- **A whole run with a scripted model.** Run the agent under ADK's `InMemoryRunner` with a
  scripted model (as in `examples/test_assistant.py`), put the ledger state in the session's
  state, and read the store's rows after the run. A scripted model goes through no
  `MeteredLiteLLMClient`, so no attempt is opened and `meter.finish` writes nothing: only the
  tool rows from `ToolLedgerPlugin` appear. To test model rows, drive the meter directly
  (`meter.new_call`, `meter.begin_attempt`, `meter.finish` or `meter.attempt_failed`, then
  `await meter.drain()`) inside a hand-built run ledger, as `LedgerPlugin` does. To skip the
  ledger in a run test, `tests/test_summary_prompt.py` replaces `meter.finish` with a no-op.

## Reference

`harness.ledger`:

| Name | What it is |
| --- | --- |
| `connect(store, catalogue=None)` | Where rows go and prices come from; `None` turns recording off. |
| `sweeping(ceiling_seconds)` | Async context manager running the stale-work sweep; nothing without a store. |
| `LedgerContext` | The interaction and run a call goes under, and who caused it. |
| `RunStart` | What a run's ledger state starts from: context, rates, requested model, list rates. |
| `context_from_state(raw, parser)` | The turn's `LedgerContext` read by the product's parser, or `None`. |
| `MeteredLiteLLMClient(on_whole_answer=None)` | ADK's LiteLLM client with one ledger attempt per request. `on_whole_answer` sees each non-streamed response. |
| `meter` | The model meter module: `metering_as`, `call_key`, `finish`, `attempt_failed`, `metering_turn_call`. |
| `direct_call` | Module with `direct_call(kind, call_id=, model=)`, a tool's own provider call as one operation. |
| `one_call_at_a_time()` | Holds the run's one open non-turn call; nothing outside a metered run. |
| `open_task_state(parent, task_id, parser=, as_task=)` | Opens a background task's interaction; its ledger state, or `None`. |
| `overage` | Module with `overage(before, cost, limit)`. |
| `settled_entry(row)` | A stored row as the input `billing.totals` takes. |
| `RequestedModel(id, name)` | The catalogue model a call asked for, when it isn't the turn's. |
| `Kind(event_type, operation_kind, feature, use_case)` | What a row of one operation kind carries. |
| `LedgerRegistry` | Kinds, tool features, connected-app servers and model ids; `add_kind`, `add_tool`, `add_app_server`, `set_model_ids`. |
| `REGISTRY` | The registry the ledger writes with. |
| `ModelIds` | How model ids name a router and a maker. |
| `MODEL`, `COMPACTION`, `IMAGE`, `TOOL` | The ledger's own kind names. |
| `INVOCATION_KINDS` | Kinds the sweep charges an estimate for: model, image, embedding, compaction, title. |

`harness.billing`:

| Name | What it is |
| --- | --- |
| `Rates` | One model's prices; `None` is unknown. `from_mapping`, `snapshot`. |
| `FallbackRates(input=15, output=60)` | Rates used where a base rate is missing; `from_catalogue`. |
| `Units(images, image_sizes, sizes_from_request, seconds)` | Measured quantities for a unit-priced call, passed to `meter.finish(units=...)`. |
| `totals(entries)` | An interaction's cost, source, split, `pricing_missing` and unresolved count. |
| `UNKNOWN` | The cost source `"unknown"`: contacted, with nothing to price. |
