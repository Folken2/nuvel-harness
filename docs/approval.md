# Approval

> **This page is short on purpose: approval is being reworked.** The names below are the
> current public API, but their shape is expected to change before the first tagged release.
> Read `src/harness/approval/` for the details, and check this page again after that release.

Some tool calls have effects a person should agree to first: sending an email, deleting a
record, paying. The approval package decides, before a tool runs, whether the call may run or
the run should pause and ask the person. A product imports it from `harness.approval` and
attaches it with `ApprovalPlugin` from `harness.plugins`.

## What it guarantees

- **The pause is ADK's tool confirmation.** `ApprovalGate` calls `request_confirmation` on
  the call; the person's answer comes back as `tool_confirmation`. An approved call runs once.
  A denied call is answered as denied, which the model reads as the person's decision, not an
  error. A call ADK brings back after an answer is never classified again, so the person is
  never asked twice for one call.
- **Layered decisions** (`Policy`). The agent's own read-only tools, and tools their app marks
  read only, never ask. Actions that send, delete or pay always ask. Everything else goes to a
  classifier in mode `smart`, or asks in mode `strict`. Mode `off` never asks.
- **A classifier that fails or times out counts as risky**: the call asks.
- **"Always allow"** (`ApprovalSettings.allowed`) skips the question for a call whose actions
  the person has all allowed.
- **It stops asking after repeated refusals.** With `max_denials` above 0, that many refusals
  in a row within one message stop the questions for the rest of the turn, and the model is
  told to stop calling tools and talk to the person. The default, 0, never stops.
- **It holds for every tool**, the person's connected apps included, because it is a plugin
  rather than an agent callback.

## What it costs

- A call the policy settles costs nothing extra: the rules are pure and make no network call.
- A call that goes to the classifier costs one classifier call before the tool runs.
- A call that asks pauses the run until the person answers. Run the app with
  `ResumabilityConfig(is_resumable=True)` so the answer resumes the same run.

## Configure it

`ApprovalSettings(mode="smart", allowed=frozenset())` is the default for a turn; `MODES` lists
the three modes. An `ApprovalGate` is built with everything that is the product's: how a call
reads as actions (`ActionReader`, `Router`), the `Policy`, the classifier (`ClassifierFor`), a
`threshold`, where an action's hints come from, whose key the classifier call uses, and how a
turn's settings are read. Hand the gate to `ApprovalPlugin(gate)`.

Because these signatures are changing, this page shows no snippet yet.

## Test code that uses it

`Policy` and the breaker are pure, so their rules can be tested on plain values. For a whole
run, pass a classifier that returns a fixed verdict, drive the run with a scripted model (see
[build-an-agent.md](build-an-agent.md#5-test-it-with-a-scripted-model)), and answer the
`adk_request_confirmation` call the way a person would.

## Reference

`Action`, `ActionReader`, `AGENT`, `ApprovalGate`, `ApprovalSettings`, `CHANGES_REASON`,
`ClassifierFor`, `code_check`, `MODES`, `own_name`, `Policy`, `REACHES_OUT_REASON`,
`read_outside_since_person`, `Router`, `Verdict`, `words`, `WRITES`. Each is documented in its
module under `src/harness/approval/`.
