# Approval

Some tool calls have effects a person should agree to first: sending an email, deleting a
record, paying. The approval package decides, before a tool runs, whether the call may run or
the run should pause and ask the person. A product builds an `ApprovalGate` from
`harness.approval` and attaches it with `ApprovalPlugin` from `harness.plugins`.

The gate can ask two questions about a call, each answered by a classifier:

- **Effects:** does the call change something? Always asked when the policy leaves the call
  to the classifier.
- **Authorised:** did the person ask for this call, in their own messages? Asked only when the
  product sets `authorisation_threshold`. It only ever adds a question, never removes one.

## What it guarantees

- **The pause is ADK's tool confirmation.** `ApprovalGate` calls `request_confirmation` on
  the call; the person's answer comes back as `tool_confirmation`. An approved call runs once.
  A denied call is answered as denied, which the model reads as the person's decision, not an
  error. A call ADK brings back after an answer is never classified again, so the person is
  never asked twice for one call.
- **Layered decisions** (`Policy`). Mode `off` never asks. A call whose actions the person has
  all allowed (`ApprovalSettings.allowed`) runs. Otherwise the agent's own read-only tools,
  and tools their app marks read only, never ask. Actions that send, delete or pay always ask.
  Everything else goes to the classifier in mode `smart`, or asks in mode `strict`.
- **A classifier that fails or times out counts as risky**: the call asks. A call too long to
  rate whole (over 24,000 characters of arguments) asks without being rated in part.
- **Outside content can't talk a call past the gate.** With `authorisation_threshold` set, a
  call the effects question rates safe still asks when the run has read anything from outside
  the conversation since the person last wrote (`read_outside_since_person`) and the
  authorised question comes back below the threshold (reason `NOT_ASKED_FOR_REASON`), or fails.
  Only the person's own messages count as their word: never the call's arguments, a page, an
  email, a document the agent read, a file they attached, or a note the agent wrote into their
  turn (`person_messages`). Tools named in `inside` are ones whose answers carry no third-party
  text, so reading them doesn't count as outside.
- **A card says whether the person asked for it.** With `authorisation_threshold` set, every
  card's payload carries `authorised` (the probability, or None when unknown) and `asked_for`
  (whether it reached the threshold, or None).
- **It stops asking after repeated refusals.** With `max_denials` above 0, that many refusals
  in a row within one message stop the questions for the rest of the turn, and the model is
  told to stop calling tools and talk to the person. The default, 0, never stops. An answer
  in the person's own words (a `said` message instead of Deny) is not a refusal: it neither
  adds to the row nor ends it, and the model receives their words as `person_said`.
- **It holds for every tool**, the person's connected apps included, because it is a plugin
  rather than an agent callback.

## What it costs

- A call the policy settles costs nothing extra: the rules are pure and make no network call.
- A call the policy leaves to the classifier costs one classifier call before the tool runs.
- With `authorisation_threshold` set: one more classifier call for a call rated safe after
  outside content was read, and one for each card (unless that call was already rated, or is
  too long to rate). The person's messages sent with it are the newest that fit, each cut to
  2,000 characters and 8,000 in all.
- A call that asks pauses the run until the person answers. Run the app with
  `ResumabilityConfig(is_resumable=True)` so the answer resumes the same run.

## Configure it

`ApprovalSettings(mode="smart", allowed=frozenset())` is the default for a turn; `MODES` lists
the three modes. An `ApprovalGate` takes everything that is the product's, as keyword
arguments:

| Argument | What it is |
| --- | --- |
| `threshold` | At or above this probability of effects, the call asks. |
| `reader` | An `ActionReader`: how a call reads as actions, with any `Router`s. |
| `policy` | A `Policy`: the read-only tools by source, the sources whose hints are trusted, and optional second checks. |
| `classifier_for` | `ClassifierFor`: a key to an ADK `BaseClassifier`. |
| `hints_for` | Async: a call's actions to what each one's app says it does. |
| `api_key_for` | The turn's state to the key the classifier call goes on. |
| `settings_of` | The turn's state to its `ApprovalSettings`. |
| `max_denials` | Refusals in a row before the gate stops asking. Default 0, never. |
| `asks_also` | Optional: the product's own reason to ask for a call the policy would run. |
| `authorisation_threshold` | Optional: turns on the authorised question. Default None. |
| `inside` | Tools whose answers carry no third-party text. Default none. |

```python
from harness.approval import ActionReader, ApprovalGate, ApprovalSettings, Policy
from harness.plugins import ApprovalPlugin


async def no_hints(actions):
    return {}


gate = ApprovalGate(
    threshold=0.5,
    reader=ActionReader(),
    policy=Policy(read_only={}, trusted_hints=frozenset()),
    classifier_for=lambda key: my_classifier,  # an ADK BaseClassifier
    hints_for=no_hints,
    api_key_for=lambda state: "",
    settings_of=lambda state: ApprovalSettings(),
    max_denials=3,
    authorisation_threshold=0.6,
    inside=("lookup",),
)
plugin = ApprovalPlugin(gate)
```

## Test code that uses it

`Policy` and the breaker are pure, so their rules can be tested on plain values, and so can
`person_messages` and `authorisation_state`. For a whole run, give the gate a fake
`BaseClassifier` that answers each question by its name (`"effects"`, `"authorised"`), drive
the run with a scripted model (see
[build-an-agent.md](build-an-agent.md#5-test-it-with-a-scripted-model)) under ADK's `Runner`
and `InMemorySessionService`, and answer the `adk_request_confirmation` call the way a person
would. `tests/test_approval_authorisation.py` does exactly this.

## Reference

| Name | What it is |
| --- | --- |
| `ApprovalGate` | Decides each call: run, or pause and ask. |
| `ApprovalSettings`, `MODES` | A turn's mode (`smart`, `strict`, `off`) and the actions the person always allows. |
| `Policy`, `Verdict` | The layered rules, and their answer (`run`, `ask` with a reason, or `classify`). |
| `Action`, `ActionReader`, `Router`, `AGENT`, `own_name` | A call as the actions it takes, each with the key a person can allow. |
| `ClassifierFor` | The type of `classifier_for`. |
| `AUTHORISED` | The authorised question the classifier is asked. |
| `person_messages`, `authorisation_state` | The person's own messages, and what the classifier rates for that question. |
| `read_outside_since_person` | Whether the run read outside content since the person last wrote. |
| `CHANGES_REASON`, `REACHES_OUT_REASON`, `NOT_ASKED_FOR_REASON` | Reasons a card can give. |
| `WRITES`, `words` | The verbs that read as writing, and how a tool name is split into words. |
| `code_check` | Whether code the agent wrote only computes, read from its syntax tree. |
