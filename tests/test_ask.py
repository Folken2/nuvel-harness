"""ask_person: a question pauses the run, and the answer resumes the same one."""

import asyncio
import time
from collections.abc import AsyncGenerator

from google.adk.agents import LlmAgent
from google.adk.apps import App, ResumabilityConfig
from google.adk.events import Event
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from harness.runs.resume import past_frames, read_history
from harness.tools.ask import TOOL_NAME, ask_tools

APP = "probe"
USER = "alice"
SESSION = "s1"
USAGE = types.GenerateContentResponseUsageMetadata(
    prompt_token_count=10, candidates_token_count=1
)

OPTIONS = [
    {"label": "1 July, the letter", "consequence": "38 invoices over", "recommended": True},
    {"label": "15 June, the contract", "consequence": "44 invoices over", "recommended": False},
]


def asks(question: str = "Which date?", options=None, call_id: str = "q1") -> types.Part:
    return types.Part(
        function_call=types.FunctionCall(
            id=call_id,
            name=TOOL_NAME,
            args={"question": question, "options": OPTIONS if options is None else options},
        )
    )


class AskingLlm(BaseLlm):
    """Asks with `first` until a result comes back, then says what it got."""

    model: str = "scripted"
    first: list = []
    seen: list = []

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        results = [
            part.function_response.response
            for content in llm_request.contents
            for part in content.parts or []
            if part.function_response
        ]
        self.seen.append(results)
        parts = self.first if not results else [types.Part(text=f"got {results[-1]}")]
        yield LlmResponse(content=types.Content(role="model", parts=parts), usage_metadata=USAGE)


def build(first: list[types.Part]):
    llm = AskingLlm(first=first, seen=[])
    agent = LlmAgent(name=APP, model=llm, tools=ask_tools())
    app = App(name=APP, root_agent=agent, resumability_config=ResumabilityConfig(is_resumable=True))
    sessions = InMemorySessionService()
    return Runner(app=app, session_service=sessions), sessions, llm


async def run(runner: Runner, message: types.Content) -> list[Event]:
    return [
        event
        async for event in runner.run_async(
            user_id=USER,
            session_id=SESSION,
            new_message=message,
            # The web app sends one with every turn, an answer included.
            state_delta={"temp:test_model": "fresh"},
        )
    ]


def text(words: str) -> types.Content:
    return types.Content(role="user", parts=[types.Part(text=words)])


def answer(response: dict, call_id: str = "q1") -> types.Content:
    return types.Content(
        role="user",
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    id=call_id, name=TOOL_NAME, response=response
                )
            )
        ],
    )


async def stored(sessions) -> list[Event]:
    session = await sessions.get_session(app_name=APP, user_id=USER, session_id=SESSION)
    return session.events


def function_responses(events: list[Event]) -> list[dict]:
    return [r.response for event in events for r in event.get_function_responses()]


def test_a_question_pauses_the_run_with_the_call_unanswered():
    async def scenario():
        runner, sessions, _ = build([asks()])
        await sessions.create_session(app_name=APP, user_id=USER, session_id=SESSION)
        events = await run(runner, text("go"))
        [asked] = [event for event in events if event.get_function_calls()]
        assert asked.long_running_tool_ids == {"q1"}
        assert function_responses(events) == []
        # Paused, not ended: an ended run can't be resumed.
        assert not any(event.actions.end_of_agent for event in events)
        # A restart now has nothing to run: only the answer goes on from here.
        assert read_history(await stored(sessions), 0).ended

    asyncio.run(scenario())


def test_the_answer_resumes_the_same_run_and_the_model_reads_it():
    async def scenario():
        runner, sessions, llm = build([asks()])
        await sessions.create_session(app_name=APP, user_id=USER, session_id=SESSION)
        first = await run(runner, text("go"))
        reply = {"answer": "15 June, the contract", "option": "15 June, the contract"}
        second = await run(runner, answer(reply))

        assert llm.seen[-1] == [reply]
        assert {event.invocation_id for event in first + second} == {first[0].invocation_id}
        assert any(event.content and "got" in str(event.content) for event in second)
        assert any(event.actions.end_of_agent for event in second)

    asyncio.run(scenario())


def test_a_question_that_breaks_the_rules_is_answered_with_what_to_fix():
    cases = [
        ([], "Give 2 to 4 options"),
        ([OPTIONS[0], {**OPTIONS[1], "recommended": True}], "at most one option as recommended"),
        ([OPTIONS[0], {**OPTIONS[1], "consequence": " "}], "both a label and a consequence"),
        ([OPTIONS[0], {**OPTIONS[1], "label": OPTIONS[0]["label"]}], "same label"),
    ]
    for options, words in cases:

        async def scenario(options=options, words=words):
            runner, sessions, _ = build([asks(options=options)])
            await sessions.create_session(app_name=APP, user_id=USER, session_id=SESSION)
            events = await run(runner, text("go"))
            [result] = function_responses(events)
            assert words in result["error"]
            # The model carried on. ADK leaves out its end marker here, so the resume
            # check reads the model's last word instead.
            assert "got" in str(events[-1].content)
            assert read_history(await stored(sessions), 0).ended

        asyncio.run(scenario())


def test_the_schema_requires_what_the_tool_refuses_without():
    # The tool refuses an option without `recommended` (exactly one must be true), so
    # the schema the model fills must not offer it as optional with a default: a model
    # that trusts the schema gets refused and asks again.
    [tool] = ask_tools()
    schema = tool._get_declaration().parameters_json_schema
    option = schema["$defs"]["AskOption"]

    assert set(option["required"]) == {"label", "consequence", "recommended"}
    assert "default" not in option["properties"]["recommended"]


def test_a_question_may_recommend_nothing_when_its_the_persons_own_choice():
    # A pure preference ("Monday or Friday?") gives the agent no basis to recommend one;
    # requiring a pick made it either invent one or be refused and ask again.
    async def scenario():
        none = [{**option, "recommended": False} for option in OPTIONS]
        runner, sessions, _ = build([asks(options=none)])
        await sessions.create_session(app_name=APP, user_id=USER, session_id=SESSION)
        events = await run(runner, text("go"))
        # Asked: the call stays unanswered and the run pauses, as any valid question does.
        assert function_responses(events) == []
        assert not any(event.actions.end_of_agent for event in events)

    asyncio.run(scenario())


def test_two_questions_at_once_get_one_asked():
    async def scenario():
        runner, sessions, _ = build([asks(call_id="q1"), asks("Which team?", call_id="q2")])
        await sessions.create_session(app_name=APP, user_id=USER, session_id=SESSION)
        events = await run(runner, text("go"))
        [refused] = function_responses(events)
        assert "one at a time" in refused["error"]

    asyncio.run(scenario())


def test_after_a_restart_the_answers_run_is_what_came_after_the_answer():
    async def scenario():
        runner, sessions, _ = build([asks()])
        await sessions.create_session(app_name=APP, user_id=USER, session_id=SESSION)
        await run(runner, text("go"))
        since = time.time()
        await run(runner, answer({"answer": "B", "option": None}))
        session = await sessions.get_session(app_name=APP, user_id=USER, session_id=SESSION)

        history = read_history(session.events, since)
        assert history is not None
        # The question was the run before; the answer itself isn't a step.
        assert not any(event.get_function_calls() for event in history.events)
        assert history.steps == 1
        assert history.ended
        frames = past_frames(history)
        assert all("functionResponse" not in frame for frame in frames)
        assert any("got" in frame for frame in frames)

    asyncio.run(scenario())
