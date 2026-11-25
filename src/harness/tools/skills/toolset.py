"""The turn's skills, as ADK's ``SkillToolset``: built per turn from the person's plugins.

The turn names the plugin versions this person has; the source it was built with
(``../plugin_source.py``; this service's is the library's loader) reads them and
``PluginSkillRegistry`` holds their skills. Each turn gets a ``SkillToolset`` over
them, kept for that turn's model calls.

Only each skill's name and description go in the prompt (``EAGER``, ADK's catalog in
the system instruction, with no ``list_skills`` call to spend). The body reaches the
model when it calls ``load_skill``; resources and scripts the same way, by name.
Scripts run in the code executor the toolset was built with (this service's is the
code runner, ``script_executor.py``); with none, they can't run. A turn with no skills
gets neither the tools nor the catalog.

A message that runs a slash command (``command.py``; the reader the toolset was built
with finds it in the turn's state, and with none no turn has one) gets that command's skill first in
the registry and its instructions at every model call, so the model never has to decide
to load it.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.code_executors import BaseCodeExecutor
from google.adk.models import LlmRequest
from google.adk.skills import Skill
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.base_toolset import BaseToolset
from google.adk.tools.skill_toolset import SkillDiscoveryMode, SkillToolset
from google.adk.tools.tool_context import ToolContext

from ..plugin_source import PluginSource
from .command import CommandReader, CommandRef, command_instruction, command_skill, no_command
from .guard import GuardedSkillTool
from .registry import PluginSkillRegistry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Turn:
    toolset: SkillToolset
    # The skill of the command the message runs, and the command, when there is one.
    command: tuple[CommandRef, Skill] | None


# Turns in flight at once, each holding its toolset; older ones are built again if asked.
_TURNS_KEPT = 64


def build_skill_toolset(
    registry: PluginSkillRegistry,
    code_executor: Optional[BaseCodeExecutor],
    script_timeout: int,
) -> SkillToolset:
    return SkillToolset(
        skills=registry.skills(),
        code_executor=code_executor,
        script_timeout=script_timeout,
        # What a script wrote is kept only when there's something to run it.
        save_output_artifacts=code_executor is not None,
        discovery_mode=SkillDiscoveryMode.EAGER,
    )


class PluginSkillToolset(BaseToolset):
    def __init__(
        self,
        loader: PluginSource,
        *,
        code_executor: Optional[BaseCodeExecutor] = None,
        script_timeout: int = 300,
        read_command: CommandReader = no_command,
    ) -> None:
        """``code_executor`` runs a skill's scripts; None, and they can't run.
        ``script_timeout``: the seconds ADK gives one script, 300 being ADK's own.
        ``read_command`` finds the command a message runs in the turn's state."""
        super().__init__()
        self._loader = loader
        self._code_executor = code_executor
        self._script_timeout = script_timeout
        self._read_command = read_command
        self._turns: OrderedDict[tuple, _Turn | None] = OrderedDict()

    async def get_tools(self, readonly_context: Optional[ReadonlyContext] = None) -> list[BaseTool]:
        turn = await self._for_turn(readonly_context)
        if turn is None:
            return []
        return [GuardedSkillTool(tool) for tool in await turn.toolset.get_tools(readonly_context)]

    async def process_llm_request(self, *, tool_context: ToolContext, llm_request: LlmRequest) -> None:
        turn = await self._for_turn(tool_context)
        if turn is None:
            return
        await turn.toolset.process_llm_request(tool_context=tool_context, llm_request=llm_request)
        # Its instructions at every call, in place of a load_skill call the model would make.
        # Not activated in ADK's state: a write here reaches no event, and no skill of ours
        # adds tools on activation (adk_additional_tools).
        if turn.command is not None:
            command, skill = turn.command
            llm_request.append_instructions([command_instruction(command, skill)])

    async def _for_turn(self, context: Optional[ReadonlyContext]) -> _Turn | None:
        if context is None:
            return None
        state = context.state
        command = self._read_command(state)
        hashes = self._loader.pinned_hashes(state)
        key = (context.invocation_id, hashes, command.key if command else None)
        if not hashes and command is None:
            return None
        if key in self._turns:
            self._turns.move_to_end(key)
            return self._turns[key]
        try:
            plugins = (await self._loader.for_turn(state)).loaded if hashes else ()
            skill = command_skill(command, plugins) if command else None
            registry = PluginSkillRegistry(plugins, first=[skill] if skill else [])
            toolset = (
                build_skill_toolset(registry, self._code_executor, self._script_timeout)
                if registry.skills()
                else None
            )
            turn = _Turn(toolset, (command, skill) if command and skill else None) if toolset else None
            if turn is not None and turn.command is not None:
                logger.info("command /%s: skill %s loaded for the turn (%s)", command.name, skill.name, command.source)
        except Exception:  # noqa: BLE001 - the turn goes on without skills
            logger.exception("the turn's skills couldn't be built")
            turn = None
        self._turns[key] = turn
        while len(self._turns) > _TURNS_KEPT:
            self._turns.popitem(last=False)
        return turn

    async def close(self) -> None:
        for turn in self._turns.values():
            if turn is not None:
                await turn.toolset.close()
        self._turns.clear()
