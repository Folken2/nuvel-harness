"""What the agent reads about model_consult: when to ask, and when not to.

ADK's own wording asks the agent to consult before any substantive work and
again before finishing, which on a chat assistant means nearly every turn. A
consultation costs many times one of the agent's own steps, so ours keeps it for
the few steps that are hard, and says so in both places the model reads: the
tool's description and a line added to the instruction.
"""

from __future__ import annotations

TOOL_DESCRIPTION = """\
Ask a stronger advisor model for guidance on one hard step. The advisor reads \
this whole conversation, your instructions and your tools' results, so ask \
only the specific decision you want reviewed and don't restate the task. It \
gives back a plan or a correction, not the answer: you still do the work.

Use it rarely. It costs far more than your own steps and is limited per turn. \
Only use it when the person asks you to, when a decision with real \
consequences is one you're unsure of (a calculation, a technical, legal or \
financial judgement, the plan for a long task), or when you are stuck: an \
approach failed twice, or results don't fit.\
"""

EXECUTOR_INSTRUCTION = """\
You have a `model_consult` tool that asks a stronger model for advice. Most \
turns never need it: don't use it for greetings, simple questions, lookups, \
writing or formatting, or anything you can answer confidently on your own.

Use it only when the person asks you to, when a step is hard and getting it \
wrong would matter (a tricky calculation, a technical, legal or financial \
judgement, the plan for a long multi-step task), or when you are stuck after \
trying. Gather what you need first, then ask one concrete question, saying \
what you are considering. Weigh the advice seriously and do the work \
yourself. If your own evidence contradicts it, tell the person what you found \
and what the advisor suggested. When it says the budget is used up, carry on \
without it.\
"""
