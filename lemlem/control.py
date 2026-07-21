"""Control-flow signals for the agent/tool loop.

These exceptions intentionally unwind the synchronous tool loop
(``adapter._execute_tool`` runs tools inline inside ``chat_json``) back to the worker
that owns the task. They are defined in lemlem so the adapter can let them propagate
without importing worker code.
"""


class AgentControlSignal(Exception):
    """Base class for control-flow signals that must escape the tool loop."""


class AgentRunCancelled(AgentControlSignal):
    """Raised between tool calls when the owning task has been cancelled."""
