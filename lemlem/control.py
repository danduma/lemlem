"""Control-flow signals for the agent/tool loop.

These exceptions intentionally unwind the synchronous tool loop (``adapter._execute_tool``
runs tools inline inside ``chat_json``) back to the worker that owns the task, so a
long-running tool can be turned into an asynchronous "start, requeue, resume" flow
without blocking the worker. They are defined in lemlem so the adapter can let them
propagate without importing worker code.
"""
from typing import Optional


class AgentControlSignal(Exception):
    """Base class for control-flow signals that must escape the tool loop."""


class DeepResearchPending(AgentControlSignal):
    """Raised when a Gemini Deep Research job has been started/queued but is not done.

    The worker catches this, persists the interaction handle on the task, and requeues
    the task to resume with a short non-blocking status check later.
    """

    def __init__(
        self,
        interaction_id: str,
        last_event_id: Optional[str] = None,
        call_id: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> None:
        self.interaction_id = interaction_id
        self.last_event_id = last_event_id
        self.call_id = call_id
        self.prompt = prompt
        super().__init__(
            f"Gemini Deep Research pending (interaction_id={interaction_id})"
        )
