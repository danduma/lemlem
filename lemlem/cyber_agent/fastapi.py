"""FastAPI helpers for CyberAgent."""
from __future__ import annotations

import json
from typing import Any, Awaitable, Callable, Optional

try:  # FastAPI is optional at library import time
    from fastapi import APIRouter, Depends, HTTPException, Request
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
except Exception as exc:  # pragma: no cover - optional dependency
    raise ImportError("fastapi is required to use lemlem.cyber_agent.fastapi") from exc

from .agent import CyberAgent


Authorizer = Callable[[Request], Awaitable[Any]]


class ChatRequest(BaseModel):
    conversation_id: Optional[str] = None
    message: str


def create_router(agent: CyberAgent, *, authorize: Optional[Authorizer] = None) -> APIRouter:
    router = APIRouter()

    async def _authorize(request: Request) -> Any:
        if authorize is None:
            return None
        return await authorize(request)

    @router.post("/chat")
    async def post_chat(payload: ChatRequest, request: Request, user=Depends(_authorize)):
        if user is None and authorize is not None:
            raise HTTPException(status_code=403, detail="Forbidden")

        async def event_stream():
            async for event in agent.stream_chat(
                conversation_id=payload.conversation_id,
                message=payload.message,
                user_metadata={"request_id": id(request)},
            ):
                yield f"data: {json.dumps(event, default=str)}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return router
