from __future__ import annotations

from typing import Any, Dict, Optional


class OpenClawSkillError(RuntimeError):
    pass


class SkillNotFoundError(OpenClawSkillError):
    pass


class InvalidSkillError(OpenClawSkillError):
    pass


class MCPUnavailableError(OpenClawSkillError):
    pass


def build_error_payload(
    *,
    error: str,
    detail: str,
    trace_summary: str,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "ok": False,
        "error": error,
        "detail": detail,
        "trace_summary": trace_summary,
        "context": context or {},
    }
