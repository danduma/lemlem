from __future__ import annotations

from typing import Any, Dict, Optional


class SkillError(RuntimeError):
    pass


class SkillNotFoundError(SkillError):
    pass


class InvalidSkillError(SkillError):
    pass


class MCPUnavailableError(SkillError):
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
