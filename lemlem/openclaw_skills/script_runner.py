from __future__ import annotations

import os
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional

from .errors import build_error_payload
from .models import DiscoveredScript, LoadedOpenClawSkill


def _runner_label(command: List[str]) -> str:
    if not command:
        return "unknown"
    if command[:3] == ["uv", "run", "python"]:
        return "uv"
    if command[:3] == ["pnpm", "exec", "tsx"]:
        return "pnpm-tsx"
    if command[:3] == ["pnpm", "exec", "node"]:
        return "pnpm-node"
    return command[0]


def build_script_command(script: DiscoveredScript, arguments: Optional[List[str]] = None) -> List[str]:
    arguments = list(arguments or [])
    script_arg = os.path.relpath(script.path, script.workdir)
    if script.suffix == ".py":
        return ["uv", "run", "python", script_arg, *arguments]
    if script.suffix == ".sh":
        return ["bash", script_arg, *arguments]
    if script.suffix in {".js", ".mjs", ".cjs"}:
        return ["pnpm", "exec", "node", script_arg, *arguments]
    if script.suffix in {".ts", ".mts", ".cts"}:
        return ["pnpm", "exec", "tsx", script_arg, *arguments]
    raise ValueError(f"Unsupported script suffix: {script.suffix}")


def inspect_script_help(script: DiscoveredScript, timeout_seconds: int = 5) -> Optional[str]:
    try:
        payload = run_skill_script(
            skill=None,
            script=script,
            arguments=["--help"],
            stdin=None,
            timeout_seconds=timeout_seconds,
        )
    except Exception:
        return None

    if not payload.get("ok"):
        return None

    text = (payload.get("stdout") or payload.get("stderr") or "").strip()
    if not text:
        return None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    return " ".join(lines[:3])[:300]


def _missing_command_payload(skill_id: str, tool_name: str, command: List[str], script: DiscoveredScript) -> Dict[str, Any]:
    executable = command[0]
    return build_error_payload(
        error="script_runner_missing",
        detail=f"Required executable '{executable}' was not found.",
        trace_summary=(
            f"{script.relative_path} cannot run because required executable '{executable}' "
            "is not available."
        ),
        context={
            "skill_id": skill_id,
            "tool_name": tool_name,
            "runner": _runner_label(command),
            "script_path": script.relative_path,
        },
    )


def run_skill_script(
    *,
    skill: Optional[LoadedOpenClawSkill],
    script: DiscoveredScript,
    arguments: Optional[List[str]],
    stdin: Optional[str],
    timeout_seconds: int,
) -> Dict[str, Any]:
    command = build_script_command(script, arguments)
    tool_name = script.tool_name or script.name
    skill_id = skill.id if skill is not None else "unknown"
    if shutil.which(command[0]) is None:
        return _missing_command_payload(skill_id, tool_name, command, script)

    started_at = time.monotonic()
    try:
        result = subprocess.run(
            command,
            cwd=str(script.workdir),
            input=stdin,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return build_error_payload(
            error="script_timeout",
            detail=f"Script exceeded timeout of {timeout_seconds} seconds.",
            trace_summary=f"{script.relative_path} timed out after {timeout_seconds} seconds.",
            context={
                "skill_id": skill_id,
                "tool_name": tool_name,
                "runner": _runner_label(command),
                "script_path": script.relative_path,
            },
        )
    except Exception as exc:
        return build_error_payload(
            error="script_execution_failed",
            detail=str(exc),
            trace_summary=f"{script.relative_path} failed before execution completed.",
            context={
                "skill_id": skill_id,
                "tool_name": tool_name,
                "runner": _runner_label(command),
                "script_path": script.relative_path,
            },
        )

    duration_ms = int((time.monotonic() - started_at) * 1000)
    payload = {
        "ok": result.returncode == 0,
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "trace_summary": (
            f"{script.relative_path} completed successfully."
            if result.returncode == 0
            else f"{script.relative_path} failed with exit code {result.returncode}."
        ),
        "runner": _runner_label(command),
        "script_path": script.relative_path,
        "duration_ms": duration_ms,
    }
    if result.returncode != 0:
        payload["error"] = "script_execution_failed"
        payload["detail"] = result.stderr.strip() or result.stdout.strip() or "Script exited non-zero."
        payload["context"] = {
            "skill_id": skill_id,
            "tool_name": tool_name,
            "runner": _runner_label(command),
            "exit_code": result.returncode,
        }
    return payload
