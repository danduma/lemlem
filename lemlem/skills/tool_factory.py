from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from lemlem.cyber_agent.config import ToolSpec

from .errors import MCPUnavailableError
from .mcp_bridge import MCPConnectionManager
from .models import LoadedSkill, SkillAgentAugmentation, SkillRuntimeConfig
from .loader import load_skill_bundle
from .prompting import build_prompt_prefix
from .script_runner import run_skill_script


def _sanitize_identifier(value: str) -> str:
    sanitized = []
    for char in value:
        if char.isalnum():
            sanitized.append(char.lower())
        else:
            sanitized.append("_")
    return "".join(sanitized).strip("_")


def _script_tool_name(skill: LoadedSkill, script_name: str) -> str:
    return f"skill__{_sanitize_identifier(skill.owner)}__{_sanitize_identifier(skill.slug)}__{_sanitize_identifier(script_name)}"


def _mcp_tool_name(skill: LoadedSkill, tool_name: str) -> str:
    return f"skill__{_sanitize_identifier(skill.owner)}__{_sanitize_identifier(skill.slug)}__mcp__{_sanitize_identifier(tool_name)}"


def _build_help_tool(bundle) -> ToolSpec:
    def handler(arguments: Dict[str, Any]) -> Dict[str, Any]:
        skill_id = str(arguments.get("skill_id") or "").strip()
        section = str(arguments.get("section") or "").strip() or None
        skill = bundle.by_id.get(skill_id)
        if skill is None:
            return {
                "ok": False,
                "error": "skill_not_found",
                "detail": f"Unknown skill '{skill_id}'.",
                "trace_summary": f"Skill '{skill_id}' is not configured for this agent.",
                "context": {"skill_id": skill_id},
            }
        return {
            "ok": True,
            "skill_id": skill.id,
            "name": skill.name,
            "description": skill.description,
            "version": skill.version,
            "env_vars": skill.env_vars,
            "sections": list(skill.sections.keys()),
            "selected_section": skill.sections.get(section) if section else None,
            "scripts": [
                {
                    "name": script.name,
                    "relative_path": script.relative_path,
                    "tool_name": script.tool_name,
                    "help_summary": script.help_summary,
                }
                for script in skill.scripts
            ],
            "mcp_servers": skill.required_mcp_servers,
            "mcp_tools": skill.generated_mcp_tool_names,
            "runtime_notes": skill.prompt_notes,
        }

    return ToolSpec(
        name="skill_help",
        description="Inspect configured skills, sections, generated tools, and runtime notes.",
        parameters={
            "type": "object",
            "properties": {
                "skill_id": {"type": "string"},
                "section": {"type": "string"},
            },
            "required": ["skill_id"],
            "additionalProperties": False,
        },
        handler=handler,
    )


def _build_script_tools(bundle) -> List[ToolSpec]:
    tool_specs: List[ToolSpec] = []
    for skill in bundle.skills:
        for script in skill.scripts:
            tool_name = _script_tool_name(skill, script.name)
            script.tool_name = tool_name
            skill.generated_tool_names.append(tool_name)

            def handler(arguments: Dict[str, Any], *, _skill=skill, _script=script):
                raw_arguments = arguments.get("arguments", [])
                if raw_arguments is None:
                    raw_arguments = []
                argv = [str(item) for item in raw_arguments]
                timeout = int(arguments.get("timeout_seconds") or bundle.config.script_timeout_seconds)
                stdin = arguments.get("stdin")
                return run_skill_script(
                    skill=_skill,
                    script=_script,
                    arguments=argv,
                    stdin=None if stdin is None else str(stdin),
                    timeout_seconds=timeout,
                )

            description = skill.description or f"Run {_script_tool_name(skill, script.name)}."
            if script.help_summary:
                description = f"{description} {script.help_summary}"
            tool_specs.append(
                ToolSpec(
                    name=tool_name,
                    description=description,
                    parameters={
                        "type": "object",
                        "properties": {
                            "arguments": {"type": "array", "items": {"type": "string"}},
                            "stdin": {"type": "string"},
                            "timeout_seconds": {"type": "integer", "minimum": 1},
                        },
                        "additionalProperties": False,
                    },
                    handler=handler,
                )
            )
    return tool_specs


async def _prepare_mcp_tools(bundle, manager: MCPConnectionManager) -> List[ToolSpec]:
    tool_specs: List[ToolSpec] = []
    for skill in bundle.skills:
        for server_name in skill.required_mcp_servers:
            try:
                tools = await manager.list_tools(server_name)
            except Exception as exc:
                if skill.ref.required:
                    raise MCPUnavailableError(str(exc)) from exc
                skill.unavailable_mcp_servers.append(server_name)
                skill.prompt_notes.append(f"MCP server '{server_name}' is unavailable.")
                continue

            enabled = set(skill.ref.enabled_mcp_tools or [])
            for remote_tool in tools.values():
                if enabled and remote_tool.name not in enabled:
                    continue
                tool_name = _mcp_tool_name(skill, remote_tool.name)
                skill.generated_mcp_tool_names.append(tool_name)

                async def handler(arguments: Dict[str, Any], *, _server=server_name, _tool=remote_tool.name):
                    return await manager.call_tool(_server, _tool, arguments or {})

                tool_specs.append(
                    ToolSpec(
                        name=tool_name,
                        description=remote_tool.description or f"Proxy to MCP tool '{remote_tool.name}'.",
                        parameters=remote_tool.input_schema or {"type": "object", "properties": {}},
                        handler=handler,
                    )
                )
    return tool_specs


def build_tools_and_prompt(config: SkillRuntimeConfig) -> SkillAgentAugmentation:
    bundle = load_skill_bundle(config)
    manager = MCPConnectionManager(config.mcp_servers)
    tool_specs: List[ToolSpec] = [_build_help_tool(bundle)]
    tool_specs.extend(_build_script_tools(bundle))
    if any(skill.required_mcp_servers for skill in bundle.skills):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            mcp_tools = asyncio.run(_prepare_mcp_tools(bundle, manager))
        else:
            loop = asyncio.new_event_loop()
            try:
                mcp_tools = loop.run_until_complete(_prepare_mcp_tools(bundle, manager))
            finally:
                loop.close()
        tool_specs.extend(mcp_tools)
    prompt_prefix = build_prompt_prefix(bundle)
    return SkillAgentAugmentation(
        prompt_prefix=prompt_prefix,
        tool_specs=tool_specs,
        bundle=bundle,
        mcp_manager=manager,
    )
