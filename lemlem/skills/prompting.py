from __future__ import annotations

from typing import List

from .models import LoadedSkill, LoadedSkillBundle


def _truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _skill_prompt_block(skill: LoadedSkill) -> str:
    lines = [f"- {skill.name} ({skill.id})"]
    if skill.description:
        lines.append(f"  Description: {skill.description}")
    when_to_use = skill.sections.get("when_to_use") or skill.sections.get("overview")
    if when_to_use:
        lines.append(f"  When to use: {_truncate_text(when_to_use.replace(chr(10), ' '), 300)}")
    if skill.env_vars:
        lines.append(f"  Required env: {', '.join(skill.env_vars[:10])}")
    if skill.generated_tool_names:
        lines.append(f"  Generated tools: {', '.join(skill.generated_tool_names[:12])}")
    if skill.generated_mcp_tool_names:
        lines.append(f"  MCP tools: {', '.join(skill.generated_mcp_tool_names[:12])}")
    if skill.required_mcp_servers:
        lines.append(f"  MCP servers: {', '.join(skill.required_mcp_servers)}")
    if skill.scripts:
        lines.append(f"  Scripts: {', '.join(script.relative_path for script in skill.scripts[:12])}")
    usage = skill.sections.get("usage") or skill.sections.get("quick_start")
    if usage:
        lines.append(f"  Usage notes: {_truncate_text(usage.replace(chr(10), ' '), 320)}")
    pitfalls = skill.sections.get("pitfalls") or skill.sections.get("tips")
    if pitfalls:
        lines.append(f"  Pitfalls: {_truncate_text(pitfalls.replace(chr(10), ' '), 220)}")
    if skill.prompt_notes:
        lines.append(f"  Runtime notes: {'; '.join(skill.prompt_notes)}")
    return "\n".join(lines)


def _build_body(skills: List[LoadedSkill]) -> str:
    blocks = [
        "Skills are configured for this agent.",
        "Use configured skill tools when they directly match the task.",
        "Call `skill_help` before using a skill when arguments or workflow details are unclear.",
        "Do not invent missing MCP servers, missing scripts, or unsupported tool names.",
        "",
        "Configured skills:",
    ]
    blocks.extend(_skill_prompt_block(skill) for skill in skills)
    return "\n".join(blocks).strip()


def build_prompt_prefix(bundle: LoadedSkillBundle) -> str:
    skills = list(bundle.skills)
    if not skills:
        return ""

    budget = bundle.config.prompt_char_budget
    body = _build_body(skills)
    if len(body) <= budget:
        return body

    required = [skill for skill in skills if skill.ref.required]
    optional = [skill for skill in skills if not skill.ref.required]
    ordered = required + optional

    kept: List[LoadedSkill] = []
    for skill in ordered:
        candidate = _build_body(kept + [skill])
        if len(candidate) <= budget or not kept:
            kept.append(skill)
            continue
        skill.sections.pop("examples", None)
        skill.sections.pop("pitfalls", None)
        skill.sections.pop("workflow", None)
        skill.sections.pop("setup", None)
        candidate = _build_body(kept + [skill])
        if len(candidate) <= budget:
            kept.append(skill)

    final = _build_body(kept)
    if len(final) > budget:
        return _truncate_text(final, budget)
    return final
