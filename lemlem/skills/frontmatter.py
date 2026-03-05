from __future__ import annotations

import re
from typing import Any, Dict, Tuple

import yaml


SECTION_ALIASES = {
    "when to use": "when_to_use",
    "prerequisites": "prerequisites",
    "setup": "setup",
    "quick start": "quick_start",
    "usage": "usage",
    "workflow": "workflow",
    "pitfalls": "pitfalls",
    "examples": "examples",
    "tips": "tips",
}


def parse_skill_markdown(text: str) -> Tuple[Dict[str, Any], str]:
    stripped = text.lstrip()
    if not stripped.startswith("---\n"):
        return {}, text

    end_marker = "\n---\n"
    end_index = stripped.find(end_marker, 4)
    if end_index == -1:
        return {}, text

    frontmatter_text = stripped[4:end_index]
    body = stripped[end_index + len(end_marker):]
    parsed = yaml.safe_load(frontmatter_text) or {}
    if not isinstance(parsed, dict):
        parsed = {}
    return parsed, body


def split_markdown_sections(text: str) -> Dict[str, str]:
    sections: Dict[str, str] = {}
    current_key = "overview"
    current_lines = []

    for line in text.splitlines():
        heading_match = re.match(r"^##+\s+(.+?)\s*$", line)
        if heading_match:
            if current_lines:
                sections[current_key] = "\n".join(current_lines).strip()
                current_lines = []
            heading = heading_match.group(1).strip().lower()
            current_key = SECTION_ALIASES.get(heading, heading.replace(" ", "_"))
            continue
        current_lines.append(line)

    if current_lines:
        sections[current_key] = "\n".join(current_lines).strip()

    return {key: value for key, value in sections.items() if value}
