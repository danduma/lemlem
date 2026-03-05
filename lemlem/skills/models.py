from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


SUPPORTED_SCRIPT_SUFFIXES = {
    ".py",
    ".sh",
    ".js",
    ".mjs",
    ".cjs",
    ".ts",
    ".mts",
    ".cts",
}


@dataclass
class MCPServerConfig:
    transport: str
    command: Optional[str] = None
    args: List[str] = field(default_factory=list)
    url: Optional[str] = None
    env: Dict[str, str] = field(default_factory=dict)
    cwd: Optional[str] = None


@dataclass
class SkillRef:
    id: str
    path: Optional[str] = None
    required: bool = True
    enabled_scripts: Optional[List[str]] = None
    enabled_mcp_tools: Optional[List[str]] = None


@dataclass
class SkillRuntimeConfig:
    skill_dirs: List[str] = field(default_factory=list)
    skills: List[SkillRef] = field(default_factory=list)
    mcp_servers: Dict[str, MCPServerConfig] = field(default_factory=dict)
    prompt_char_budget: int = 12000
    script_timeout_seconds: int = 120
    allow_network_scripts: bool = True


@dataclass
class DiscoveredScript:
    name: str
    path: Path
    relative_path: str
    suffix: str
    workdir: Path
    help_summary: Optional[str] = None
    tool_name: Optional[str] = None


@dataclass
class LoadedSkill:
    ref: SkillRef
    id: str
    owner: str
    slug: str
    path: Path
    name: str
    description: str
    version: Optional[str]
    frontmatter: Dict[str, Any]
    meta: Dict[str, Any]
    sections: Dict[str, str]
    env_vars: List[str]
    manifests: List[str]
    scripts: List[DiscoveredScript]
    required_mcp_servers: List[str]
    prompt_notes: List[str] = field(default_factory=list)
    generated_tool_names: List[str] = field(default_factory=list)
    generated_mcp_tool_names: List[str] = field(default_factory=list)
    unavailable_mcp_servers: List[str] = field(default_factory=list)


@dataclass
class LoadedSkillBundle:
    config: SkillRuntimeConfig
    search_dirs: List[Path]
    skills: List[LoadedSkill]
    by_id: Dict[str, LoadedSkill]


@dataclass
class SkillAgentAugmentation:
    prompt_prefix: str
    tool_specs: Sequence[Any]
    bundle: LoadedSkillBundle
    mcp_manager: Optional[Any] = None
