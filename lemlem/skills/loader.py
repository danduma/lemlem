from __future__ import annotations

from pathlib import Path
import re
from typing import Dict, Iterable, List, Set

from .errors import InvalidSkillError, SkillNotFoundError
from .frontmatter import parse_skill_markdown, split_markdown_sections
from .models import (
    DiscoveredScript,
    LoadedSkill,
    LoadedSkillBundle,
    SkillRuntimeConfig,
    SkillRef,
    SUPPORTED_SCRIPT_SUFFIXES,
)
from .script_runner import inspect_script_help


def _unique_paths(paths: Iterable[Path]) -> List[Path]:
    seen: Set[str] = set()
    ordered: List[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(path)
    return ordered


def resolve_search_dirs(config: SkillRuntimeConfig) -> List[Path]:
    cwd = Path.cwd()
    home = Path.home()
    candidates = [Path(item).expanduser() for item in config.skill_dirs]
    candidates.extend([cwd / "skills", home / ".skills"])
    return _unique_paths(path for path in candidates if str(path).strip())


def _resolve_skill_path(ref: SkillRef, search_dirs: List[Path]) -> Path:
    if ref.path:
        path = Path(ref.path).expanduser()
        if not path.exists():
            raise SkillNotFoundError(f"Skill path not found: {path}")
        return path

    if "/" not in ref.id:
        raise SkillNotFoundError(f"Skill id must be owner/slug: {ref.id}")

    owner, slug = ref.id.split("/", 1)
    for base_dir in search_dirs:
        candidate = base_dir / owner / slug
        if candidate.exists():
            return candidate

    raise SkillNotFoundError(f"Skill '{ref.id}' was not found in configured directories.")


def _extract_env_vars(frontmatter: Dict[str, object], text: str) -> List[str]:
    env_vars = []
    metadata = frontmatter.get("metadata")
    if isinstance(metadata, dict):
        skill_metadata = metadata.get("skills")
        if isinstance(skill_metadata, dict):
            env_vars.extend([item for item in skill_metadata.get("env", []) if isinstance(item, str)])

    pattern = re.compile(r"\b[A-Z][A-Z0-9_]{2,}\b")
    env_vars.extend(pattern.findall(text))
    deduped = []
    seen = set()
    for item in env_vars:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _discover_manifests(skill_root: Path) -> List[str]:
    manifests = []
    for manifest_name in ("pyproject.toml", "package.json", "requirements.txt", "uv.lock"):
        for candidate in [skill_root / manifest_name, skill_root / "scripts" / manifest_name]:
            if candidate.exists():
                manifests.append(str(candidate.relative_to(skill_root)))
    return manifests


def _nearest_workdir(skill_root: Path, script_path: Path) -> Path:
    candidates = [script_path.parent]
    candidates.extend(script_path.parents)
    for directory in candidates:
        if directory == directory.parent:
            break
        if skill_root not in {directory, *directory.parents}:
            continue
        for manifest_name in ("pyproject.toml", "package.json", "requirements.txt"):
            if (directory / manifest_name).exists():
                return directory
        if directory == skill_root:
            break
    return skill_root


def _discover_root_references(skill_root: Path, text: str) -> Set[Path]:
    matches: Set[Path] = set()
    pattern = re.compile(r"([A-Za-z0-9_./-]+\.(?:py|sh|js|mjs|cjs|ts|mts|cts))")
    for relative_path in pattern.findall(text):
        candidate = (skill_root / relative_path).resolve()
        if candidate.exists() and candidate.is_file() and skill_root in {candidate.parent, *candidate.parents}:
            matches.add(candidate)
    return matches


def _discover_scripts(skill_root: Path, skill_body: str, ref: SkillRef) -> List[DiscoveredScript]:
    candidates: Set[Path] = set()
    for folder_name in ("scripts", "bin"):
        folder = skill_root / folder_name
        if not folder.exists():
            continue
        for path in folder.rglob("*"):
            if not path.is_file():
                continue
            if any(part.startswith(".") or part == "node_modules" for part in path.parts):
                continue
            if path.suffix not in SUPPORTED_SCRIPT_SUFFIXES:
                continue
            candidates.add(path)

    candidates.update(_discover_root_references(skill_root, skill_body))
    selected = []
    enabled = set(ref.enabled_scripts or [])
    for path in sorted(candidates):
        relative_path = str(path.relative_to(skill_root))
        script_name = path.stem
        if enabled and relative_path not in enabled and script_name not in enabled:
            continue
        script = DiscoveredScript(
            name=script_name,
            path=path,
            relative_path=relative_path,
            suffix=path.suffix,
            workdir=_nearest_workdir(skill_root, path),
        )
        script.help_summary = inspect_script_help(script)
        selected.append(script)
    return selected


def _section_value(sections: Dict[str, str], *keys: str) -> str:
    for key in keys:
        if sections.get(key):
            return sections[key]
    return ""


def _infer_owner_slug(ref: SkillRef, skill_path: Path) -> tuple[str, str]:
    if "/" in ref.id:
        return tuple(ref.id.split("/", 1))  # type: ignore[return-value]
    parent = skill_path.parent.name or "local"
    return parent, skill_path.name


def load_skill_bundle(config: SkillRuntimeConfig) -> LoadedSkillBundle:
    search_dirs = resolve_search_dirs(config)
    skills: List[LoadedSkill] = []
    by_id: Dict[str, LoadedSkill] = {}

    for ref in config.skills:
        try:
            skill_path = _resolve_skill_path(ref, search_dirs)
        except SkillNotFoundError:
            if ref.required:
                raise
            continue
        skill_file = skill_path / "SKILL.md"
        if not skill_file.exists():
            if ref.required:
                raise InvalidSkillError(f"Skill at {skill_path} is missing SKILL.md")
            continue

        frontmatter, body = parse_skill_markdown(skill_file.read_text(encoding="utf-8"))
        sections = split_markdown_sections(body)
        meta_path = skill_path / "_meta.json"
        meta = {}
        if meta_path.exists():
            import json

            meta = json.loads(meta_path.read_text(encoding="utf-8"))

        owner, slug = _infer_owner_slug(ref, skill_path)
        skill_id = ref.id if ref.id else f"{owner}/{slug}"
        description = str(frontmatter.get("description") or _section_value(sections, "overview")).strip()
        version = frontmatter.get("version")
        if version is None and isinstance(meta.get("latest"), dict):
            version = meta["latest"].get("version")

        requires = frontmatter.get("requires") if isinstance(frontmatter.get("requires"), dict) else {}
        required_mcp_servers = [item for item in requires.get("mcp", []) if isinstance(item, str)]
        scripts = _discover_scripts(skill_path, body, ref)

        skill = LoadedSkill(
            ref=ref,
            id=skill_id,
            owner=owner,
            slug=slug,
            path=skill_path,
            name=str(frontmatter.get("name") or meta.get("displayName") or slug),
            description=description,
            version=str(version) if version is not None else None,
            frontmatter=frontmatter,
            meta=meta,
            sections=sections,
            env_vars=_extract_env_vars(frontmatter, skill_file.read_text(encoding="utf-8")),
            manifests=_discover_manifests(skill_path),
            scripts=scripts,
            required_mcp_servers=required_mcp_servers,
        )
        skills.append(skill)
        by_id[skill_id] = skill

    return LoadedSkillBundle(config=config, search_dirs=search_dirs, skills=skills, by_id=by_id)
