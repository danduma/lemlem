# Lemlem Changelog

This changelog tracks library-level behavior changes in `lemlem` and how dependent apps should adopt them.

## 2026-03-05

### Added
- Generic local `skills` runtime for `CyberAgent`.
  - New package: `lemlem.skills`
  - Entry points:
    - `load_skill_bundle(...)`
    - `build_tools_and_prompt(...)`
- New skill runtime config types:
  - `SkillRef`
  - `SkillRuntimeConfig`
  - `MCPServerConfig`
  - `SkillAgentAugmentation`
- New `AgentConfig` field:
  - `skills_runtime`
- Local skill loading support for:
  - `SKILL.md` frontmatter and markdown sections
  - optional `_meta.json`
  - bundled `scripts/` and `bin/`
  - MCP-backed skills via `requires.mcp`
- Generated tool surface for skills:
  - `skill_help`
  - script wrapper tools named like `skill__owner__slug__script_name`
  - MCP proxy tools named like `skill__owner__slug__mcp__tool_name`
- Structured error payloads for script and MCP failures.
- Test coverage for the full runtime:
  - loader unit tests
  - prompt synthesis unit tests
  - script runner unit tests
  - MCP bridge integration test
  - `CyberAgent` integration test

### Changed
- Skill runtime naming is now generic `skills`, not provider-branded.
- `CyberAgent` now prepends synthesized skill guidance to `system_prompt` when `skills_runtime` is configured.
- `CyberAgent` now appends generated skill tools to any explicitly configured agent tools.
- Runtime search path order for local skills:
  1. `SkillRuntimeConfig.skill_dirs`
  2. `<cwd>/skills`
  3. `~/.skills`

### Why this matters
- Apps using `lemlem` can give agents reusable local skill packs without hardcoding each script or MCP tool manually.
- Skills stay explicit per agent run or per agent config.
- Failures from external scripts or MCP servers are returned as structured tool output instead of opaque runtime exceptions.

### How to use in dependent projects
1. Create a local skill directory with `owner/slug/SKILL.md`.
2. If the skill includes scripts, keep them under `scripts/` or `bin/`.
3. If the skill needs MCP, declare server names under `requires.mcp` in `SKILL.md`.
4. Configure the agent:

```python
from lemlem.cyber_agent.config import AgentConfig
from lemlem.skills import SkillRuntimeConfig, SkillRef

config = AgentConfig(
    agent_id="demo",
    system_prompt="You are a helpful assistant.",
    model="google:gemini-2.5-flash",
    skills_runtime=SkillRuntimeConfig(
        skill_dirs=["/app/skills"],
        skills=[SkillRef(id="exampleowner/script-skill")],
    ),
)
```

5. If the skill needs MCP, provide `mcp_servers`:

```python
from lemlem.skills import MCPServerConfig

mcp_servers = {
    "fixturemcp": MCPServerConfig(
        transport="stdio",
        command="uv",
        args=["run", "python", "/app/path/to/server.py"],
    )
}
```

6. Use `skill_help` in agent flows when the model needs more context about a configured skill.

### Operational notes
- `lemlem` does not fetch skills from GitHub or registries.
- `lemlem` does not install missing dependencies for third-party skills.
- Python scripts run via `uv run python`.
- Shell scripts run via `bash`.
- JS scripts run via `pnpm exec node`.
- TS scripts run via `pnpm exec tsx`.
- Missing executables or failed script runs return structured JSON tool output.

## Template for future entries

```md
## YYYY-MM-DD

### Added
- ...

### Changed
- ...

### Fixed
- ...

### How to use in dependent projects
1. ...
2. ...
```
