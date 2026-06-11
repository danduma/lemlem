# lemlem

A Python library for defining and running LLM agents. Handles model routing, fallback chains, key rotation, tool execution, conversation memory, and local skills — so you can focus on what your agent does, not how it talks to the API.

## Quickstart

```python
import os
from lemlem import Agent

agent = Agent(
    system_prompt="You are a concise technical writer.",
    model="gemini-3.5-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
)

session = agent.spawn()
result = session.send("Summarize the key tradeoffs of event sourcing in 3 bullet points.")
print(result.text)
```

That's it. No config files, no client setup, no message formatting.

## What you get back

`session.send()` returns an `AgentResult` — a structured object, not a raw string:

```python
result = session.send("Analyze this image and extract all table data.", images=["chart.png"])

result.text          # str   — primary text response (always present)
result.structured    # dict  — JSON if the agent returned structured output
result.files         # list  — any file outputs (name, bytes, mime_type)
result.images        # list  — any image outputs
result.tool_calls    # list  — tools invoked during this turn, with args and return values
result.model_used    # str   — which model was actually used (after fallbacks)
result.usage         # Usage — prompt_tokens, completion_tokens, cost_usd
result.session_id    # str   — identifies the session that produced this
```

## Multi-turn sessions

`agent.spawn()` creates an `AgentSession`. Send multiple messages to the same session and the agent remembers the conversation:

```python
session = agent.spawn()

result = session.send("My codebase uses PostgreSQL with 50M rows in the events table.")
result = session.send("What indexes should I add for time-range queries?")
# Agent knows about PostgreSQL and 50M rows from the previous turn

result = session.send("What about for user-specific queries with filters?")
# Still in context
```

For sessions that survive process restarts or span multiple tasks, pass a `session_id`. Calling `spawn` with the same id resumes the session:

```python
# First run — creates the session
session = agent.spawn(session_id="review-pr-142")
result = session.send("Review this diff", files=["changes.diff"])

# Later — resumes from where it left off
session = agent.spawn(session_id="review-pr-142")
result = session.send("Now write a one-paragraph summary of your review.")
```

## Memory modes

Control whether and how sessions persist their history:

```python
# Agent.STATELESS — no memory between turns (fastest, cheapest)
agent = Agent(system_prompt="...", model="gemini-2.5-flash", memory=Agent.STATELESS)

# Agent.EPHEMERAL — in-memory per session (default; lost when process exits)
agent = Agent(system_prompt="...", model="gemini-2.5-flash", memory=Agent.EPHEMERAL)

# Agent.PERSISTENT — stored in DB; survives restarts, resumable across processes
agent = Agent(system_prompt="...", model="gemini-2.5-flash", memory=Agent.PERSISTENT)
```

## Multimodal input

Pass files, images, and structured payloads alongside your message:

```python
result = session.send(
    "What's wrong with this PR? Focus on correctness and security.",
    files=["backend/api.py", "tests/test_api.py"],
    images=["architecture_diagram.png"],
    payload={"ticket_id": "ENG-4421", "priority": "high"},
)
```

## Tools

Define tools with a handler function. Lemlem infers the parameter schema from type annotations:

```python
from lemlem import Agent, Tool
import subprocess

agent = Agent(
    system_prompt="You are a DevOps engineer. Use tools to inspect the system.",
    model="gemini-2.5-flash",
    tools=[
        Tool(
            name="run_command",
            description="Execute a shell command and return its output.",
            params={"cmd": str},
            handler=lambda args: subprocess.check_output(
                args["cmd"], shell=True, text=True
            ),
        ),
        Tool(
            name="read_file",
            description="Read a file and return its contents.",
            params={"path": str},
            handler=lambda args: open(args["path"]).read(),
        ),
    ],
)

session = agent.spawn()
result = session.send("Check disk usage and summarize the top 5 largest directories.")
print(result.text)
print(result.tool_calls)  # see exactly what the agent ran
```

Tool handlers can be sync or async. Lemlem detects and handles both.

You can also register tools on an agent with a decorator:

```python
@agent.tool("search_docs", "Search internal documentation")
def search_docs(query: str) -> str:
    return my_search_index.query(query)
```

## Skills

Skills are reusable capability bundles — collections of scripts and MCP-backed tools that agents can discover and invoke. They live on disk and are loaded at agent startup.

```python
from lemlem import Agent
from lemlem.skills import SkillRuntimeConfig, SkillRef, MCPServerConfig

agent = Agent(
    system_prompt="You are a Git expert. Use your git skills to help with repo tasks.",
    model="gemini-2.5-flash",
    skills=SkillRuntimeConfig(
        skill_dirs=["/app/skills"],
        skills=[
            SkillRef(id="acme/git-tools"),
            SkillRef(id="acme/code-analysis"),
        ],
        mcp_servers={
            "github": MCPServerConfig(
                transport="stdio",
                command="uv",
                args=["run", "python", "/app/mcp/github_server.py"],
            )
        },
    ),
)
```

When skills are configured, lemlem:
- Loads `SKILL.md` definitions from disk
- Injects compact skill guidance into the system prompt
- Exposes bundled scripts as tools (`skill__acme__git-tools__script_name`)
- Proxies MCP-backed tools (`skill__acme__git-tools__mcp__tool_name`)

Skills are declared per-agent, not per-call. They're part of what the agent *is*.

## Async streaming

For chat UIs or real-time display, stream events as they arrive:

```python
session = agent.spawn()

async for event in session.stream("Explain how attention mechanisms work."):
    if event["type"] == "text":
        print(event["content"], end="", flush=True)
    elif event["type"] == "tool_call":
        print(f"\n[calling {event['name']}...]")
    elif event["type"] == "done":
        usage = event["usage"]
        print(f"\n\nTokens: {usage['prompt_tokens']} in, {usage['completion_tokens']} out")
```

Event types: `ack`, `text`, `tool_call`, `tool_result`, `message`, `done`

## Model configuration

### Inline (simplest)

Pass model name and api_key directly:

```python
agent = Agent(
    system_prompt="...",
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY"),
)
```

### YAML config file (recommended for multiple models)

Set `LEMLEM_MODELS_CONFIG_PATH` and define named configs:

```yaml
models:
  "gemini-2.5-flash":
    meta:
      cost_per_1m_input_tokens: 0.075
      cost_per_1m_output_tokens: 0.30
      context_window: 1048576

  "gpt-4o":
    meta:
      cost_per_1m_input_tokens: 2.50
      cost_per_1m_output_tokens: 10.00
      context_window: 128000

configs:
  "google:gemini-flash":
    model: "gemini-2.5-flash"
    base_url: "https://generativelanguage.googleapis.com/v1beta/openai/"
    api_key: "${GEMINI_API_KEY}"
    default_temp: 0.7

  "openai:gpt4o":
    model: "gpt-4o"
    api_key: "${OPENAI_API_KEY}"
    default_temp: 0.3
```

Then use config names as model IDs:

```python
agent = Agent(system_prompt="...", model="google:gemini-flash")
```

### Fallback chains

Pass a list of model IDs. Lemlem tries each in order on failure:

```python
agent = Agent(
    system_prompt="...",
    model=["google:gemini-flash", "openai:gpt4o", "anthropic:claude-sonnet"],
)
```

## Advanced: direct LLM access

For cases where you need raw API access without the agent layer:

```python
from lemlem import LLMClient, load_models_file

models = load_models_file("models.yaml")
client = LLMClient(models)

response = client.generate(
    model="google:gemini-flash",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2 + 2?"},
    ],
)
print(response.text)
print(response.model_used)
print(response.get_usage())
```

`LLMClient` handles: fallback chains, exponential backoff, key rotation, rate limit tracking, OpenAI Responses API for reasoning models, Gemini native API, cost extraction.

## Install

```bash
# With uv (recommended)
uv add git+https://github.com/danduma/evergreen.git#subdirectory=libs/lemlem

# With pip
pip install git+https://github.com/danduma/evergreen.git#subdirectory=libs/lemlem
```

For local development:

```bash
git clone https://github.com/danduma/evergreen.git
cd evergreen/libs/lemlem
uv add -e .
```
