lemlem
======

Minimal light wrapper for LLM calls with:
- Fallback chains across models/providers
- Exponential backoff and configurable retry conditions
- OpenAI Responses API for direct OpenAI usage
- OpenAI-compatible Chat Completions for other providers/base URLs
- Environment variable expansion inside a model config dict (JSON/YAML)

Install (local / editable) with uv
----------------------------------

From this monorepo:

```
# Backend project (adds a path dep)
cd backend
uv add -e ../libs/lemlem
uv sync

# Workers venv
cd ../workers
uv venv
uv pip install -r requirements.txt
uv pip install -e ../libs/lemlem
```

Model Config
------------

You pass a MODELS_CONFIG dict (can be loaded from JSON or YAML). Values support env var expansion (e.g. `${OPENAI_API_KEY}`):

```
MODELS_CONFIG = {
  "gpt-5-nano": {
    "model_name": "gpt-5-nano",
    "base_url": "${OPENAI_BASE_URL}",
    "api_key": "${OPENAI_API_KEY}",
    "default_temp": 1,
    "thinking": true,
    "fallback": ["o4-mini", "gpt-4o-mini"],
    "rpm_limit": 3500,
    "cost_per_1k_input_tokens": 0.0015,
    "cost_per_1k_output_tokens": 0.002,
    "context_window": 128000
  }
}
```

Quick Start
-----------

```
from lemlem import LLMClient, load_models_config

models = load_models_config(MODELS_CONFIG)
client = LLMClient(models)

messages = [
  {"role": "system", "content": "You are helpful."},
  {"role": "user", "content": "Summarize HTTP/2 in one paragraph."},
]

resp = client.generate(
    model_or_chain="gpt-5-nano",
    messages=messages,
    temperature=0.2,
)
print(resp.text)
```

Fallback and Retry Controls
---------------------------

```
resp = client.generate(
  model_or_chain="gpt-5-nano",
  messages=messages,
  max_retries_per_model=2,
  retry_on_status={408, 429, 500, 502, 503, 504},
  backoff_base=0.5,
  backoff_max=8.0,
)
```

Behavior
--------
- If the target `base_url` points to OpenAI (`api.openai.com`), uses the OpenAI Responses API.
- Otherwise, uses the OpenAI-compatible Chat Completions API against the configured `base_url`.
- Returns a small, normalized result object with `.text`, `.model_used`, `.provider`, and `.raw`.

Extracting as its own repo
--------------------------

Option A: Submodule in this repo
- Add `libs/lemlem` as a git submodule pointing to `github.com/<you>/lemlem`.
- Keep path dependency/editability and push changes from here using `git -C libs/lemlem push`.

Option B: Subtree split
- Keep this directory and mirror to a new repo via `git subtree`.

Option C: Separate sibling repo
- Keep `../lemlem` as a sibling, add `uv add -e ../lemlem` during local dev, publish to use in CI.

