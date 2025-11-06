lemlem
======

Minimal light wrapper for LLM calls with:
- Fallback chains across models/providers
- Exponential backoff and configurable retry conditions
- OpenAI Responses API for direct OpenAI usage
- OpenAI-compatible Chat Completions for other providers/base URLs
- Environment variable expansion inside a model config dict (JSON/YAML)

Installation
------------

**Install directly from GitHub:**

```bash
# With uv (recommended)
uv add git+https://github.com/danduma/evergreen.git#subdirectory=libs/lemlem

# With pip
pip install git+https://github.com/danduma/evergreen.git#subdirectory=libs/lemlem
```

**For local development:**

```bash
# Clone and install in editable mode
git clone https://github.com/danduma/evergreen.git
cd evergreen/libs/lemlem
uv add -e .
# or: pip install -e .
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
    "meta": {
      "is_thinking": true,
      "rpm_limit": 3500,
      "cost_per_1m_input_tokens": 0.0015,
      "cost_per_1m_output_tokens": 0.002,
      "context_window": 128000
    }
  }
}
```

API Usage
---------

### Basic Usage

```python
from lemlem import LLMClient, load_models_config

# Define your models configuration
MODELS_CONFIG = {
    "gpt-4o": {
        "model_name": "gpt-4o",
        "base_url": "${OPENAI_BASE_URL}",  # Optional, defaults to OpenAI
        "api_key": "${OPENAI_API_KEY}",
        "default_temp": 0.7,
        "meta": {
            "rpm_limit": 3500,
            "cost_per_1m_input_tokens": 0.005,
            "cost_per_1m_output_tokens": 0.015,
            "context_window": 128000
        }
    },
    "claude-3-sonnet": {
        "model_name": "claude-3-sonnet-20240229",
        "base_url": "https://api.anthropic.com/v1",
        "api_key": "${ANTHROPIC_API_KEY}",
        "default_temp": 0.3
    }
}

# Initialize client
models = load_models_config(MODELS_CONFIG)
client = LLMClient(models)

# Make a request with messages
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain quantum computing in simple terms."},
]

response = client.generate(
    model="gpt-4o",
    messages=messages,
    temperature=0.2,
)

print(f"Response: {response.text}")
print(f"Model used: {response.model_used}")
print(f"Provider: {response.provider}")
```

### Using Simple Prompts

```python
# You can also use simple string prompts instead of messages
response = client.generate(
    model="claude-3-sonnet",
    prompt="Write a haiku about programming",
    temperature=0.8,
)
print(response.text)
```

### Loading Configuration from Files

```python
from lemlem import load_models_file, load_models_from_env

# Load from JSON or YAML file
models = load_models_file("models_config.json")
# or: models = load_models_file("models_config.yaml")

# Load from environment variable (JSON string)
models = load_models_from_env("MODELS_CONFIG")

client = LLMClient(models)
```

### Fallback Chains

```python
# Explicit fallback chain (recommended and supported)
response = client.generate(
    model=["claude-3-sonnet", "gpt-5-nano", "gpt-4.1-nano"],
    messages=messages,
)
```

### Advanced Retry and Error Handling

```python
response = client.generate(
    model="gpt-4o",
    messages=messages,
    max_retries_per_model=3,
    retry_on_status={408, 429, 500, 502, 503, 504},
    backoff_base=0.5,
    backoff_max=8.0,
    extra={"max_tokens": 1000, "top_p": 0.9},  # Pass additional parameters
)
```

### Working with Different Model Types

```python
# OpenAI models (uses chat.completions)
openai_response = client.generate(
    model="gpt-4o",
    messages=messages,
    temperature=0.7,
)

# Custom API endpoint (uses OpenAI-compatible interface)
custom_response = client.generate(
    model="local-llama",  # Configured with custom base_url
    messages=messages,
    extra={"stream": False, "max_tokens": 500},
)
```

## Configuration Options

Each model in your config can have these options:

- `model_name`: The actual model name to send to the API
- `base_url`: API endpoint URL (defaults to OpenAI if not specified)
- `api_key`: API key (supports environment variable expansion with `${VAR}`)
- `default_temp`: Default temperature for this model
- `meta.is_thinking`: Set to `true` for reasoning models (o1, etc.) to use Responses API
- `meta.rpm_limit`: Rate limit (for documentation/cost tracking)
- `meta.cost_per_1m_input_tokens`: Cost tracking
- `meta.cost_per_1m_output_tokens`: Cost tracking  
- `meta.context_window`: Maximum context length
- `meta.verbosity`: Default text verbosity for Responses API
- `meta.reasoning_effort`: Default reasoning effort for Responses API

## Behavior

- Uses OpenAI Responses API for reasoning models (with `thinking: true`) on OpenAI endpoints
- Otherwise uses OpenAI-compatible Chat Completions API
- Explicit fallback through caller-provided chains with exponential backoff retry
- Returns normalized `LLMResult` object with `.text`, `.model_used`, `.provider`, and `.raw`
- Environment variable expansion in all config values using `${VARIABLE_NAME}` syntax

