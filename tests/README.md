# Lemlem Tests

This directory contains comprehensive tests for the lemlem library, including both API tests with fake data and real endpoint tests.

## Test Files

- `test_fake_api.py` - Unit tests using mocked API responses (always runnable)
- `test_real_openrouter.py` - Integration tests using real OpenRouter API calls (requires API key)
- `demo_haiku.py` - Interactive demo showing haiku generation capabilities
- `run_tests.py` - Complete test runner script

## Quick Start

### Run All Tests (Recommended)
```bash
python run_tests.py
```

### Run Individual Test Suites

**Fake API Tests (No API Key Required):**
```bash
python -m unittest tests.test_fake_api -v
```

**Real API Tests (Requires OPENROUTER_API_KEY):**
```bash
export OPENROUTER_API_KEY=your_key_here
python -m unittest tests.test_real_openrouter -v
```

**Interactive Demo:**
```bash
python tests/demo_haiku.py
```

## OpenRouter Setup

To run real API tests, you need an OpenRouter API key:

1. Go to https://openrouter.ai/
2. Sign up for a free account
3. Get your API key
4. Set it as an environment variable:
   ```bash
   export OPENROUTER_API_KEY=your_key_here
   ```

The tests use the free DeepSeek model: `deepseek/deepseek-chat-v3.1:free`

## Test Coverage

### Fake API Tests (`test_fake_api.py`)
- ✅ OpenAI-compatible API calls
- ✅ OpenAI Responses API calls  
- ✅ Fallback chain functionality
- ✅ Error handling
- ✅ Model configuration
- ✅ Temperature settings
- ✅ Message vs prompt formats

### Real API Tests (`test_real_openrouter.py`)
- ✅ Real haiku generation with messages
- ✅ Real haiku generation with prompts
- ✅ Temperature variation testing
- ✅ Error handling with invalid models
- ✅ Retry functionality

### Demo Features (`demo_haiku.py`)
- ✅ Mocked API demonstration
- ✅ Real API demonstration (if key available)
- ✅ Environment variable expansion
- ✅ Fallback chain visualization
- ✅ Haiku generation examples

## Example Output

When tests run successfully, you'll see haiku output like:

```
Code flows like water,
Logic branches, functions bloom—
Digital zen speaks.
```

The real API tests generate actual haikus using the DeepSeek thinking model through OpenRouter's free tier.