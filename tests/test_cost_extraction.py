import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from lemlem import LLMClient
from lemlem.client import LLMResult


class TestCostExtraction(unittest.TestCase):
    def setUp(self) -> None:
        self.model_data = {
            "models": {
                "base-model": {
                    "model_name": "base-model",
                    "meta": {
                        "cost_per_1m_input_tokens": 1.0,
                        "cost_per_1m_output_tokens": 2.0,
                        "cost_per_1m_cached_input": 0.5,
                    },
                }
            },
            "configs": {
                "demo-config": {
                    "model": "base-model",
                    "default_temp": 0.7,
                }
            },
        }

    def test_prefers_direct_cost_when_present(self):
        """Use provider-returned cost (e.g., OpenRouter usage.include) when available."""
        usage = SimpleNamespace(cost="0.0042", prompt_tokens=120, completion_tokens=80)
        raw = SimpleNamespace(usage=usage)

        result = LLMResult(
            text="",
            model_used="demo-config",
            provider="openrouter",
            raw=raw,
        )

        self.assertAlmostEqual(result.get_cost(self.model_data), 0.0042)

    def test_falls_back_to_computed_cost_without_cost_field(self):
        """Compute cost from tokens when provider does not supply explicit cost."""
        usage = SimpleNamespace(
            prompt_tokens=1000,
            completion_tokens=500,
            cached_tokens=200,
        )
        raw = SimpleNamespace(usage=usage)

        result = LLMResult(
            text="",
            model_used="demo-config",
            provider="openrouter",
            raw=raw,
        )

        expected = (800 / 1_000_000) * 1.0 + (200 / 1_000_000) * 0.5 + (500 / 1_000_000) * 2.0
        self.assertAlmostEqual(result.get_cost(self.model_data), expected)


class TestOpenRouterUsageFlag(unittest.TestCase):
    @patch("lemlem.client.OpenAI")
    def test_usage_include_is_forced_for_openrouter(self, mock_openai_class):
        """Ensure usage.include is always enabled so costs are returned."""
        model_data = {
            "models": {
                "openrouter-base": {
                    "model_name": "deepseek/deepseek-chat",
                    "base_url": "https://openrouter.ai/api/v1",
                    "api_key": "test-key",
                    "meta": {
                        "cost_per_1m_input_tokens": 1.0,
                        "cost_per_1m_output_tokens": 1.0,
                    },
                }
            },
            "configs": {
                "openrouter-config": {
                    "model": "openrouter-base",
                    "default_temp": 0.2,
                }
            },
        }

        client = LLMClient(model_data)

        mock_instance = Mock()
        mock_openai_class.return_value = mock_instance

        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "hello"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_instance.chat.completions.create.return_value = mock_response

        client.generate(
            model="openrouter-config",
            messages=[{"role": "user", "content": "hi"}],
            extra={"usage": {"foo": "bar"}},
        )

        called_kwargs = mock_instance.chat.completions.create.call_args.kwargs
        self.assertIn("usage", called_kwargs)
        self.assertEqual(called_kwargs["usage"].get("foo"), "bar")
        self.assertTrue(called_kwargs["usage"].get("include"))


if __name__ == "__main__":
    unittest.main()
