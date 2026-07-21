import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from lemlem import LLMClient
from lemlem.client import LLMResult
from lemlem.costs import compute_cost_for_model_strict


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

    def test_cached_tokens_from_prompt_tokens_details_dict(self):
        """Extract cached tokens when OpenRouter returns usage as a dict."""
        usage = {
            "prompt_tokens": 194,
            "prompt_tokens_details": {
                "cached_tokens": 120,
                "audio_tokens": 0,
            },
            "completion_tokens": 2,
        }
        raw = SimpleNamespace(usage=usage)

        result = LLMResult(
            text="",
            model_used="demo-config",
            provider="openrouter",
            raw=raw,
        )

        self.assertEqual(result.get_cached_tokens(), 120)

        expected = (
            (74 / 1_000_000) * 1.0  # fresh tokens
            + (120 / 1_000_000) * 0.5  # cached tokens
            + (2 / 1_000_000) * 2.0  # completion tokens
        )
        self.assertAlmostEqual(result.get_cost(self.model_data), expected)

    def test_strict_cost_marks_missing_pricing_unpriced(self):
        model_data = {
            "models": {
                "codex": {
                    "model_name": "gpt-5.6-luna",
                    "runtime": "codex_app_server",
                    "meta": {},
                }
            },
            "configs": {"codex_luna": {"model": "codex"}},
        }

        estimate = compute_cost_for_model_strict(
            "codex_luna",
            prompt_tokens=100,
            completion_tokens=20,
            cached_tokens=10,
            model_configs=model_data,
        )

        self.assertEqual(estimate.status, "unpriced")
        self.assertIsNone(estimate.amount)

    def test_strict_cost_uses_fresh_cached_and_output_prices(self):
        estimate = compute_cost_for_model_strict(
            "demo-config",
            prompt_tokens=1000,
            completion_tokens=500,
            cached_tokens=200,
            model_configs=self.model_data,
        )

        self.assertEqual(estimate.status, "priced")
        self.assertAlmostEqual(
            estimate.amount,
            (800 / 1_000_000) * 1.0
            + (200 / 1_000_000) * 0.5
            + (500 / 1_000_000) * 2.0,
        )


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
        mock_response.usage = SimpleNamespace(
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
        )
        mock_raw_response = Mock(headers={})
        mock_raw_response.parse.return_value = mock_response
        mock_instance.chat.completions.with_raw_response.create.return_value = mock_raw_response

        client.generate(
            model="openrouter-config",
            messages=[{"role": "user", "content": "hi"}],
            extra={"usage": {"foo": "bar"}},
        )

        called_kwargs = mock_instance.chat.completions.with_raw_response.create.call_args.kwargs
        usage = called_kwargs["extra_body"]["usage"]
        self.assertEqual(usage.get("foo"), "bar")
        self.assertTrue(usage.get("include"))


if __name__ == "__main__":
    unittest.main()
