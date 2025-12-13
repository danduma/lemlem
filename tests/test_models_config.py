import unittest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch

from lemlem.models import load_models_config, get_model_metadata, get_config
from lemlem.costs import compute_cost_for_model, validate_model_pricing


class TestModelsConfig(unittest.TestCase):
    """Test the new structured model configuration loading."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary YAML file with the new structure
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_models.yaml"

        self.test_config = {
            "models": {
                "gpt-5-mini": {
                    "model_name": "gpt-5-mini",
                    "meta": {
                        "is_thinking": True,
                        "cost_per_1m_input_tokens": 0.25,
                        "cost_per_1m_output_tokens": 2.0,
                        "cost_per_1m_cached_input": 0.025,
                        "context_window": 400000
                    }
                },
                "openrouter:kimi-k2": {
                    "model_name": "moonshotai/kimi-k2-0905",
                    "meta": {
                        "is_thinking": False,
                        "cost_per_1m_input_tokens": 0.50,
                        "cost_per_1m_output_tokens": 2.40,
                        "cost_per_1m_cached_input": 0.15,
                        "context_window": 131072
                    }
                }
            },
            "configs": {
                "openrouter:kimi-k2": {
                    "model": "openrouter:kimi-k2",
                    "base_url": "https://openrouter.ai/api/v1",
                    "api_key": "${OPENROUTER_API_KEY}",
                    "default_temp": 0.7
                },
                "direct:gpt-5-mini": {
                    "model": "gpt-5-mini",
                    "base_url": "https://api.openai.com/v1",
                    "api_key": "${OPENAI_API_KEY}",
                    "default_temp": 1.0,
                    "reasoning_effort": "low"
                }
            }
        }

        # Write test config to file
        import yaml
        with open(self.config_path, 'w') as f:
            yaml.safe_dump(self.test_config, f)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_models_config_new_structure(self):
        """Test loading the new structured config format."""
        result = load_models_config(self.test_config)

        self.assertIn("models", result)
        self.assertIn("configs", result)

        # Check models section
        models = result["models"]
        self.assertIn("gpt-5-mini", models)
        self.assertIn("openrouter:kimi-k2", models)

        # Check model metadata
        gpt_mini = models["gpt-5-mini"]
        self.assertEqual(gpt_mini["meta"]["is_thinking"], True)
        self.assertEqual(gpt_mini["meta"]["cost_per_1m_input_tokens"], 0.25)

        # Check configs section
        configs = result["configs"]
        self.assertIn("openrouter:kimi-k2", configs)
        self.assertIn("direct:gpt-5-mini", configs)

        # Check config structure
        kimi_config = configs["openrouter:kimi-k2"]
        self.assertEqual(kimi_config["model"], "openrouter:kimi-k2")
        self.assertEqual(kimi_config["base_url"], "https://openrouter.ai/api/v1")

    def test_get_model_metadata(self):
        """Test getting model metadata by ID."""
        metadata = get_model_metadata("openrouter:kimi-k2", self.test_config)
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata["meta"]["is_thinking"], False)
        self.assertEqual(metadata["meta"]["cost_per_1m_input_tokens"], 0.50)

        # Test non-existent model
        metadata = get_model_metadata("non-existent-model", self.test_config)
        self.assertIsNone(metadata)

    def test_get_config(self):
        """Test getting config with model metadata resolved."""
        config = get_config("openrouter:kimi-k2", self.test_config)
        self.assertIsNotNone(config)

        # Should have model metadata merged in
        self.assertIn("_meta", config)
        self.assertEqual(config["_meta"]["is_thinking"], False)
        self.assertEqual(config["_meta"]["cost_per_1m_input_tokens"], 0.50)

        # Should still have config-specific fields
        self.assertEqual(config["model"], "openrouter:kimi-k2")
        self.assertEqual(config["base_url"], "https://openrouter.ai/api/v1")

    def test_compute_cost_for_config_id(self):
        """Test cost computation using config ID."""
        cost = compute_cost_for_model("openrouter:kimi-k2", 1000, 500, model_configs=self.test_config)
        expected_cost = (1000 / 1_000_000) * 0.50 + (500 / 1_000_000) * 2.40
        self.assertAlmostEqual(cost, expected_cost, places=6)

    def test_compute_cost_for_model_id(self):
        """Test cost computation using model ID directly."""
        cost = compute_cost_for_model("gpt-5-mini", 1000, 500, model_configs=self.test_config)
        expected_cost = (1000 / 1_000_000) * 0.25 + (500 / 1_000_000) * 2.0
        self.assertAlmostEqual(cost, expected_cost, places=6)

    def test_compute_cost_for_provider_model_name(self):
        """Test cost computation resolves provider model_name IDs."""
        cost = compute_cost_for_model("moonshotai/kimi-k2-0905", 1000, 500, model_configs=self.test_config)
        expected_cost = (1000 / 1_000_000) * 0.50 + (500 / 1_000_000) * 2.40
        self.assertAlmostEqual(cost, expected_cost, places=6)

    def test_validate_model_pricing_new_structure(self):
        """Test pricing validation with new structure."""
        errors = validate_model_pricing(self.test_config)
        self.assertEqual(len(errors), 0)  # All models should have pricing

        # Test with missing pricing
        bad_config = {
            "models": {
                "bad-model": {
                    "meta": {
                        "is_thinking": False
                        # Missing cost fields
                    }
                }
            },
            "configs": {}
        }

        errors = validate_model_pricing(bad_config)
        self.assertIn("bad-model", errors)
        # Should report the first missing field
        self.assertTrue("Missing cost_per_1m_input_tokens" in errors["bad-model"] or
                       "Missing cost_per_1m_output_tokens" in errors["bad-model"])

    def test_compute_cost_missing_pricing_defaults_to_zero(self):
        """Missing/null pricing should compute as zero cost (not raise)."""
        bad_config = {
            "models": {
                "free-model": {
                    "model_name": "free-model",
                    "meta": {
                        "is_thinking": False,
                        "cost_per_1m_input_tokens": None,
                        "cost_per_1m_output_tokens": None,
                    },
                },
            },
            "configs": {},
        }

        cost = compute_cost_for_model("free-model", 1000, 500, model_configs=bad_config)
        self.assertIsInstance(cost, float)
        self.assertEqual(cost, 0.0)

if __name__ == '__main__':
    unittest.main()
