#!/usr/bin/env python3
"""
Unit tests for cost calculation in libs/lemlem.
Tests the compute_cost_for_model function with various model configurations.
"""
from __future__ import annotations

import unittest
from lemlem import compute_cost_for_model
from app.llm import MODEL_DATA


class TestCostCalculation(unittest.TestCase):
    """Test cost calculation logic."""

    def test_compute_cost_for_known_models(self):
        """Test cost calculation for known models in MODEL_DATA."""
        # Check if we have models loaded
        if not MODEL_DATA.get("models"):
            print("Warning: No models loaded in MODEL_DATA, skipping test")
            return

        # Test a few standard models if they exist
        test_models = [
            ("google:gemini-3-flash-free", 1000, 500),
            ("gpt-4o-mini", 1000, 500),
            ("claude-3-5-sonnet", 1000, 500)
        ]

        for config_id, prompt_tokens, completion_tokens in test_models:
            if config_id not in MODEL_DATA.get("configs", {}):
                continue
                
            cost = compute_cost_for_model(
                config_id,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                model_configs=MODEL_DATA
            )
            
            # Cost should be a float and >= 0 (some might be 0 like free models)
            self.assertIsInstance(cost, float)
            self.assertGreaterEqual(cost, 0.0)
            
            print(f"Cost for {config_id}: ${cost:.6f}")

    def test_compute_cost_with_cached_tokens(self):
        """Test cost calculation with cached tokens."""
        # Find a model that supports caching or just test the calculation logic
        # We can create a mock MODEL_DATA for deterministic testing
        
        mock_model_data = {
            "configs": {
                "test-cache-model": {
                    "model": "provider/test-model",
                    "provider": "test_provider"
                }
            },
            "models": {
                "provider/test-model": {
                    "pricing": {
                        "prompt": "0.00001",
                        "completion": "0.00003",
                        "cache_read": "0.000001" 
                    }
                }
            }
        }
        
        # Scenario: 1000 prompt tokens, 500 cached tokens, 200 completion tokens
        # Cost = (1000 * 1e-5) + (500 * 1e-6) + (200 * 3e-5)
        #      = 0.01 + 0.0005 + 0.006 = 0.0165
        
        cost = compute_cost_for_model(
            "test-cache-model",
            prompt_tokens=1000,
            completion_tokens=200,
            cached_tokens=500,
            model_configs=mock_model_data
        )
        
        self.assertAlmostEqual(cost, 0.0165, places=6)

    def test_compute_cost_fallback(self):
        """Test fallback when pricing is missing."""
        mock_model_data = {
            "configs": {
                "unknown-model": {
                    "model": "provider/unknown",
                }
            },
            "models": {
                "provider/unknown": {} # No pricing
            }
        }
        
        cost = compute_cost_for_model(
            "unknown-model",
            prompt_tokens=1000,
            completion_tokens=500,
            model_configs=mock_model_data
        )
        
        # Should return 0.0 safely
        self.assertEqual(cost, 0.0)


if __name__ == "__main__":
    unittest.main()
