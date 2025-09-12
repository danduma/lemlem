import os
import unittest
from lemlem import LLMClient, load_models_config


class TestLLMClientRealOpenRouter(unittest.TestCase):
    """
    Real API tests using OpenRouter's free DeepSeek model.
    These tests require OPENROUTER_API_KEY environment variable to be set.
    Run with: OPENROUTER_API_KEY=your_key python -m unittest test_real_openrouter.py
    """
    
    @classmethod
    def setUpClass(cls):
        cls.api_key = os.environ.get('OPENROUTER_API_KEY')
        if not cls.api_key:
            raise unittest.SkipTest("OPENROUTER_API_KEY not set - skipping real API tests")
        
        cls.models_config = {
            "deepseek-free": {
                "model_name": "deepseek/deepseek-chat-v3.1:free",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": cls.api_key,
                "default_temp": 0.7
            }
        }
        
        cls.client = LLMClient(load_models_config(cls.models_config))

    def test_haiku_generation_with_messages(self):
        """Test generating a haiku using the messages format."""
        messages = [
            {"role": "system", "content": "You are a haiku poet. Always respond with exactly one haiku (3 lines, 5-7-5 syllable pattern)."},
            {"role": "user", "content": "Write a haiku about coding and programming."}
        ]
        
        result = self.client.generate(
            model_or_chain="deepseek-free",
            messages=messages,
            temperature=0.8
        )
        
        self.assertIsNotNone(result.text)
        self.assertGreater(len(result.text.strip()), 0)
        self.assertEqual(result.model_used, "deepseek/deepseek-chat-v3.1:free")
        self.assertEqual(result.provider, "openai-compatible")
        
        print(f"\nGenerated haiku:\n{result.text}")
        
        # Basic haiku validation - should have 3 lines
        lines = result.text.strip().split('\n')
        # Allow some flexibility as AI might add extra formatting
        self.assertGreaterEqual(len([line for line in lines if line.strip()]), 3)

    def test_haiku_generation_with_prompt(self):
        """Test generating a haiku using the prompt format."""
        result = self.client.generate(
            model_or_chain="deepseek-free",
            prompt="Write a haiku about nature and seasons. Respond with exactly one haiku (3 lines, 5-7-5 syllables).",
            temperature=0.9
        )
        
        self.assertIsNotNone(result.text)
        self.assertGreater(len(result.text.strip()), 0)
        self.assertEqual(result.model_used, "deepseek/deepseek-chat-v3.1:free")
        self.assertEqual(result.provider, "openai-compatible")
        
        print(f"\nGenerated nature haiku:\n{result.text}")

    def test_haiku_generation_with_custom_temperature(self):
        """Test generating a haiku with different temperature settings."""
        messages = [
            {"role": "system", "content": "You are a haiku poet. Create beautiful, contemplative haikus."},
            {"role": "user", "content": "Write a haiku about artificial intelligence and creativity."}
        ]
        
        # Test with very low temperature for more deterministic output
        result = self.client.generate(
            model_or_chain="deepseek-free",
            messages=messages,
            temperature=0.1
        )
        
        self.assertIsNotNone(result.text)
        self.assertGreater(len(result.text.strip()), 0)
        print(f"\nLow-temperature AI haiku:\n{result.text}")
        
        # Test with high temperature for more creative output
        result2 = self.client.generate(
            model_or_chain="deepseek-free",
            messages=messages,
            temperature=1.2
        )
        
        self.assertIsNotNone(result2.text)
        self.assertGreater(len(result2.text.strip()), 0)
        print(f"\nHigh-temperature AI haiku:\n{result2.text}")

    def test_error_handling_with_invalid_model(self):
        """Test error handling when using an invalid model name."""
        invalid_config = {
            "nonexistent-model": {
                "model_name": "nonexistent/model:free",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": self.api_key
            }
        }
        
        client = LLMClient(load_models_config(invalid_config))
        
        # This should raise an exception due to invalid model
        with self.assertRaises(Exception):
            client.generate(
                model_or_chain="nonexistent-model",
                prompt="Test prompt"
            )

    def test_retry_functionality(self):
        """Test retry functionality with specific retry parameters."""
        messages = [
            {"role": "system", "content": "You are a haiku poet."},
            {"role": "user", "content": "Write a haiku about testing and debugging software."}
        ]
        
        result = self.client.generate(
            model_or_chain="deepseek-free",
            messages=messages,
            max_retries_per_model=3,
            backoff_base=0.1,
            backoff_max=2.0
        )
        
        self.assertIsNotNone(result.text)
        self.assertGreater(len(result.text.strip()), 0)
        print(f"\nTesting haiku:\n{result.text}")


if __name__ == '__main__':
    # Print instructions if API key is not set
    if not os.environ.get('OPENROUTER_API_KEY'):
        print("\nTo run real API tests, set your OpenRouter API key:")
        print("export OPENROUTER_API_KEY=your_key_here")
        print("python -m unittest test_real_openrouter.py\n")
        print("Get a free API key at: https://openrouter.ai/\n")
    
    unittest.main(verbosity=2)