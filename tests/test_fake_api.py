import unittest
from unittest.mock import Mock, patch
from lemlem import LLMClient, LLMResult


class TestLLMClientFakeAPI(unittest.TestCase):
    def setUp(self):
        self.models_config = {
            "test-model": {
                "model_name": "test-model",
                "base_url": "https://api.test.com",
                "api_key": "test-key",
                "default_temp": 0.7
            },
            "openai-model": {
                "model_name": "gpt-4o-mini",
                "base_url": "https://api.openai.com/v1",
                "api_key": "test-openai-key",
                "default_temp": 0.5
            }
        }
        self.client = LLMClient(self.models_config)

    @patch('lemlem.client.OpenAI')
    def test_generate_with_fake_openai_compatible_response(self, mock_openai_class):
        # Mock OpenAI client and response for OpenAI-compatible API
        mock_openai_instance = Mock()
        mock_openai_class.return_value = mock_openai_instance
        
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Cherry blossoms fall,\nSilent whispers in the breeze,\nSpring's gentle embrace."
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        
        mock_openai_instance.chat.completions.create.return_value = mock_response
        
        messages = [
            {"role": "system", "content": "You are a haiku poet."},
            {"role": "user", "content": "Write a haiku about spring."}
        ]
        
        result = self.client.generate(
            model_or_chain="test-model",
            messages=messages,
            temperature=0.8
        )
        
        self.assertIsInstance(result, LLMResult)
        self.assertEqual(result.text, "Cherry blossoms fall,\nSilent whispers in the breeze,\nSpring's gentle embrace.")
        self.assertEqual(result.model_used, "test-model")
        self.assertEqual(result.provider, "openai-compatible")
        self.assertEqual(result.raw, mock_response)
        
        # Verify the API was called correctly
        mock_openai_instance.chat.completions.create.assert_called_once_with(
            model="test-model",
            messages=messages,
            temperature=0.8
        )

    @patch('lemlem.client.OpenAI')
    def test_generate_with_fake_openai_responses_api(self, mock_openai_class):
        # Mock OpenAI client and response for OpenAI Responses API
        mock_openai_instance = Mock()
        mock_openai_class.return_value = mock_openai_instance
        
        mock_response = Mock()
        mock_response.output_text = "Ocean waves crash down,\nSandcastles wash away quick,\nTides of change return."
        
        mock_openai_instance.responses.create.return_value = mock_response
        
        result = self.client.generate(
            model_or_chain="openai-model",
            prompt="Write a haiku about the ocean."
        )
        
        self.assertIsInstance(result, LLMResult)
        self.assertEqual(result.text, "Ocean waves crash down,\nSandcastles wash away quick,\nTides of change return.")
        self.assertEqual(result.model_used, "gpt-4o-mini")
        self.assertEqual(result.provider, "openai")
        self.assertEqual(result.raw, mock_response)
        
        # Verify the API was called correctly
        mock_openai_instance.responses.create.assert_called_once_with(
            model="gpt-4o-mini",
            input="Write a haiku about the ocean.",
            temperature=0.5
        )

    @patch('lemlem.client.OpenAI')
    def test_generate_with_fallback_chain(self, mock_openai_class):
        # Add fallback to config
        models_with_fallback = {
            "primary-model": {
                "model_name": "primary",
                "base_url": "https://api.primary.com",
                "api_key": "primary-key",
                "fallback": ["backup-model"]
            },
            "backup-model": {
                "model_name": "backup",
                "base_url": "https://api.backup.com", 
                "api_key": "backup-key"
            }
        }
        
        client_with_fallback = LLMClient(models_with_fallback)
        
        from openai._exceptions import RateLimitError
        
        # Create separate mock instances for each model call
        mock_primary_instance = Mock()
        mock_backup_instance = Mock()
        
        def mock_openai_side_effect(*args, **kwargs):
            # Return different instances based on the API key
            api_key = kwargs.get('api_key')
            if api_key == 'primary-key':
                return mock_primary_instance
            elif api_key == 'backup-key':
                return mock_backup_instance
            return Mock()
        
        mock_openai_class.side_effect = mock_openai_side_effect
        
        # First model fails with rate limit
        mock_primary_instance.chat.completions.create.side_effect = RateLimitError(
            "Rate limited", response=Mock(status_code=429), body=""
        )
        
        # Second model succeeds
        mock_backup_response = Mock()
        mock_backup_response.choices = [Mock(message=Mock(content="Mountain peak stands tall,\nClouds embrace its snowy crown,\nSilence speaks volumes."))]
        mock_backup_instance.chat.completions.create.return_value = mock_backup_response
        
        result = client_with_fallback.generate(
            model_or_chain="primary-model",
            messages=[{"role": "user", "content": "Write a haiku about mountains."}]
        )
        
        self.assertIsInstance(result, LLMResult)
        self.assertEqual(result.text, "Mountain peak stands tall,\nClouds embrace its snowy crown,\nSilence speaks volumes.")
        self.assertEqual(result.model_used, "backup")
        self.assertEqual(result.provider, "openai-compatible")

    def test_build_chain_single_model(self):
        chain = self.client._build_chain("test-model")
        self.assertEqual(chain, ["test-model"])

    def test_build_chain_with_fallback(self):
        models_with_fallback = {
            "main": {
                "fallback": ["backup1", "backup2"]
            },
            "backup1": {},
            "backup2": {}
        }
        client = LLMClient(models_with_fallback)
        chain = client._build_chain("main")
        self.assertEqual(chain, ["main", "backup1", "backup2"])

    def test_get_cfg_unknown_model(self):
        with self.assertRaises(KeyError):
            self.client._get_cfg("unknown-model")


if __name__ == '__main__':
    unittest.main()