import unittest
from types import SimpleNamespace
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
                "model_name": "gpt-5-nano",
                "base_url": "https://api.openai.com/v1",
                "api_key": "test-openai-key",
                "default_temp": 0.5,
                "_meta": {
                    "is_thinking": True,
                    "verbosity": "low",
                    "reasoning_effort": "minimal"
                }
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
            model="test-model",
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
        # Mock OpenAI client and response for OpenAI-compatible API
        mock_openai_instance = Mock()
        mock_openai_class.return_value = mock_openai_instance

        mock_response = Mock()
        mock_response.output = []
        mock_response.output_text = "Ocean waves crash down,\nSandcastles wash away quick,\nTides of change return."

        mock_openai_instance.responses.create.return_value = mock_response

        result = self.client.generate(
            model="openai-model",
            prompt="Write a haiku about the ocean."
        )

        self.assertIsInstance(result, LLMResult)
        self.assertEqual(result.text, "Ocean waves crash down,\nSandcastles wash away quick,\nTides of change return.")
        self.assertEqual(result.model_used, "gpt-5-nano")
        self.assertEqual(result.provider, "openai-responses")
        self.assertEqual(result.raw, mock_response)

        # Verify the API was called correctly
        mock_openai_instance.responses.create.assert_called_once()

    @patch('lemlem.client.OpenAI')
    def test_generate_with_responses_output_items(self, mock_openai_class):
        mock_instance = Mock()
        mock_openai_class.return_value = mock_instance

        message_content = [SimpleNamespace(type="output_text", text='{"task": "route"}')]
        output_item = SimpleNamespace(type="message", content=message_content)
        mock_response = SimpleNamespace(output=[output_item], output_text="")

        mock_instance.responses.create.return_value = mock_response

        result = self.client.generate(
            model="openai-model",
            prompt="Return JSON task result.",
        )

        self.assertEqual(result.text, '{"task": "route"}')
        mock_instance.responses.create.assert_called_once()

    @patch('lemlem.client.OpenAI')
    def test_generate_with_fallback_chain(self, mock_openai_class):
        # Explicit fallback via model list (config fallback is ignored)
        models_with_fallback = {
            "primary-model": {
                "model_name": "primary",
                "base_url": "https://api.primary.com",
                "api_key": "primary-key",
            },
            "backup-model": {
                "model_name": "backup",
                "base_url": "https://api.backup.com", 
                "api_key": "backup-key"
            }
        }
        
        client_with_fallback = LLMClient(models_with_fallback)
        
        from openai._exceptions import RateLimitError
        
        mock_instance = Mock()
        mock_openai_class.return_value = mock_instance

        # First model fails with rate limit, second succeeds
        first_error = RateLimitError("Rate limited", response=Mock(status_code=429), body="")
        success_response = Mock()
        success_message = Mock()
        success_message.content = "Mountain peak stands tall,\nClouds embrace its snowy crown,\nSilence speaks volumes."
        success_response.choices = [Mock(message=success_message)]

        def side_effect(*args, **kwargs):
            if not hasattr(side_effect, "called"):
                side_effect.called = True
                raise first_error
            return success_response

        mock_instance.chat.completions.create.side_effect = side_effect

        result = client_with_fallback.generate(
            model=["primary-model", "backup-model"],
            messages=[{"role": "user", "content": "Write a haiku about mountains."}]
        )

        self.assertIsInstance(result, LLMResult)
        self.assertEqual(result.text, "Mountain peak stands tall,\nClouds embrace its snowy crown,\nSilence speaks volumes.")
        self.assertEqual(result.model_used, "backup")
        self.assertEqual(result.provider, "openai-compatible")

    def test_build_chain_single_model(self):
        chain = self.client._build_chain("test-model")
        self.assertEqual(chain, ["test-model"])

    def test_build_chain_with_explicit_sequence(self):
        models = {
            "main": {},
            "backup1": {},
            "backup2": {}
        }
        client = LLMClient(models)
        chain = client._build_chain(["main", "backup1", "backup2"])
        self.assertEqual(chain, ["main", "backup1", "backup2"])

    def test_get_cfg_unknown_model(self):
        with self.assertRaises(KeyError):
            self.client._get_cfg("unknown-model")


if __name__ == '__main__':
    unittest.main()
