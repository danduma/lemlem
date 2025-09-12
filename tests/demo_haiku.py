#!/usr/bin/env python3
"""
Demo script to test lemlem with haiku generation.
This script demonstrates both fake (mocked) and real API usage.
"""

import os
import sys
from unittest.mock import Mock, patch

# Add the parent directory to sys.path to import lemlem
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lemlem import LLMClient, load_models_config


def demo_fake_haiku():
    """Demo with mocked API calls for testing without real API key."""
    print("=== DEMO: Fake API (Mocked) ===")
    
    models_config = {
        "demo-model": {
            "model_name": "demo-haiku-model",
            "base_url": "https://api.demo.com",
            "api_key": "demo-key",
            "default_temp": 0.8
        }
    }
    
    client = LLMClient(load_models_config(models_config))
    
    with patch('lemlem.client.OpenAI') as mock_openai_class:
        # Mock the response
        mock_openai_instance = Mock()
        mock_openai_class.return_value = mock_openai_instance
        
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = """Code flows like water,
Logic branches, functions bloom—
Digital zen speaks."""
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        
        mock_openai_instance.chat.completions.create.return_value = mock_response
        
        messages = [
            {"role": "system", "content": "You are a haiku poet who writes about technology and programming."},
            {"role": "user", "content": "Write a haiku about coding and software development."}
        ]
        
        result = client.generate(
            model="demo-model",
            messages=messages,
            temperature=0.8
        )
        
        print(f"Model used: {result.model_used}")
        print(f"Provider: {result.provider}")
        print(f"Generated haiku:\n{result.text}")
        print()


def demo_real_haiku():
    """Demo with real API calls using OpenRouter's free DeepSeek model."""
    api_key = os.environ.get('OPENROUTER_API_KEY')
    
    if not api_key:
        print("=== DEMO: Real API (Skipped - No API Key) ===")
        print("To test with real API, set OPENROUTER_API_KEY environment variable.")
        print("Get a free API key at: https://openrouter.ai/")
        print("Then run: OPENROUTER_API_KEY=your_key python demo_haiku.py")
        print()
        return
    
    print("=== DEMO: Real API (OpenRouter DeepSeek) ===")
    
    models_config = {
        "deepseek-free": {
            "model_name": "deepseek/deepseek-chat-v3.1:free", 
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": api_key,
            "default_temp": 0.7
        }
    }
    
    client = LLMClient(load_models_config(models_config))
    
    try:
        messages = [
            {"role": "system", "content": "You are a master haiku poet. Create beautiful, contemplative haikus that capture the essence of the subject in exactly 3 lines with 5-7-5 syllable pattern."},
            {"role": "user", "content": "Write a haiku about artificial intelligence and creativity."}
        ]
        
        print("Generating haiku... (this may take a few seconds)")
        result = client.generate(
            model="deepseek-free",
            messages=messages,
            temperature=0.8
        )
        
        print(f"Model used: {result.model_used}")
        print(f"Provider: {result.provider}")
        print(f"Generated haiku:\n{result.text}")
        print()
        
        # Generate another one with different topic
        messages[1]["content"] = "Write a haiku about the ocean and its mysteries."
        result2 = client.generate(
            model="deepseek-free",
            messages=messages,
            temperature=0.9
        )
        
        print(f"Second haiku:\n{result2.text}")
        print()
        
    except Exception as e:
        print(f"Error calling real API: {e}")
        print("This might be due to rate limiting, network issues, or invalid API key.")
        print()


def demo_model_config_features():
    """Demo advanced model configuration features."""
    print("=== DEMO: Model Configuration Features ===")
    
    models_config = {
        "primary-with-fallback": {
            "model_name": "primary-model",
            "base_url": "https://api.primary.com",
            "api_key": "${OPENROUTER_API_KEY}",  # Environment variable expansion
            "default_temp": 0.5,
            "fallback": ["backup-model"]
        },
        "backup-model": {
            "model_name": "backup-model", 
            "base_url": "https://api.backup.com",
            "api_key": "${OPENROUTER_API_KEY}",
            "default_temp": 0.7
        }
    }
    
    # Show environment variable expansion
    expanded_config = load_models_config(models_config)
    
    print("Original config (with env vars):")
    print(f"  API key: {models_config['primary-with-fallback']['api_key']}")
    
    print("Expanded config:")
    api_key_expanded = expanded_config['primary-with-fallback']['api_key']
    if api_key_expanded.startswith("${"):
        print(f"  API key: {api_key_expanded} (not set)")
    else:
        print(f"  API key: {api_key_expanded[:10]}... (set)")
    
    client = LLMClient(expanded_config)
    chain = client._build_chain("primary-with-fallback")
    print(f"Fallback chain: {chain}")
    print()


def main():
    """Run all demos."""
    print("Lemlem Library Haiku Generation Demo")
    print("=" * 40)
    print()
    
    demo_fake_haiku()
    demo_real_haiku()
    demo_model_config_features()
    
    print("Demo completed!")
    print()
    print("To run unit tests:")
    print("  python -m unittest tests.test_fake_api -v")
    print("  OPENROUTER_API_KEY=your_key python -m unittest tests.test_real_openrouter -v")


if __name__ == "__main__":
    main()