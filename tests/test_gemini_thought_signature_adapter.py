import unittest
from types import SimpleNamespace
from unittest.mock import patch

from lemlem.adapter import LLMAdapter
from lemlem.client import LLMClient, LLMResult


class _FakeTool:
    name = "output_message"
    description = "Emit a message."
    input_schema = {
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "additionalProperties": False,
    }

    def function(self, **kwargs):
        return {"content": [{"type": "text", "text": kwargs.get("message", "")}]}


class _FakeClient:
    def __init__(self):
        self.messages_by_call = []

    def generate(self, *, model, messages, **kwargs):
        self.messages_by_call.append([dict(message) for message in messages])
        if len(self.messages_by_call) == 1:
            tool_call = {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "output_message",
                    "arguments": "{\"message\": \"hello\"}",
                },
                "thought_signature": "c2lnbmF0dXJl",
            }
            raw = SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="",
                            tool_calls=[tool_call],
                        )
                    )
                ],
                usage=None,
            )
            return LLMResult(text="", model_used=model, provider="google", raw=raw)

        raw = SimpleNamespace(content="{\"ok\": true}", usage=None)
        return LLMResult(
            text="{\"ok\": true}",
            model_used=model,
            provider="google",
            raw=raw,
        )


class GeminiThoughtSignatureAdapterTests(unittest.TestCase):
    def test_chat_json_preserves_thought_signature_in_tool_history(self):
        client = _FakeClient()
        adapter = LLMAdapter(
            client=client,
            model_data={
                "models": {},
                "configs": {
                    "google:gemini-3-free": {
                        "model": "gemini-3-free",
                        "provider": "google",
                    }
                },
            },
        )

        adapter.chat_json(
            "system",
            {"task": "test"},
            model="google:gemini-3-free",
            tools=[_FakeTool()],
            max_tool_iterations=2,
        )

        second_call_messages = client.messages_by_call[1]
        assistant_messages = [
            message for message in second_call_messages if message.get("role") == "assistant"
        ]
        self.assertEqual(len(assistant_messages), 1)
        tool_call = assistant_messages[0]["tool_calls"][0]
        self.assertEqual(tool_call["thought_signature"], "c2lnbmF0dXJl")

    def test_llm_client_preserves_gemini_wrapper_thought_signature(self):
        class FakeGeminiWrapper:
            def __init__(self, **kwargs):
                pass

            def generate_content(self, **kwargs):
                return {
                    "choices": [
                        {
                            "message": {
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "run_wiki_shell",
                                            "arguments": "{\"cmd\": \"pwd\"}",
                                        },
                                        "thought_signature": "c2lnbmF0dXJl",
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                    "model": "gemini-test",
                }

        model_data = {
            "models": {
                "gemini-test": {
                    "model_name": "gemini-test",
                    "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
                    "api_key": "test-key",
                    "meta": {},
                }
            },
            "configs": {
                "google:gemini-test": {
                    "model": "gemini-test",
                    "enabled": True,
                }
            },
        }

        with patch("lemlem.gemini_wrapper.GeminiWrapper", FakeGeminiWrapper):
            result = LLMClient(model_data).generate(
                model="google:gemini-test",
                messages=[{"role": "user", "content": "call a tool"}],
                extra={"tools": []},
            )

        tool_call = result.raw.choices[0].message.tool_calls[0]
        self.assertEqual(tool_call.thought_signature, "c2lnbmF0dXJl")


if __name__ == "__main__":
    unittest.main()
