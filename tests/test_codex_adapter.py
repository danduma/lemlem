import unittest
from unittest.mock import patch

from lemlem.adapter import LLMAdapter


class CodexAdapterTestCase(unittest.TestCase):
    def test_routes_codex_runtime_without_http_client(self) -> None:
        model_data = {
            "models": {
                "codex:luna": {
                    "runtime": "codex_app_server",
                    "model_name": "gpt-5.6-luna",
                }
            },
            "configs": {
                "codex_luna_xhigh": {
                    "model": "codex:luna",
                    "reasoning_effort": "xhigh",
                }
            },
        }
        runtime_result = {
            "data": {"answer": "ok"},
            "final_text": '{"answer":"ok"}',
            "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
            "thread_id": "thread-1",
            "turn_id": "turn-1",
            "tool_interactions": [],
        }

        with patch(
            "lemlem.adapter.CodexAppServerRuntime.call_json",
            return_value=runtime_result,
        ) as call:
            result = LLMAdapter(model_data=model_data).chat_json(
                system_prompt="test",
                user_payload={"value": 1},
                model="codex_luna_xhigh",
            )

        call.assert_called_once()
        self.assertEqual(result["data"], {"answer": "ok"})
        self.assertEqual(result["model_used"], "gpt-5.6-luna")

    def test_routes_codex_text_call(self) -> None:
        model_data = {
            "models": {"codex:luna": {"runtime": "codex_app_server", "model_name": "gpt-5.6-luna"}},
            "configs": {"codex_luna_xhigh": {"model": "codex:luna", "reasoning_effort": "xhigh"}},
        }
        with patch(
            "lemlem.adapter.CodexAppServerRuntime.call_json",
            return_value={
                "data": {"text": "A title"},
                "usage": {"total_tokens": 12},
                "thread_id": "thread-1",
                "turn_id": "turn-1",
            },
        ):
            result = LLMAdapter(model_data=model_data).chat_text(
                system_prompt="title",
                user_payload={"request": "test"},
                model="codex_luna_xhigh",
            )

        self.assertEqual(result["text"], "A title")
        self.assertEqual(result["pricing_status"], "unpriced")
