import unittest
from types import SimpleNamespace

from lemlem.client import _extract_chat_message_text, _extract_responses_output_text


class TestMessageExtraction(unittest.TestCase):
    def test_tool_call_arguments_do_not_leak(self) -> None:
        function = SimpleNamespace(arguments='{"ok": true}')
        tool_call = SimpleNamespace(function=function)
        message = SimpleNamespace(tool_calls=[tool_call], content="use_this_text")

        self.assertEqual(_extract_chat_message_text(message), "use_this_text")

    def test_tool_call_without_content_returns_empty_string(self) -> None:
        function = SimpleNamespace(arguments='{"ok": true}')
        tool_call = SimpleNamespace(function=function)
        message = SimpleNamespace(tool_calls=[tool_call], content=None)

        self.assertEqual(_extract_chat_message_text(message), "")

    def test_reasoning_content_used_when_content_empty(self) -> None:
        reasoning_part = {"type": "output_text", "text": '{"task": "route"}'}
        message = SimpleNamespace(tool_calls=None, content=None, reasoning_content=[reasoning_part])

        self.assertEqual(_extract_chat_message_text(message), '{"task": "route"}')

    def test_response_metadata_fallback(self) -> None:
        metadata = {"output_text": '{"task": "metadata"}'}
        message = SimpleNamespace(
            tool_calls=None,
            content=None,
            reasoning_content=None,
            response_metadata=metadata,
        )

        self.assertEqual(_extract_chat_message_text(message), '{"task": "metadata"}')

    def test_responses_message_items(self) -> None:
        content_items = [SimpleNamespace(type="output_text", text="hello world")]
        response_item = SimpleNamespace(type="message", content=content_items)
        response = SimpleNamespace(output=[response_item], output_text="")

        self.assertEqual(_extract_responses_output_text(response), "hello world")

    def test_responses_fallback_output_text(self) -> None:
        response = SimpleNamespace(output=[], output_text="fallback text")
        self.assertEqual(_extract_responses_output_text(response), "fallback text")


if __name__ == "__main__":
    unittest.main()
