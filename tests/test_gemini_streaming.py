import os
import unittest
from types import SimpleNamespace

from lemlem.gemini_wrapper import GeminiWrapper


class _FunctionCall:
    def __init__(self, name, args):
        self.name = name
        self.args = args


class _Part:
    def __init__(self, text=None, function_call=None, thought_signature=None):
        self.text = text
        self.function_call = function_call
        self.thought_signature = thought_signature


class _Content:
    def __init__(self, parts):
        self.parts = parts


class _Candidate:
    def __init__(self, parts, finish_reason="stop"):
        self.content = _Content(parts)
        self.finish_reason = finish_reason


class _Usage:
    def __init__(self, prompt, completion, total):
        self.prompt_token_count = prompt
        self.candidates_token_count = completion
        self.total_token_count = total


class _Response:
    def __init__(self, parts, usage):
        self.candidates = [_Candidate(parts)]
        self.usage_metadata = usage


class _FakeModels:
    def __init__(self, chunks):
        self._chunks = chunks
        self.stream_called = False

    def generate_content_stream(self, **kwargs):
        self.stream_called = True
        return iter(self._chunks)


def _make_wrapper(model_name="gemini-3.5-flash", chunks=None):
    wrapper = GeminiWrapper.__new__(GeminiWrapper)
    wrapper.model_name = model_name
    wrapper._thought_signatures = {}
    wrapper._function_names = {}
    wrapper._tool_call_counter = 0
    wrapper._types = SimpleNamespace()
    wrapper.client = SimpleNamespace(models=_FakeModels(chunks or []))
    return wrapper


class TestGeminiStreaming(unittest.TestCase):
    def setUp(self):
        # Force a flush per chunk so incremental deltas are deterministic.
        self._prev = os.environ.get("LEMLEM_STREAM_DELTA_CHARS")
        os.environ["LEMLEM_STREAM_DELTA_CHARS"] = "1"

    def tearDown(self):
        if self._prev is None:
            os.environ.pop("LEMLEM_STREAM_DELTA_CHARS", None)
        else:
            os.environ["LEMLEM_STREAM_DELTA_CHARS"] = self._prev

    def test_streamed_output_equals_non_streamed(self):
        usage = _Usage(10, 20, 30)
        fc = _FunctionCall("search", {"q": "berberine"})

        # Non-streaming: the full message as a single text part (as Gemini returns it).
        non_stream_resp = _Response(
            [_Part(text="Hello world"), _Part(function_call=fc)],
            usage,
        )
        non_wrapper = _make_wrapper()
        expected = non_wrapper._convert_gemini_response_to_openai(non_stream_resp)

        # Streaming: same logical content split across chunks; usage on the last chunk.
        chunks = [
            _Response([_Part(text="Hello ")], None),
            _Response([_Part(text="world")], None),
            _Response([_Part(function_call=fc)], usage),
        ]
        deltas = []
        stream_wrapper = _make_wrapper(chunks=chunks)
        actual = stream_wrapper._generate_content_streaming(
            {}, lambda d: deltas.append(d), stream_iteration=1, hard_timeout=None
        )

        self.assertTrue(stream_wrapper.client.models.stream_called)
        self.assertEqual(actual, expected)
        # Assembled text + tool call survive streaming.
        self.assertEqual(actual["choices"][0]["message"]["content"], "Hello world")
        self.assertEqual(
            actual["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "search",
        )
        self.assertEqual(actual["usage"]["total_tokens"], 30)

        # Deltas: incremental text, final delta carries done=True, all stamped with iteration.
        self.assertGreaterEqual(len(deltas), 2)
        self.assertTrue(deltas[-1]["done"])
        self.assertTrue(all(d["iteration"] == 1 for d in deltas))
        streamed_text = "".join(d["delta_text"] for d in deltas)
        self.assertEqual(streamed_text, "Hello world")
        # Each delta is small (well under the PG NOTIFY limit).
        self.assertTrue(all(len(d["delta_text"]) < 7600 for d in deltas))

    def test_is_v35_or_newer(self):
        self.assertTrue(_make_wrapper("gemini-3.5-flash")._is_v35_or_newer())
        self.assertTrue(_make_wrapper("gemini-4.0-pro")._is_v35_or_newer())
        self.assertFalse(_make_wrapper("gemini-3-flash-preview")._is_v35_or_newer())
        self.assertFalse(_make_wrapper("gemini-2.5-flash")._is_v35_or_newer())
        self.assertFalse(_make_wrapper("gpt-5-mini")._is_v35_or_newer())


if __name__ == "__main__":
    unittest.main()
