import inspect
import unittest

from lemlem.gemini_wrapper import GeminiWrapper


class TestGeminiTimeoutWrapper(unittest.TestCase):
    def test_non_streaming_gemini_call_does_not_use_hard_thread_cancel(self):
        source = inspect.getsource(GeminiWrapper.generate_content)

        self.assertNotIn("ThreadPoolExecutor", source)
        self.assertNotIn("future.result(timeout=", source)
        self.assertNotIn("future.cancel()", source)
        self.assertNotIn("cancel_futures=True", source)


if __name__ == "__main__":
    unittest.main()
