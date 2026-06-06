import threading
import time
import unittest

from lemlem.adapter import LLMAdapter


class _FakeTool:
    """Minimal tool object exposing the attributes the adapter reads."""

    def __init__(self, handler, *, parallel_safe=False, resource_key=None):
        self.handler = handler
        self.parallel_safe = parallel_safe
        if resource_key is not None:
            self.resource_key = resource_key


def _sleep_tool(delay: float):
    def handler(args):
        time.sleep(delay)
        return {"content": [{"type": "text", "text": "ok"}]}

    return handler


class TestParallelToolCalls(unittest.TestCase):
    def setUp(self):
        self.adapter = LLMAdapter(model_data={"models": {}, "configs": {}})
        # Force a generous pool so concurrency isn't capped below the test size.
        self.adapter._tool_max_concurrency = 8

    def _calls(self, names):
        return [
            {"pos": i, "tool_name": n, "call_id": f"c{i}", "parsed_args": {}}
            for i, n in enumerate(names)
        ]

    def test_independent_safe_tools_run_in_parallel(self):
        registry = {
            "a": _FakeTool(_sleep_tool(0.3), parallel_safe=True),
            "b": _FakeTool(_sleep_tool(0.3), parallel_safe=True),
            "c": _FakeTool(_sleep_tool(0.3), parallel_safe=True),
        }
        start = time.monotonic()
        results = self.adapter._execute_tool_calls(registry, self._calls(["a", "b", "c"]))
        elapsed = time.monotonic() - start
        self.assertEqual(len(results), 3)
        # ~max latency (0.3s), nowhere near the serial sum (0.9s).
        self.assertLess(elapsed, 0.6)

    def test_non_allowlisted_tools_run_serially(self):
        registry = {
            "a": _FakeTool(_sleep_tool(0.25)),  # not parallel_safe
            "b": _FakeTool(_sleep_tool(0.25)),
        }
        start = time.monotonic()
        self.adapter._execute_tool_calls(registry, self._calls(["a", "b"]))
        elapsed = time.monotonic() - start
        # Serial -> ~sum of latencies.
        self.assertGreaterEqual(elapsed, 0.45)

    def test_same_resource_key_serializes(self):
        # Two safe tools writing the same resource must not overlap.
        active = {"count": 0}
        lock = threading.Lock()
        overlapped = {"seen": False}

        def handler(args):
            with lock:
                active["count"] += 1
                if active["count"] > 1:
                    overlapped["seen"] = True
            time.sleep(0.2)
            with lock:
                active["count"] -= 1
            return {"content": [{"type": "text", "text": "ok"}]}

        registry = {
            "w": _FakeTool(handler, parallel_safe=True, resource_key="same/path"),
        }
        self.adapter._execute_tool_calls(registry, self._calls(["w", "w"]))
        self.assertFalse(overlapped["seen"], "same-resource calls overlapped")

    def test_results_preserve_order(self):
        registry = {
            "first": _FakeTool(
                lambda a: {"content": [{"type": "text", "text": "FIRST"}]},
                parallel_safe=True,
            ),
            "second": _FakeTool(
                lambda a: {"content": [{"type": "text", "text": "SECOND"}]},
                parallel_safe=True,
            ),
        }
        results = self.adapter._execute_tool_calls(registry, self._calls(["first", "second"]))
        self.assertEqual(results[0]["content"][0]["text"], "FIRST")
        self.assertEqual(results[1]["content"][0]["text"], "SECOND")


if __name__ == "__main__":
    unittest.main()
