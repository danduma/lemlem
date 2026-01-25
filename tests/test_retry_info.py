import unittest

from lemlem import extract_retry_after_seconds


class TestRetryAfterExtraction(unittest.TestCase):
    def test_extract_retry_after_from_message(self) -> None:
        message = "Please retry in 5.866367787s."
        retry_after = extract_retry_after_seconds(message)
        self.assertIsNotNone(retry_after)
        self.assertAlmostEqual(retry_after or 0.0, 5.866367787, places=6)

    def test_extract_retry_after_from_retry_delay_detail(self) -> None:
        message = (
            "429 RESOURCE_EXHAUSTED. "
            "{'error': {'code': 429, 'message': 'Quota exceeded', "
            "'details': [{'@type': 'type.googleapis.com/google.rpc.RetryInfo', "
            "'retryDelay': '5s'}]}}"
        )
        retry_after = extract_retry_after_seconds(message)
        self.assertEqual(retry_after, 5.0)

    def test_extract_retry_after_from_retry_delay_value(self) -> None:
        message = "retryDelay: 250ms"
        retry_after = extract_retry_after_seconds(message)
        self.assertIsNotNone(retry_after)
        self.assertAlmostEqual(retry_after or 0.0, 0.25, places=3)


if __name__ == "__main__":
    unittest.main()
