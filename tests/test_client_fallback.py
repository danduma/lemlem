import time
import unittest

from lemlem.client import LLMClient


class LLMClientFallbackTests(unittest.TestCase):
    def setUp(self) -> None:
        # Minimal model/config setup to avoid any network usage
        self.model_data = {
            "models": {
                "m1": {"model_name": "m1", "base_url": None, "api_key": "k1", "meta": {}},
                "m2": {"model_name": "m2", "base_url": None, "api_key": "k2", "meta": {}},
            },
            "configs": {
                "c1": {
                    "model": "m1",
                    "models": ["m1", "m2"],
                    "keys": [{"key": "k1", "max_rpm": 1, "cooldown_seconds": 0.1}, {"key": "k2"}],
                    "enabled": True,
                    "key_strategy": "sequential_on_failure",
                }
            },
        }
        self.client = LLMClient(self.model_data)

    def test_normalize_key_entries_falls_back_to_api_key(self) -> None:
        entries = self.client._normalize_key_entries(
            keys_field=None, fallback_key="fallback", max_rpm_default=2, cooldown_default=1.5
        )
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["key"], "fallback")
        self.assertEqual(entries[0]["max_rpm"], 2)
        self.assertEqual(entries[0]["cooldown_seconds"], 1.5)

    def test_rpm_limit_and_cooldown_skip_busy_key(self) -> None:
        model_key = "c1:m1"
        keys = self.client._normalize_key_entries(
            keys_field=self.model_data["configs"]["c1"]["keys"],
            fallback_key=None,
            max_rpm_default=None,
            cooldown_default=None,
        )

        # Mark first key as just used and on cooldown
        now_ts = time.time()
        self.client._record_usage(model_key, 0, now_ts)
        self.client._apply_cooldown(model_key, 0, backoff_base=0.1, backoff_max=0.1)

        # Should pick second key due to rpm/cooldown on first
        idx = self.client._choose_key_index(model_key, keys, "sequential_on_failure")
        self.assertEqual(idx, 1)

    def test_resolve_config_injects_models_and_enabled(self) -> None:
        cfg = self.client._resolve_config("c1", model_override="m2")
        self.assertTrue(cfg["enabled"])
        self.assertTrue(cfg["_model_enabled"])
        self.assertEqual(cfg["models"], ["m1", "m2"])
        self.assertEqual(cfg["_keys"][0]["key"], "k1")
        self.assertEqual(cfg["_keys"][1]["key"], "k2")


if __name__ == "__main__":
    unittest.main()

