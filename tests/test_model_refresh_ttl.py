import unittest
from unittest import mock

from lemlem import adapter
from lemlem.adapter import ExternalModelDataResolver


class TestModelRefreshTTL(unittest.TestCase):
    def setUp(self) -> None:
        # Snapshot module globals so each test runs in isolation.
        self._saved = (
            adapter._EXTERNAL_RESOLVER,
            adapter._LAST_REFRESH_CHECK,
            adapter._MODEL_DATA_TIMESTAMP,
            adapter._REFRESH_TTL_S,
        )

    def tearDown(self) -> None:
        (
            adapter._EXTERNAL_RESOLVER,
            adapter._LAST_REFRESH_CHECK,
            adapter._MODEL_DATA_TIMESTAMP,
            adapter._REFRESH_TTL_S,
        ) = self._saved

    def test_get_timestamp_called_at_most_once_within_ttl(self) -> None:
        get_timestamp = mock.Mock(return_value=1)
        ensure_configured = mock.Mock()
        load_bundle = mock.Mock(return_value={"models": {}, "configs": {}, "env_vars": {}})
        adapter._EXTERNAL_RESOLVER = ExternalModelDataResolver(
            load_bundle=load_bundle,
            get_timestamp=get_timestamp,
            ensure_configured=ensure_configured,
        )
        adapter._REFRESH_TTL_S = 30.0
        adapter._LAST_REFRESH_CHECK = 0.0
        adapter._MODEL_DATA_TIMESTAMP = 1

        # Freeze monotonic clock so all calls land inside the TTL window.
        with mock.patch.object(adapter.time, "monotonic", return_value=100.0):
            for _ in range(10):
                adapter._refresh_model_data()

        # Only the first call within the window polls the DB-backed timestamp.
        self.assertEqual(get_timestamp.call_count, 1)
        self.assertEqual(ensure_configured.call_count, 1)

    def test_force_bypasses_ttl(self) -> None:
        get_timestamp = mock.Mock(return_value=1)
        load_bundle = mock.Mock(return_value={"models": {}, "configs": {}, "env_vars": {}})
        adapter._EXTERNAL_RESOLVER = ExternalModelDataResolver(
            load_bundle=load_bundle,
            get_timestamp=get_timestamp,
            ensure_configured=None,
        )
        adapter._REFRESH_TTL_S = 30.0
        adapter._LAST_REFRESH_CHECK = 0.0
        adapter._MODEL_DATA_TIMESTAMP = 1

        with mock.patch.object(adapter.time, "monotonic", return_value=100.0):
            adapter._refresh_model_data(force=True)
            adapter._refresh_model_data(force=True)

        self.assertEqual(get_timestamp.call_count, 2)

    def test_check_runs_again_after_ttl_elapses(self) -> None:
        get_timestamp = mock.Mock(return_value=1)
        load_bundle = mock.Mock(return_value={"models": {}, "configs": {}, "env_vars": {}})
        adapter._EXTERNAL_RESOLVER = ExternalModelDataResolver(
            load_bundle=load_bundle,
            get_timestamp=get_timestamp,
            ensure_configured=None,
        )
        adapter._REFRESH_TTL_S = 30.0
        adapter._LAST_REFRESH_CHECK = 0.0
        adapter._MODEL_DATA_TIMESTAMP = 1

        with mock.patch.object(adapter.time, "monotonic", side_effect=[100.0, 140.0]):
            adapter._refresh_model_data()
            adapter._refresh_model_data()

        self.assertEqual(get_timestamp.call_count, 2)


if __name__ == "__main__":
    unittest.main()
