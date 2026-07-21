import unittest
from pathlib import Path
from unittest.mock import patch

from lemlem.codex_app_server import _ProtocolClient


class CodexAppServerPersistenceTestCase(unittest.TestCase):
    def test_starts_background_threads_as_ephemeral(self) -> None:
        client = _ProtocolClient()

        with patch.object(
            client,
            "request",
            return_value={"thread": {"id": "thread-1"}},
        ) as request:
            client.start_thread(Path("/tmp"), "gpt-5.6-luna")

        params = request.call_args.args[1]
        self.assertIs(params["ephemeral"], True)


if __name__ == "__main__":
    unittest.main()
