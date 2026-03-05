import asyncio
import json
from pathlib import Path
import unittest

from lemlem.openclaw_skills import MCPServerConfig, OpenClawRuntimeConfig, OpenClawSkillRef, build_tools_and_prompt
from lemlem.openclaw_skills.mcp_bridge import MCP_IMPORT_ERROR


FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "openclaw_skills"
FIXTURE_SERVER = Path(__file__).resolve().parent / "fixtures" / "fixture_mcp_server.py"


@unittest.skipIf(MCP_IMPORT_ERROR is not None, f"MCP client imports unavailable: {MCP_IMPORT_ERROR}")
class OpenClawMCPBridgeTests(unittest.TestCase):
    def test_mcp_tools_are_discovered_and_callable(self):
        augmentation = build_tools_and_prompt(
            OpenClawRuntimeConfig(
                skill_dirs=[str(FIXTURE_ROOT)],
                skills=[OpenClawSkillRef(id="exampleowner/mcp-skill")],
                mcp_servers={
                    "fixturemcp": MCPServerConfig(
                        transport="stdio",
                        command="uv",
                        args=["run", "python", str(FIXTURE_SERVER)],
                    )
                },
            )
        )

        tool = next(spec for spec in augmentation.tool_specs if spec.name.endswith("__mcp__add_numbers"))
        payload = asyncio.run(tool.handler({"a": 2, "b": 5}))
        self.assertTrue(payload["ok"])
        result = payload["result"]
        structured = result.get("structuredContent")
        if structured is None:
            structured = json.loads(result["content"][0]["text"])
        self.assertEqual(structured["sum"], 7)
        asyncio.run(augmentation.mcp_manager.aclose())


if __name__ == "__main__":
    unittest.main()
