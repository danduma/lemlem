import asyncio
from pathlib import Path
import unittest

from lemlem.cyber_agent.agent import CyberAgent
from lemlem.cyber_agent.config import AgentConfig
from lemlem.openclaw_skills import OpenClawRuntimeConfig, OpenClawSkillRef


FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "openclaw_skills"


class FakeAdapter:
    def __init__(self):
        self.calls = []

    def chat_json(
        self,
        system_prompt,
        user_payload,
        *,
        model,
        temperature,
        tools,
        max_tool_iterations,
        on_turn,
        logging_context,
    ):
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_payload": user_payload,
                "model": model,
                "tools": tools,
                "logging_context": logging_context,
            }
        )
        return {"final_text": '{"status":"ok"}', "usage": None}


class OpenClawAgentIntegrationTests(unittest.TestCase):
    def test_cyber_agent_augments_prompt_and_tools(self):
        adapter = FakeAdapter()
        agent = CyberAgent(
            config=AgentConfig(
                agent_id="agent_1",
                system_prompt="Base system prompt",
                model="fake-model",
                openclaw_runtime=OpenClawRuntimeConfig(
                    skill_dirs=[str(FIXTURE_ROOT)],
                    skills=[OpenClawSkillRef(id="exampleowner/script-skill")],
                ),
            ),
            adapter=adapter,
        )

        self.assertIn("OpenClaw skills are configured", agent.config.system_prompt)
        tool_names = [tool.name for tool in agent.config.tools]
        self.assertIn("openclaw_skill_help", tool_names)
        self.assertIn("openclaw__exampleowner__script_skill__echo_args", tool_names)

        help_tool = next(tool for tool in agent.config.tools if tool.name == "openclaw_skill_help")
        help_payload = help_tool.handler({"skill_id": "exampleowner/script-skill"})
        self.assertTrue(help_payload["ok"])
        self.assertEqual(help_payload["name"], "Script Skill")

        async def run_chat():
            events = []
            async for event in agent.stream_chat(conversation_id="conv_1", message="hello"):
                events.append(event)
            return events

        events = asyncio.run(run_chat())
        self.assertTrue(any(event["type"] == "message" for event in events))
        self.assertTrue(adapter.calls)
        self.assertGreaterEqual(len(adapter.calls[0]["tools"]), 2)


if __name__ == "__main__":
    unittest.main()
