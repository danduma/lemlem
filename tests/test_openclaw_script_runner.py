import json
import unittest
from pathlib import Path

from lemlem.openclaw_skills import OpenClawRuntimeConfig, OpenClawSkillRef, load_skill_bundle
from lemlem.openclaw_skills.script_runner import run_skill_script


FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "openclaw_skills"


class OpenClawScriptRunnerTests(unittest.TestCase):
    def setUp(self):
        bundle = load_skill_bundle(
            OpenClawRuntimeConfig(
                skill_dirs=[str(FIXTURE_ROOT)],
                skills=[OpenClawSkillRef(id="exampleowner/script-skill")],
            )
        )
        self.skill = bundle.skills[0]
        self.scripts = {script.name: script for script in self.skill.scripts}

    def test_python_script_runs_with_arguments_and_stdin(self):
        payload = run_skill_script(
            skill=self.skill,
            script=self.scripts["echo_args"],
            arguments=["alpha", "beta"],
            stdin="from-stdin",
            timeout_seconds=10,
        )

        self.assertTrue(payload["ok"])
        data = json.loads(payload["stdout"])
        self.assertEqual(data["argv"], ["alpha", "beta"])
        self.assertEqual(data["stdin"], "from-stdin")

    def test_non_zero_exit_returns_structured_error(self):
        payload = run_skill_script(
            skill=self.skill,
            script=self.scripts["exit_fail"],
            arguments=[],
            stdin=None,
            timeout_seconds=10,
        )

        self.assertFalse(payload["ok"])
        self.assertEqual(payload["exit_code"], 3)
        self.assertEqual(payload["error"], "script_execution_failed")
        self.assertIn("fixture failure", payload["detail"])


if __name__ == "__main__":
    unittest.main()
