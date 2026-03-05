import unittest
from pathlib import Path

from lemlem.openclaw_skills import OpenClawRuntimeConfig, OpenClawSkillRef, load_skill_bundle
from lemlem.openclaw_skills.prompting import build_prompt_prefix


FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "openclaw_skills"


class OpenClawPromptingTests(unittest.TestCase):
    def test_prompt_includes_skill_metadata(self):
        bundle = load_skill_bundle(
            OpenClawRuntimeConfig(
                skill_dirs=[str(FIXTURE_ROOT)],
                skills=[
                    OpenClawSkillRef(id="exampleowner/script-skill"),
                    OpenClawSkillRef(id="exampleowner/markdown-skill"),
                ],
            )
        )
        bundle.skills[0].generated_tool_names.append("openclaw__exampleowner__script_skill__echo_args")
        prompt = build_prompt_prefix(bundle)

        self.assertIn("Script Skill", prompt)
        self.assertIn("FIXTURE_API_KEY", prompt)
        self.assertIn("openclaw__exampleowner__script_skill__echo_args", prompt)

    def test_prompt_respects_budget(self):
        bundle = load_skill_bundle(
            OpenClawRuntimeConfig(
                skill_dirs=[str(FIXTURE_ROOT)],
                skills=[OpenClawSkillRef(id="exampleowner/script-skill")],
                prompt_char_budget=200,
            )
        )
        prompt = build_prompt_prefix(bundle)
        self.assertLessEqual(len(prompt), 200)


if __name__ == "__main__":
    unittest.main()
