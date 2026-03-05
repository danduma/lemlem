import unittest
from pathlib import Path

from lemlem.skills import SkillRuntimeConfig, SkillRef, load_skill_bundle
from lemlem.skills.prompting import build_prompt_prefix


FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "skills"


class SkillPromptingTests(unittest.TestCase):
    def test_prompt_includes_skill_metadata(self):
        bundle = load_skill_bundle(
            SkillRuntimeConfig(
                skill_dirs=[str(FIXTURE_ROOT)],
                skills=[
                    SkillRef(id="exampleowner/script-skill"),
                    SkillRef(id="exampleowner/markdown-skill"),
                ],
            )
        )
        bundle.skills[0].generated_tool_names.append("skill__exampleowner__script_skill__echo_args")
        prompt = build_prompt_prefix(bundle)

        self.assertIn("Script Skill", prompt)
        self.assertIn("FIXTURE_API_KEY", prompt)
        self.assertIn("skill__exampleowner__script_skill__echo_args", prompt)

    def test_prompt_respects_budget(self):
        bundle = load_skill_bundle(
            SkillRuntimeConfig(
                skill_dirs=[str(FIXTURE_ROOT)],
                skills=[SkillRef(id="exampleowner/script-skill")],
                prompt_char_budget=200,
            )
        )
        prompt = build_prompt_prefix(bundle)
        self.assertLessEqual(len(prompt), 200)


if __name__ == "__main__":
    unittest.main()
