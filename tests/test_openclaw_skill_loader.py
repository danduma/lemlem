import unittest
from pathlib import Path

from lemlem.openclaw_skills import OpenClawRuntimeConfig, OpenClawSkillRef, load_skill_bundle


FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "openclaw_skills"


class OpenClawSkillLoaderTests(unittest.TestCase):
    def test_loads_skill_bundle_from_local_dir(self):
        bundle = load_skill_bundle(
            OpenClawRuntimeConfig(
                skill_dirs=[str(FIXTURE_ROOT)],
                skills=[OpenClawSkillRef(id="exampleowner/script-skill")],
            )
        )

        self.assertEqual(len(bundle.skills), 1)
        skill = bundle.skills[0]
        self.assertEqual(skill.id, "exampleowner/script-skill")
        self.assertEqual(skill.version, "1.2.3")
        self.assertIn("FIXTURE_API_KEY", skill.env_vars)
        self.assertTrue(skill.scripts)
        self.assertIn("usage", skill.sections)

    def test_optional_missing_skill_is_skipped(self):
        bundle = load_skill_bundle(
            OpenClawRuntimeConfig(
                skill_dirs=[str(FIXTURE_ROOT)],
                skills=[OpenClawSkillRef(id="exampleowner/missing-skill", required=False)],
            )
        )
        self.assertEqual(bundle.skills, [])

    def test_discovers_markdown_referenced_script(self):
        bundle = load_skill_bundle(
            OpenClawRuntimeConfig(
                skill_dirs=[str(FIXTURE_ROOT)],
                skills=[OpenClawSkillRef(id="exampleowner/script-skill", enabled_scripts=["echo_args"])],
            )
        )

        skill = bundle.skills[0]
        self.assertEqual([script.name for script in skill.scripts], ["echo_args"])


if __name__ == "__main__":
    unittest.main()
