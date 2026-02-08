import unittest
from types import SimpleNamespace

from lemlem.gemini_wrapper import GeminiWrapper


class _FunctionDeclaration:
    def __init__(self, name, description, parameters):
        self.name = name
        self.description = description
        self.parameters = parameters


class _Tool:
    def __init__(self, function_declarations):
        self.function_declarations = function_declarations


class TestGeminiSchemaSanitization(unittest.TestCase):
    def setUp(self):
        self.wrapper = GeminiWrapper.__new__(GeminiWrapper)
        self.wrapper._types = SimpleNamespace(
            Tool=_Tool,
            FunctionDeclaration=_FunctionDeclaration,
        )

    def test_type_list_is_collapsed_to_single_type(self):
        schema = {
            "type": "object",
            "properties": {
                "states": {
                    "type": ["array", "string"],
                    "items": {"type": "string"},
                }
            },
        }
        sanitized = self.wrapper._sanitize_json_schema(schema)
        self.assertEqual(sanitized["properties"]["states"]["type"], "string")
        self.assertNotIsInstance(
            sanitized["properties"]["states"]["type"], list
        )
        self.assertNotIn("items", sanitized["properties"]["states"])

    def test_oneof_is_collapsed(self):
        schema = {
            "type": "object",
            "properties": {
                "stages": {
                    "oneOf": [
                        {"type": "array", "items": {"type": "string"}},
                        {"type": "string"},
                    ],
                    "description": "Stages filter.",
                }
            },
        }
        sanitized = self.wrapper._sanitize_json_schema(schema)
        stages = sanitized["properties"]["stages"]
        self.assertNotIn("oneOf", stages)
        self.assertEqual(stages["type"], "array")

    def test_convert_tools_sanitizes_union_schema(self):
        tools = [
            {
                "type": "function",
                "name": "list_mission_objectives",
                "description": "List mission objectives",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "states": {
                            "type": ["array", "string"],
                            "items": {"type": "string"},
                        },
                        "objective_id": {
                            "oneOf": [{"type": "integer"}, {"type": "string"}],
                        },
                    },
                },
            }
        ]
        gemini_tools = self.wrapper.convert_openai_tools_to_gemini(tools)
        self.assertEqual(len(gemini_tools), 1)
        decl = gemini_tools[0].function_declarations[0]
        states_schema = decl.parameters["properties"]["states"]
        objective_schema = decl.parameters["properties"]["objective_id"]
        self.assertEqual(states_schema["type"], "string")
        self.assertNotIn("oneOf", objective_schema)
        self.assertEqual(objective_schema["type"], "integer")


if __name__ == "__main__":
    unittest.main()
