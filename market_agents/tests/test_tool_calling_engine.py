import unittest
from tool_caller.tool_calling_engine import ToolCallingEngine, FunctionCall

# Declare mock functions
def get_rain_probability(location: str) -> float:
    return 0.3

def get_current_temperature(location: str, unit: str) -> float:
    return 72.5

class TestToolCallingEngine(unittest.TestCase):
    def setUp(self):
        self.engine = ToolCallingEngine()
        self.engine.register_tools([get_rain_probability, get_current_temperature])

    def test_parse_tool_calls(self):
        tool_calls = [
            {
                "id": "call_FthC9qRpsL5kBpwwyw6c7j4k",
                "function": {
                    "arguments": '{"location": "San Francisco, CA"}',
                    "name": "get_rain_probability"
                },
                "type": "function"
            },
            {
                "id": "call_RpEDoB8O0FTL9JoKTuCVFOyR",
                "function": {
                    "arguments": '{"location": "San Francisco, CA", "unit": "Fahrenheit"}',
                    "name": "get_current_temperature"
                },
                "type": "function"
            }
        ]

        parsed_calls = self.engine.parse_tool_calls(tool_calls)

        self.assertEqual(len(parsed_calls), 2)
        self.assertIsInstance(parsed_calls[0], FunctionCall)
        self.assertIsInstance(parsed_calls[1], FunctionCall)

        self.assertEqual(parsed_calls[0].name, "get_rain_probability")
        self.assertEqual(parsed_calls[0].parameters, {"location": "San Francisco, CA"})

        self.assertEqual(parsed_calls[1].name, "get_current_temperature")
        self.assertEqual(parsed_calls[1].parameters, {"location": "San Francisco, CA", "unit": "Fahrenheit"})

    def test_process_tool_calls(self):

        tool_calls = [
            {
                "id": "call_FthC9qRpsL5kBpwwyw6c7j4k",
                "function": {
                    "arguments": '{"location": "San Francisco, CA"}',
                    "name": "get_rain_probability"
                },
                "type": "function"
            },
            {
                "id": "call_RpEDoB8O0FTL9JoKTuCVFOyR",
                "function": {
                    "arguments": '{"location": "San Francisco, CA", "unit": "Fahrenheit"}',
                    "name": "get_current_temperature"
                },
                "type": "function"
            }
        ]

        results = self.engine.process_tool_calls(tool_calls)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["tool_name"], "get_rain_probability")
        self.assertEqual(results[0]["tool_call_id"], "call_FthC9qRpsL5kBpwwyw6c7j4k")
        self.assertEqual(results[0]["result"], 0.3)

        self.assertEqual(results[1]["tool_name"], "get_current_temperature")
        self.assertEqual(results[1]["tool_call_id"], "call_RpEDoB8O0FTL9JoKTuCVFOyR")
        self.assertEqual(results[1]["result"], 72.5)

if __name__ == '__main__':
    unittest.main()