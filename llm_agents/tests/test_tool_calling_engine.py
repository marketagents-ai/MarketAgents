import unittest
from tool_calling_engine import ToolCallingEngine, FunctionCall

class TestToolCallingEngine(unittest.TestCase):
    def setUp(self):
        self.engine = ToolCallingEngine()

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
        # Mock the execute_tool_call method to return a predefined result
        def mock_execute_tool_call(tool_call):
            if tool_call.name == "get_rain_probability":
                return 0.3
            elif tool_call.name == "get_current_temperature":
                return 72.5

        self.engine.execute_tool_call = mock_execute_tool_call

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