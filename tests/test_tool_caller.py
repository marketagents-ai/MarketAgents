import unittest

from market_agents.agents.tool_caller import ToolCallingEngine
from market_agents.inference.message_models import GeneratedJsonObject

# Declare constants
EXAMPLE_JSON_OBJECT = GeneratedJsonObject(name="search", object={"query": "shirts"})

def search(query: str) -> str:
    return f"Search results for {query}"

class TestToolCallingEngine(unittest.TestCase):
    """
    Test the ToolCallingEngine class.
    """
    def setUp(self) -> None:
        self.engine = ToolCallingEngine()
        self.engine.add_functions([search])

    def test_parse_and_call_functions(self):
        """
        Test the parse_and_call_functions method.
        """
        results = self.engine.parse_and_call_functions(EXAMPLE_JSON_OBJECT)
        self.assertEqual(results, [search("shirts")])

    def test_get_trace(self):
        """
        Test the get_trace method.
        """
        results = self.engine.parse_and_call_functions(EXAMPLE_JSON_OBJECT)
        self.assertEqual(results, [search("shirts")])
        trace = self.engine.get_trace()
        self.assertEqual(trace, [
            {
                "function": "search",
                "arguments": {"query": "shirts"},
                "result": "Search results for shirts"
            }
        ])

if __name__ == '__main__':
    unittest.main()
