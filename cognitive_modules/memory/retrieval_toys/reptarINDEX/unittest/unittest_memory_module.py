import unittest
from unittest.mock import patch, MagicMock
import asyncio
from memory_moduleMAKEOVER import MemorySystem, AbstractionLevel

class TestMemorySystem(unittest.TestCase):
    def setUp(self):
        self.memory_system = MemorySystem(AbstractionLevel.HIGH_LEVEL, "mock_index_file.json")

    def test_add_memory(self):
        memory = {"key": "value"}
        self.memory_system.add_memory(memory)
        self.assertIn(memory, self.memory_system.memories)

    @patch('memory_moduleMAKEOVER.load_graph')
    async def test_search_memories(self, mock_load_graph):
        mock_graph = MagicMock()
        mock_graph.search.return_value = [
            (MagicMock(text="Result 1", is_cluster=False), 0.9, 1, None),
            (MagicMock(text="Result 2", is_cluster=True), 0.8, 2, None)
        ]
        mock_load_graph.return_value = (mock_graph, None)

        self.memory_system.graph = mock_graph
        results = await self.memory_system.search_memories("test query")

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["text"], "Result 1")
        self.assertEqual(results[1]["text"], "Result 2")

    async def test_format_input_prompt(self):
        context = {
            "market_state": "Bullish",
            "recent_events": "Interest rate hike",
            "key_insights": "Increased volatility expected"
        }
        
        with patch.object(self.memory_system, 'search_memories', return_value=[]):
            prompt = await self.memory_system.format_input_prompt(context)

        self.assertIn("Role: Intelligent Economic Agent", prompt)
        self.assertIn("Market State: Bullish", prompt)
        self.assertIn("Recent Events: Interest rate hike", prompt)

    def test_format_output_prompt(self):
        action = "Buy 100 shares of XYZ"
        result = "Transaction successful"
        prompt = self.memory_system.format_output_prompt(action, result)

        self.assertIn("Action Executed: Buy 100 shares of XYZ", prompt)
        self.assertIn("Outcome: Transaction successful", prompt)

def run_async_test(coro):
    return asyncio.get_event_loop().run_until_complete(coro)

if __name__ == '__main__':
    unittest.main()