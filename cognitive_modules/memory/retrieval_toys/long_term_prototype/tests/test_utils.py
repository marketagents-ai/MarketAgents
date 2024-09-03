import unittest
import tempfile
import os
from long_term_memory.utils import format_memory_for_prompt, format_memories_for_prompt, load_json_file, save_json_file

class TestUtils(unittest.TestCase):
    def test_format_memory_for_prompt(self):
        memory = {
            "content": "Test content",
            "metadata": {"category": "test"},
            "relevance_score": 0.85
        }
        formatted = format_memory_for_prompt(memory)
        self.assertIn("Test content", formatted)
        self.assertIn("category: test", formatted)
        self.assertIn("relevance: 0.85", formatted)

    def test_format_memories_for_prompt(self):
        memories = [
            {"content": "Memory 1", "relevance_score": 0.9},
            {"content": "Memory 2", "relevance_score": 0.8},
            {"content": "Memory 3", "relevance_score": 0.7}
        ]
        formatted = format_memories_for_prompt(memories, max_memories=2)
        self.assertIn("Memory 1", formatted)
        self.assertIn("Memory 2", formatted)
        self.assertNotIn("Memory 3", formatted)

    def test_load_and_save_json_file(self):
        test_data = {"key": "value"}
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            save_json_file(test_data, temp_file.name)
            loaded_data = load_json_file(temp_file.name)
        self.assertEqual(test_data, loaded_data)
        os.unlink(temp_file.name)

if __name__ == '__main__':
    unittest.main()