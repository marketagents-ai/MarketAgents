import unittest
import tempfile
import os
from long_term_memory import LongTermMemory, Config

class TestLongTermMemory(unittest.TestCase):
    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.config = Config(db_path=self.temp_db.name)
        self.ltm = LongTermMemory(self.config)

    def tearDown(self):
        self.ltm.close()
        os.unlink(self.temp_db.name)

    def test_add_and_get_memory(self):
        memory_id = self.ltm.add_memory("Test memory", {"category": "test"})
        memories = self.ltm.get_memories(limit=1)
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0]["content"], "Test memory")
        self.assertEqual(memories[0]["metadata"], {"category": "test"})

    def test_search_memories(self):
        self.ltm.add_memory("Apple is a fruit", {"category": "food"})
        self.ltm.add_memory("Python is a programming language", {"category": "technology"})
        results = self.ltm.search_memories("fruit apple")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["content"], "Apple is a fruit")

    def test_forget_memories(self):
        self.ltm.add_memory("Old memory", {"category": "old"})
        self.config.forget_threshold_days = 0
        self.config.forget_access_threshold = 2
        self.ltm.forget_memories()
        memories = self.ltm.get_memories()
        self.assertEqual(len(memories), 0)

if __name__ == '__main__':
    unittest.main()